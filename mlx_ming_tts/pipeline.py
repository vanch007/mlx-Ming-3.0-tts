"""End-to-end pure MLX text-to-speech pipeline entry point.

Loads model weights (native MLX safetensors or converted checkpoint), runs
generation purely on MLX, and writes standard 44.1kHz 16-bit PCM WAV using
the Python standard library 'wave'. Zero dependency on torch, torchaudio,
soundfile, or numpy for synthesis.
"""

import argparse
import json
import os
import re
import sys
import wave

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from mlx_ming_tts.config import load_config, get_llm_config
from mlx_ming_tts.models.bailingmm import BailingMM
from mlx_ming_tts.models.lingv3_tts import LingV3TTS


DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"


def split_text_fragments(text, max_len=60):
    if len(text) <= max_len or len(text) <= 50:
        return [text]
    """Split long text into coherent semantic fragments to prevent AR drift."""
    pattern = r'([。！？.!?\n]+)'
    raw = [s.strip() for s in re.split(pattern, text) if s.strip()]
    merged = []
    i = 0
    while i < len(raw):
        if i + 1 < len(raw) and re.match(pattern, raw[i+1]):
            merged.append(raw[i] + raw[i+1])
            i += 2
        else:
            merged.append(raw[i])
            i += 1

    final_frags = []
    for s in merged:
        if len(s) > max_len and ('，' in s or ',' in s):
            parts = re.split(r'([，,;；]+)', s)
            curr = ''
            for p in parts:
                if not p: continue
                curr += p
                if len(curr) >= 15 and re.match(r'([，,;；]+)', p):
                    final_frags.append(curr.strip())
                    curr = ''
            if curr.strip():
                final_frags.append(curr.strip())
        else:
            final_frags.append(s)
    return [f for f in final_frags if f.strip()]


def build_ip_instruction(ip_name, is_chinese=True):
    """Build structured IP voice instruction."""
    if is_chinese:
        fields = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
        values = (1, "speaker_1", None, None, None, None, None, None)
        item = dict(zip(fields, values))
        item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
        item["IP"] = ip_name
    else:
        fields = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion")
        values = (1, "speaker_1", None, None, None, None, None, None)
        item = dict(zip(fields, values))
        item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
        item["ip"] = ip_name
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def build_voice_design_instruction(style_desc, is_chinese=True):
    """Build structured voice design instruction."""
    if is_chinese:
        fields = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
        values = (1, "speaker_1", None, style_desc, None, None, None, None)
        item = dict(zip(fields, values))
        item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
        item["IP"] = None
    else:
        fields = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion")
        values = (1, "speaker_1", None, style_desc, None, None, None, None)
        item = dict(zip(fields, values))
        item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
        item["ip"] = None
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def trim_trailing_silence(speech, sample_rate=44100, threshold=1e-3, tail_seconds=0.2, frame_seconds=0.1):
    """Trim excessive trailing silence while preserving a natural tail."""
    frame = int(sample_rate * frame_seconds)
    tail = int(sample_rate * tail_seconds)
    if speech.shape[0] < frame:
        return speech[:tail]

    num_frames = speech.shape[0] // frame
    truncated = speech[:num_frames * frame]
    frames = truncated.reshape(num_frames, frame)
    scores = mx.mean(mx.abs(frames), axis=-1)

    scores_list = scores.tolist()
    index = len(scores_list) - 1
    while index >= 0 and scores_list[index] <= threshold:
        index -= 1
    if index < 0:
        return speech[:tail]
    keep = min(speech.shape[0], (index + 1) * frame + tail)
    return speech[:keep]


def resolve_dtype(model_dir, dtype="auto"):
    """Resolve compute/load dtype. MoE defaults to bf16 to fit comfortably."""
    if dtype != "auto":
        return {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
    llm_cfg = get_llm_config(load_config(model_dir))
    return mx.bfloat16 if llm_cfg.get("num_experts") else mx.float32


def metadata_path(model_dir):
    return os.path.join(model_dir, "mlx_model_config.json")


def load_mlx_metadata(model_dir):
    path = metadata_path(model_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_wav(audio, path, sample_rate=44100):
    """Save a 1-D MLX float array as a 16-bit PCM WAV file via standard library 'wave'."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio = mx.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(mx.int16)
    pcm_bytes = bytes(audio_int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def load_single_resampled_wav(wav_path, target_sr=44100):
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        data = wf.readframes(n)

    if sampwidth == 2:
        import struct
        fmt = f"<{n * channels}h"
        raw = struct.unpack(fmt, data)
        raw_arr = [float(x) / 32768.0 for x in raw]
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        raw_arr = [sum(raw_arr[i*channels : (i+1)*channels]) / channels for i in range(n)]

    wav_mx = mx.array(raw_arr, dtype=mx.float32)

    if sr != target_sr:
        target_len = int(len(raw_arr) * target_sr / sr)
        src_idx = mx.linspace(0, len(raw_arr) - 1, target_len)
        src_left = mx.floor(src_idx).astype(mx.int32)
        src_right = mx.minimum(src_left + 1, len(raw_arr) - 1)
        src_frac = src_idx - src_left.astype(mx.float32)
        wav_mx = wav_mx[src_left] * (1.0 - src_frac) + wav_mx[src_right] * src_frac
    return wav_mx[None, :]

def load_and_resample_prompt(wav_path_or_paths, target_sr=44100):
    """Pure MLX / standard wave loader and resampler supporting single or multiple reference audios."""
    if isinstance(wav_path_or_paths, (list, tuple)):
        parts = [load_single_resampled_wav(p, target_sr) for p in wav_path_or_paths]
        wav_mx = mx.concatenate(parts, axis=1)
    else:
        wav_mx = load_single_resampled_wav(wav_path_or_paths, target_sr)

    align = int(1 / 12.5 * 1 * target_sr)
    rem = wav_mx.shape[-1] % align
    if rem:
        wav_mx = mx.concatenate([wav_mx, mx.zeros((1, align - rem))], axis=1)
    return wav_mx


class MingMLX:
    def __init__(self, model_dir, dtype="auto"):
        self.model_dir = model_dir
        metadata = load_mlx_metadata(model_dir)
        if dtype == "auto" and metadata is not None:
            dtype = metadata.get("dtype", dtype)
        self.dtype = resolve_dtype(model_dir, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        is_tiny = metadata is not None and metadata.get("model_type") == "lingv3_tts"
        if is_tiny:
            print("Detected Ming-3.0-tts-tiny (LingV3TTS MoE+KDA) model architecture.")
            self.model = LingV3TTS(model_dir, self.tokenizer)
        else:
            self.model = BailingMM(model_dir, self.tokenizer)

        if metadata is not None:
            # Check if weights are prequantized
            q_meta = metadata.get("quantization")
            if q_meta:
                q_bits = q_meta.get("bits", 8)
                q_group = q_meta.get("group_size", 64)
                print(f"Applying {q_bits}-bit quantization structure (group_size={q_group})...")
                def predicate(path, m):
                    if not isinstance(m, nn.Linear):
                        return False
                    if any(k in path for k in ('head_out', 'stop_head', 'spk_head')):
                        return False
                    if m.weight.shape[-1] % q_group != 0:
                        return False
                    return True
                nn.quantize(self.model, group_size=q_group, bits=q_bits, class_predicate=predicate)

            weights_file = metadata.get("weights_file", "mlx_weights.safetensors")
            weights_path = os.path.join(model_dir, weights_file)
            print(f"Loading preconverted MLX weights from {weights_path}...")
            self.model.load_mlx_weights(weights_path)
        else:
            print(f"Loading weights from HF directory {model_dir}...")
            self.model.load_all_weights(dtype=self.dtype)
        self.sample_rate = self.model.atc.get("sample_rate", 44100)

    def generate(self, prompt, text, instruction=None, ip=None, style=None,
                 ref_audio=None, prompt_text=None,
                 sigma=0.15, cfg=2.0, steps=10, max_decode_steps=200,
                 noise_seed=42, trim_silence=True, split_long_text=True, **kwargs):
        """Generate speech waveform purely in MLX with quality-preserving defaults."""
        is_chinese = bool(re.search(r'[一-龥]', text))
        prompt_waveform = None

        if ref_audio is not None and prompt_text is not None:
            prompt_waveform = load_and_resample_prompt(ref_audio, self.sample_rate)
            if instruction is None and style is not None:
                instruction = build_voice_design_instruction(style, is_chinese=is_chinese)
        elif instruction is None:
            if ip is not None:
                instruction = build_ip_instruction(ip, is_chinese=is_chinese)
            elif style is not None:
                instruction = build_voice_design_instruction(style, is_chinese=is_chinese)
            else:
                default_ip = "灵小萌" if is_chinese else "US_Emma"
                instruction = build_ip_instruction(default_ip, is_chinese=is_chinese)

        if split_long_text and (prompt_waveform is None) and len(text) > 60:
            fragments = split_text_fragments(text, max_len=30)
        else:
            fragments = [text]

        chunks = []
        for i, frag in enumerate(fragments):
            seed = (noise_seed + i) if noise_seed is not None else None
            latents = []
            for sampled, last in self.model.sample(
                prompt, frag, instruction=instruction,
                prompt_waveform=prompt_waveform,
                prompt_text=prompt_text,
                max_decode_steps=max_decode_steps, cfg=cfg, sigma=sigma,
                steps=steps, noise_seed=seed, **kwargs
            ):
                latents.append(sampled)
            full = mx.concatenate(latents, axis=1)
            chunk_wav = self.model.audio.decode_full(full)
            if trim_silence:
                chunk_wav = trim_trailing_silence(chunk_wav, self.sample_rate)
            chunks.append(chunk_wav)

        return mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def main():
    ap = argparse.ArgumentParser(description="Pure MLX Ming-3.0-tts generation")
    ap.add_argument("--model-dir", default="models/Ming-3.0-tts-0.5b-mlx")
    ap.add_argument("--prompt", default=DEFAULT_TTS_PROMPT)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ip", default=None, help="Built-in IP voice name (e.g. 灵小萌, 灵小锋, 灵小柔, US_Emma, UK_George)")
    ap.add_argument("--style", default=None, help="Voice style prompt")
    ap.add_argument("--ref-audio", default=None, help="Path to reference WAV file for voice cloning")
    ap.add_argument("--prompt-text", default=None, help="Transcript of reference WAV file")
    ap.add_argument("--out", default="output/mlx_out.wav")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = ap.parse_args()

    ming = MingMLX(args.model_dir, dtype=args.dtype)
    wav = ming.generate(
        args.prompt, args.text, ip=args.ip, style=args.style,
        ref_audio=args.ref_audio, prompt_text=args.prompt_text,
        max_decode_steps=args.max_steps, cfg=args.cfg, sigma=args.sigma,
        noise_seed=args.seed
    )
    save_wav(wav, args.out, ming.sample_rate)
    print(f"Successfully generated {len(wav)} samples ({len(wav)/ming.sample_rate:.2f}s) @ {ming.sample_rate}Hz -> {args.out}")


if __name__ == "__main__":
    main()
