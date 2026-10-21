
import json, os, time, sys, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_same_cases_comparison.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    cases = json.load(f)

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_audio'
os.makedirs(OUT_DIR, exist_ok=True)

TINY_MODELS = [
    {"key": "tiny_8bit", "name": "Tiny 8-bit (推荐主力)", "dir": "models/Ming-3.0-tts-tiny-mlx-8bit"},
    {"key": "tiny_4bit", "name": "Tiny 4-bit (超轻量)", "dir": "models/Ming-3.0-tts-tiny-mlx-4bit"},
    {"key": "tiny_bf16", "name": "Tiny bf16 (全精度)", "dir": "models/Ming-3.0-tts-tiny-mlx-bf16"},
]

for m in TINY_MODELS:
    mkey = m["key"]
    mname = m["name"]
    mdir = m["dir"]
    print(f"\n==========================================")
    print(f"Loading {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, "mlx_weights.safetensors")
    size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    runner = MingMLX(mdir)
    
    for idx, c in enumerate(cases):
        cid = c["case_id"]
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(OUT_DIR, out_fname)
        
        prompt = c.get("prompt", DEFAULT_TTS_PROMPT)
        text = c["text"]
        ip = c.get("ip")
        style = c.get("style")
        
        # Determine language
        target_lang = "Chinese"
        if "ja" in cid: target_lang = "Japanese"
        elif "ko" in cid: target_lang = "Korean"
        elif "fr" in cid: target_lang = "French"
        elif "en" in cid or "james" in cid: target_lang = "English"
        
        # Audio prompt
        ref_audio = None
        prompt_text = None
        if cid in ("case01_clone_zh", "case02_paralinguistic_zh", "case03_paralinguistic_en", "case04_multilingual_ja", "case05_multilingual_ko", "case06_multilingual_fr", "case15_podcast"):
            ref_audio = PROMPT_WAV
            prompt_text = PROMPT_TEXT
            
        instr_str = None
        if "fast" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": None, "风格": None, "语速": "快速", "基频": "中", "音量": "中", "情感": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
        elif "sad" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": None, "风格": None, "语速": None, "基频": None, "音量": None, "情感": "悲伤", "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
        elif "sichuan" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": "川渝话", "风格": None, "语速": None, "基频": None, "音量": None, "情感": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
        elif "james" in cid:
            instr_str = json.dumps({"audio_sequence": [{"id": 1, "speaker": "speaker_1", "dialect": None, "style": None, "speed": None, "pitch": None, "volume": None, "emotion": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "ip": "US_James"}]}, ensure_ascii=False)
        elif "lingxiaomeng" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": None, "风格": None, "语速": None, "基频": None, "音量": None, "情感": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": "灵小萌"}]}, ensure_ascii=False)
            
        t0 = time.perf_counter()
        wav = runner.generate(
            prompt,
            text,
            instruction=instr_str,
            ip=ip,
            style=style,
            ref_audio=ref_audio,
            prompt_text=prompt_text,
            target_language=target_lang,
            max_decode_steps=75,
            sigma=0.15,
            cfg=2.0
        )
        mx.eval(wav)
        elapsed = time.perf_counter() - t0
        
        save_wav(wav, out_path, runner.sample_rate)
        duration = len(wav) / runner.sample_rate
        rtf = elapsed / duration
        
        with open(out_path, "rb") as f:
            b64 = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('ascii')}"
            
        c["tiny_moe"][mkey] = {
            "name": mname,
            "weight_size_mb": size_mb,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "b64": b64
        }
        print(f"  [{cid}] Elapsed: {elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

with open(DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(cases, f, indent=2, ensure_ascii=False)

print("\nFull 100% matched regeneration finished!")
