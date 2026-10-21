
import soundfile as sf
import mlx.core as mx
import numpy as np
import time, os, sys

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

ref_path = '/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav'
wav, sr = sf.read(ref_path)
if wav.ndim > 1: wav = wav.mean(axis=1)

target_sr = 44100
target_len = int(len(wav) * target_sr / sr)
src_idx = mx.linspace(0, len(wav)-1, target_len)
src_left = mx.floor(src_idx).astype(mx.int32)
src_right = mx.minimum(src_left + 1, len(wav) - 1)
src_frac = src_idx - src_left.astype(mx.float32)

wav_mx = mx.array(wav.astype(np.float32))
resampled = wav_mx[src_left] * (1.0 - src_frac) + wav_mx[src_right] * src_frac
resampled = resampled[None, :]

align = int(1 / 12.5 * 1 * 44100)
new_len = (resampled.shape[-1] + align - 1) // align * align
if new_len > resampled.shape[-1]:
    resampled = mx.concatenate([resampled, mx.zeros((1, new_len - resampled.shape[-1]))], axis=1)

ming = MingMLX('models/Ming-3.0-tts-0.5b-mlx')

prompt_text = '冬天对于田野和河流来说，是一个放松休息的季节。'
target_text = '请用参考音色自然地说出这句话：测试需要同时关注内容、音色和稳定性。'

t0 = time.perf_counter()
wav_out = ming.model.generate(
    DEFAULT_TTS_PROMPT,
    target_text,
    prompt_waveform=resampled,
    prompt_text=prompt_text,
    sigma=0.15,
    cfg=2.0,
    noise_seed=42
)
mx.eval(wav_out)
elapsed = time.perf_counter() - t0

out_path = '/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval/demo_clone_voice.wav'
save_wav(wav_out, out_path, ming.sample_rate)
duration = len(wav_out) / ming.sample_rate
rtf = elapsed / duration

print(f"Clone generated successfully! Audio duration: {duration:.2f}s, Elapsed: {elapsed:.2f}s, RTF: {rtf:.3f}")
