
import time, os, sys, json
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

TEXT = "你好，这是纯MLX架构在全精度、八比特以及四比特量化下的横向对比测试。"
MODELS = [
    {"name": "16-bit (bfloat16)", "dir": "models/Ming-3.0-tts-0.5b-mlx", "out": "output/compare_bf16.wav"},
    {"name": "8-bit (int8)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit", "out": "output/compare_8bit.wav"},
    {"name": "4-bit (int4)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit", "out": "output/compare_4bit.wav"},
]

results = []
for m in MODELS:
    print(f"\n--- Testing {m['name']} ---")
    weights_path = os.path.join(m['dir'], 'mlx_weights.safetensors')
    size_mb = os.path.getsize(weights_path) / 1024 / 1024
    
    t_load = time.perf_counter()
    runner = MingMLX(m['dir'])
    load_time = time.perf_counter() - t_load
    
    # Warmup
    _ = runner.generate(DEFAULT_TTS_PROMPT, "热身", ip="灵小萌", max_decode_steps=5)
    mx.eval(_)
    
    t0 = time.perf_counter()
    wav = runner.generate(DEFAULT_TTS_PROMPT, TEXT, ip="灵小萌", noise_seed=42)
    mx.eval(wav)
    elapsed = time.perf_counter() - t0
    
    save_wav(wav, m['out'], runner.sample_rate)
    duration = len(wav) / runner.sample_rate
    rtf = elapsed / duration
    
    wav_np = np.array(wav)
    rms = float(np.sqrt(np.mean(wav_np**2)))
    
    res = {
        "model": m['name'],
        "weight_size_mb": round(size_mb, 1),
        "load_time_s": round(load_time, 2),
        "audio_duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 3),
        "rms": round(rms, 4),
        "out_path": m['out']
    }
    results.append(res)
    print(f"Weights: {size_mb:.1f} MB | Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open('output/quantization_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nComparison run completed!")
