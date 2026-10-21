
import json, os, time, sys, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_same_cases_comparison.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    cases = json.load(f)

dialogue = """speaker_1: 你最近是不是又去那家面馆了？
speaker_2: 对啊，排了半个小时的队，不过真的值得。"""

prompt_wav = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
prompt_text = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_audio'

TINY_MODELS = [
    {"key": "tiny_8bit", "name": "Tiny 8-bit (推荐主力)", "dir": "models/Ming-3.0-tts-tiny-mlx-8bit"},
    {"key": "tiny_4bit", "name": "Tiny 4-bit (超轻量)", "dir": "models/Ming-3.0-tts-tiny-mlx-4bit"},
    {"key": "tiny_bf16", "name": "Tiny bf16 (全精度)", "dir": "models/Ming-3.0-tts-tiny-mlx-bf16"},
]

# Find case 15
case15 = None
for c in cases:
    if c['case_id'] == 'case15_podcast':
        case15 = c
        break

if not case15:
    raise ValueError("case15_podcast not found in json!")

for m in TINY_MODELS:
    mkey = m["key"]
    mname = m["name"]
    mdir = m["dir"]
    print(f"Generating podcast for {mname}...")
    
    runner = MingMLX(mdir)
    weights_path = os.path.join(mdir, "mlx_weights.safetensors")
    size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    out_path = os.path.join(OUT_DIR, f"case15_podcast_{mkey}.wav")
    t0 = time.perf_counter()
    wav = runner.generate(
        DEFAULT_TTS_PROMPT,
        dialogue,
        ref_audio=prompt_wav,
        prompt_text=prompt_text,
        max_decode_steps=75,
        sigma=0.15,
        cfg=2.0,
        noise_seed=42
    )
    mx.eval(wav)
    elapsed = time.perf_counter() - t0
    save_wav(wav, out_path, runner.sample_rate)
    duration = len(wav) / runner.sample_rate
    rtf = elapsed / duration
    
    with open(out_path, "rb") as f:
        b64 = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('ascii')}"
        
    case15["tiny_moe"][mkey] = {
        "name": mname,
        "weight_size_mb": size_mb,
        "elapsed_s": round(elapsed, 2),
        "duration_s": round(duration, 2),
        "rtf": round(rtf, 3),
        "b64": b64
    }
    print(f"  [{mkey}] Done! Duration: {duration:.2f}s | Elapsed: {elapsed:.2f}s | RTF: {rtf:.3f}")

with open(DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(cases, f, indent=2, ensure_ascii=False)

print("Updated case 15 podcast in JSON successfully!")
