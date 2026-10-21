
import json, os, time, sys, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/cookbook_comparison.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    cases = json.load(f)

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

NEW_CASES = [
    {
        "case_id": "cb11_multilingual_vi",
        "category": "多语言扩展实验：越南语 (Vietnamese)",
        "desc": "实验性语种测试：越南语拉美扩展字符与声调发音（官方未经完整多模态对齐，音色呈现拼音拟音状态）",
        "prompt": "Please generate speech based on the following description.\nTarget language: Vietnamese.\n",
        "text": "Xin chào, đây là hệ thống tổng hợp giọng nói Ming chạy hoàn toàn trên Apple Silicon.",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "variants": {}
    },
    {
        "case_id": "cb12_multilingual_ms",
        "category": "多语言扩展实验：马来语 (Malay)",
        "desc": "实验性语种测试：马来语词汇发音测试（基于Qwen2多语种知识合成，Whisper可准确识别大部分词汇）",
        "prompt": "Please generate speech based on the following description.\nTarget language: Malay.\n",
        "text": "Helo, ini adalah sistem sintesis pertuturan Ming yang berjalan sepenuhnya pada Apple Silicon.",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "variants": {}
    }
]

MODELS = [
    {"key": "bf16", "name": "MLX bf16 (全精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-bf16"},
    {"key": "int8", "name": "MLX 8-bit (整数量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit"},
    {"key": "int4", "name": "MLX 4-bit (超轻量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit"},
]

OUT_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/cookbook_audio'

for m in MODELS:
    mkey = m['key']
    mname = m['name']
    mdir = m['dir']
    print(f"\nLoading {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, 'mlx_weights.safetensors')
    weight_size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    runner = MingMLX(mdir)
    
    for idx, c in enumerate(NEW_CASES):
        cid = c['case_id']
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(OUT_AUDIO_DIR, out_fname)
        
        t0 = time.perf_counter()
        wav = runner.generate(
            c['prompt'],
            c['text'],
            ref_audio=c.get('ref_audio'),
            prompt_text=c.get('prompt_text'),
            sigma=0.15,
            cfg=2.0,
            noise_seed=42
        )
        mx.eval(wav)
        elapsed = time.perf_counter() - t0
        
        save_wav(wav, out_path, runner.sample_rate)
        duration = len(wav) / runner.sample_rate
        rtf = elapsed / duration
        
        with open(out_path, 'rb') as f:
            b64_str = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('ascii')}"
            
        c['variants'][mkey] = {
            "name": mname,
            "weight_size_mb": weight_size_mb,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "b64": b64_str
        }
        print(f"  [{cid}] Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

# Append to cases
cases.extend(NEW_CASES)

with open(DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(cases, f, indent=2, ensure_ascii=False)

print("\nSuccessfully updated cookbook_comparison.json with Vietnamese and Malay cases!")
