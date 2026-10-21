
import json, os, time, sys, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

TINY_CASES = [
    {
        "id": "tiny01_seed_clone",
        "category": "Tiny 零样本声音克隆 (Seed TTS: 灵光)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "傍晚的风从山口吹过来，远处的云慢慢染成了金色。",
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "ip": None
    },
    {
        "id": "tiny02_podcast",
        "category": "Tiny 独家能力：双人对话剧本 (Podcast Dialogue)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "speaker_1: 你最近是不是又去那家面馆了？\nspeaker_2: 对啊，排了半个小时的队，不过真的值得。",
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "ip": None
    },
    {
        "id": "tiny03_multilingual_ja",
        "category": "Tiny 多语种拓展：日语 (Japanese)",
        "prompt": "Please generate speech based on the following description.\nTarget language: Japanese.\n",
        "text": "あなた、ご自分のことばかりですのね。",
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
        "ip": None
    },
    {
        "id": "tiny04_voice_design",
        "category": "Tiny 自然语言音色设计 (Voice Design)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "冰川在漫长岁月中缓缓移动，也悄悄记录着气候变化的痕迹。",
        "ref_audio": None,
        "prompt_text": None,
        "style": "成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓，语气理性而有厚度。"
    },
    {
        "id": "tiny05_ip_us_james",
        "category": "Tiny 官方推荐英文内置音色 (US_James)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "The garden will remain open until sunset, provided the weather stays clear.",
        "ref_audio": None,
        "prompt_text": None,
        "ip": "US_James"
    }
]

DATA_PATH = "/Users/vanch/mlx-Ming-3.0-tts/demo_page/tiny_comparison.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

OUT_DIR = "/Users/vanch/mlx-Ming-3.0-tts/demo_page/tiny_audio"
os.makedirs(OUT_DIR, exist_ok=True)

bf16_dir = "models/Ming-3.0-tts-tiny-mlx-bf16"
weights_path = os.path.join(bf16_dir, "mlx_weights.safetensors")
size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)

print(f"Loading Tiny bf16 ({bf16_dir})...")
t_load = time.perf_counter()
runner = MingMLX(bf16_dir)
load_time = round(time.perf_counter() - t_load, 2)
print(f"Loaded Tiny bf16 in {load_time}s!")

for idx, c in enumerate(TINY_CASES):
    cid = c["id"]
    out_path = os.path.join(OUT_DIR, f"{cid}_tiny_bf16.wav")
    print(f"\nGenerating bf16 for [{cid}] {c['category']}...")
    
    t0 = time.perf_counter()
    wav = runner.generate(
        c["prompt"],
        c["text"],
        ip=c.get("ip"),
        style=c.get("style"),
        ref_audio=c.get("ref_audio"),
        prompt_text=c.get("prompt_text"),
        max_decode_steps=70,
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
        
    results[idx]["variants"]["tiny_bf16"] = {
        "name": "Tiny bf16 (全精度)",
        "weight_size_mb": size_mb,
        "elapsed_s": round(elapsed, 2),
        "duration_s": round(duration, 2),
        "rtf": round(rtf, 3),
        "b64": b64
    }
    print(f"  [tiny_bf16] Elapsed: {elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nTiny bf16 generation and JSON update finished successfully!")
