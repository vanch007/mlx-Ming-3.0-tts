
import json, os, time, sys
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

OFFICIAL_DIR = '/Users/vanch/mlx-Ming-omni-tts/benchmarks/ming-official-demo-listening-report-fixed/official'
OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval'
os.makedirs(OUT_DIR, exist_ok=True)

ming = MingMLX('models/Ming-3.0-tts-0.5b-mlx')

cases = [
    {
        "id": "demo_ip_lingxiaozhen",
        "category": "内置高品质音色 (IP: 灵小甄 - 带货主播)",
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "ip": "灵小甄",
        "style": None,
        "official_wav": os.path.join(OFFICIAL_DIR, "static__audio__ip__ip9.wav")
    },
    {
        "id": "demo_style_child_protest",
        "category": "自然语言音色设计 (女童声抗议语气)",
        "text": "人家从那走过，他们就说我故意偷听，还说我是小广播，我偏要广播，偏要广播偏。",
        "ip": None,
        "style": "性别: 女童声音. 音高: 音高尖锐，持续偏高. 语速: 语速迅捷，语气急促. 音量: 音量响亮，情绪饱满. 年龄: 学龄儿童. 清晰度: 吐字清晰，发音用力. 流畅度: 表达流畅，伴强调性重复. 口音: 标准普通话. 音色质感: 童声清亮，略显尖锐. 情绪: 激动委屈，带有抗议. 语调: 声调高昂，语势急切. 性格: 急躁率真，不甘示弱.",
        "official_wav": os.path.join(OFFICIAL_DIR, "static__audio__style__st1.wav")
    },
    {
        "id": "demo_basic_slow_nav",
        "category": "基础属性控制 (语速: 慢速导航播报)",
        "text": "导航开始，全程二十五公里，预计需要十二分钟。",
        "ip": "灵小柔",
        "style": "语速偏慢，发音沉稳清晰的导航播音员",
        "official_wav": os.path.join(OFFICIAL_DIR, "static__audio__basic__bs12.wav")
    },
    {
        "id": "demo_dialect_cantonese",
        "category": "方言控制 (广粤话口音)",
        "text": "佢系头大冇脑脑大生草种",
        "ip": None,
        "style": "地道纯正广粤方言口音，语气自然生动",
        "official_wav": os.path.join(OFFICIAL_DIR, "static__audio__dialect__dialect_1.wav")
    }
]

print("Warming up MLX JIT graph...")
_ = ming.generate(DEFAULT_TTS_PROMPT, "热身测试", ip="灵小萌", max_decode_steps=10)
mx.eval(_)

results = []
for c in cases:
    print(f"\n--- Testing {c['id']} ({c['category']}) ---")
    out_path = os.path.join(OUT_DIR, f"{c['id']}.wav")
    
    t0 = time.perf_counter()
    wav = ming.generate(
        DEFAULT_TTS_PROMPT,
        c['text'],
        ip=c['ip'],
        style=c['style'],
        noise_seed=42
    )
    mx.eval(wav)
    elapsed = time.perf_counter() - t0
    
    save_wav(wav, out_path, ming.sample_rate)
    duration = len(wav) / ming.sample_rate
    rtf = elapsed / duration
    
    off_info = sf.info(c['official_wav'])
    
    item = {
        "id": c['id'],
        "category": c['category'],
        "text": c['text'],
        "local_wav": out_path,
        "official_wav": c['official_wav'],
        "local_duration_s": round(duration, 2),
        "official_duration_s": round(off_info.duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 3)
    }
    results.append(item)
    print(f"Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f} | Output: {out_path}")

with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nAll cases tested successfully!")
