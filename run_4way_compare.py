
import json, os, time, sys
import soundfile as sf
import numpy as np
import shutil

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

BASE_DIR = '/Users/vanch/mlx-Ming-omni-tts/benchmarks/ming-official-demo-listening-report-fixed'
WEB_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/audio'
os.makedirs(WEB_AUDIO_DIR, exist_ok=True)

# 3 MLX models: bf16, 8bit, 4bit
MLX_MODELS = [
    {"key": "bf16", "name": "MLX bf16 (全精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-bf16"},
    {"key": "int8", "name": "MLX 8-bit (整数量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit"},
    {"key": "int4", "name": "MLX 4-bit (超轻量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit"},
]

CASES = [
    {
        "case_id": "01_ip",
        "category": "内置品质音色 (灵小甄 - 带货主播)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "ip": "灵小甄",
        "style": None,
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__ip__ip9.wav"
    },
    {
        "case_id": "02_style",
        "category": "自然语言音色设计 (女童声急促抗议)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "人家从那走过，他们就说我故意偷听，还说我是小广播，我偏要广播，偏要广播偏。",
        "ip": None,
        "style": "性别: 女童声音. 音高: 音高尖锐，持续偏高. 语速: 语速迅捷，语气急促. 音量: 音量响亮，情绪饱满. 年龄: 学龄儿童. 清晰度: 吐字清晰，发音用力. 流畅度: 表达流畅，伴强调性重复. 口音: 标准普通话. 音色质感: 童声清亮，略显尖锐. 情绪: 激动委屈，带有抗议. 语调: 声调高昂，语势急切. 性格: 急躁率真，不甘示弱.",
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__style__st1.wav"
    },
    {
        "case_id": "03_basic",
        "category": "基础属性控制 (语速: 慢速导航播音)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "导航开始，全程二十五公里，预计需要十二分钟。",
        "ip": "灵小柔",
        "style": "语速偏慢，发音沉稳清晰的导航播音员",
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__basic__bs12.wav"
    },
    {
        "case_id": "04_dialect",
        "category": "方言控制 (广粤话地道俚语)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "佢系头大冇脑脑大生草种",
        "ip": None,
        "style": "地道纯正广粤方言口音，语气自然生动",
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__dialect__dialect_1.wav"
    },
    {
        "case_id": "05_clone",
        "category": "零样本声音克隆 (In-Context 跨句克隆)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "请用参考音色自然地说出这句话：测试需要同时关注内容、音色和稳定性。",
        "ip": None,
        "style": None,
        "ref_audio": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav",
        "prompt_text": "冬天对于田野和河流来说，是一个放松休息的季节。",
        "official_rel": None # ref_audio acts as the reference
    }
]

# 1. Prepare official audio files
for c in CASES:
    off_target = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
    if c['official_rel']:
        src = os.path.join(BASE_DIR, c['official_rel'])
        shutil.copy2(src, off_target)
    elif c['ref_audio']:
        shutil.copy2(c['ref_audio'], off_target)

# 2. Build metadata list
summary_list = []
for c in CASES:
    off_p = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
    info = sf.info(off_p)
    rec = {
        "case_id": c['case_id'],
        "category": c['category'],
        "text": c['text'],
        "official": {
            "name": "官方原始 Demo",
            "audio_file": f"audio/{c['case_id']}_official.wav",
            "duration_s": round(info.duration, 2),
            "samplerate": info.samplerate,
            "size_kb": round(os.path.getsize(off_p) / 1024, 1)
        },
        "mlx_versions": {}
    }
    summary_list.append(rec)

# 3. Generate MLX versions: bf16, 8bit, 4bit
for m in MLX_MODELS:
    mkey = m['key']
    mname = m['name']
    mdir = m['dir']
    print(f"\n>>> Processing model: {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, 'mlx_weights.safetensors')
    weight_size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    t_load = time.perf_counter()
    runner = MingMLX(mdir)
    load_time = round(time.perf_counter() - t_load, 2)
    
    # Warmup
    _ = runner.generate(DEFAULT_TTS_PROMPT, "热身", ip="灵小萌", max_decode_steps=5)
    mx.eval(_)
    
    for idx, c in enumerate(CASES):
        cid = c['case_id']
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(WEB_AUDIO_DIR, out_fname)
        
        t0 = time.perf_counter()
        wav = runner.generate(
            c['prompt'],
            c['text'],
            ip=c['ip'],
            style=c['style'],
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
        
        summary_list[idx]['mlx_versions'][mkey] = {
            "name": mname,
            "audio_file": f"audio/{out_fname}",
            "weight_size_mb": weight_size_mb,
            "load_time_s": load_time,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "size_kb": round(os.path.getsize(out_path) / 1024, 1)
        }
        print(f"  [{mkey}] Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/comparison_data.json', 'w', encoding='utf-8') as f:
    json.dump(summary_list, f, indent=2, ensure_ascii=False)

print("\nGeneration and data export completed successfully!")
