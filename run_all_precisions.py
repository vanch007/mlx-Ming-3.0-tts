
import json, os, time, sys
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

BASE_DIR = '/Users/vanch/mlx-Ming-omni-tts/benchmarks/ming-official-demo-listening-report-fixed'
WEB_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/audio'
os.makedirs(WEB_AUDIO_DIR, exist_ok=True)

PRECISIONS = [
    {"key": "bf16", "name": "BFloat16 (全精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-bf16"},
    {"key": "fp16", "name": "Float16 (半精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-fp16"},
    {"key": "int8", "name": "8-bit (整数量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit"},
    {"key": "int4", "name": "4-bit (超轻量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit"},
]

CASES = [
    {
        "case_id": "case1_ip",
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
        "case_id": "case2_style",
        "category": "自然语言音色设计 (女童声激昂抗议)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "人家从那走过，他们就说我故意偷听，还说我是小广播，我偏要广播，偏要广播偏。",
        "ip": None,
        "style": "性别: 女童声音. 音高: 音高尖锐，持续偏高. 语速: 语速迅捷，语气急促. 音量: 音量响亮，情绪饱满. 年龄: 学龄儿童. 清晰度: 吐字清晰，发音用力. 流畅度: 表达流畅，伴强调性重复. 口音: 标准普通话. 音色质感: 童声清亮，略显尖锐. 情绪: 激动委屈，带有抗议. 语调: 声调高昂，语势急切. 性格: 急躁率真，不甘示弱.",
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__style__st1.wav"
    },
    {
        "case_id": "case3_basic",
        "category": "基础属性控制 (语速: 慢速导航)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "导航开始，全程二十五公里，预计需要十二分钟。",
        "ip": "灵小柔",
        "style": "语速偏慢，发音沉稳清晰的导航播音员",
        "ref_audio": None,
        "prompt_text": None,
        "official_rel": "official/static__audio__basic__bs12.wav"
    },
    {
        "case_id": "case4_dialect",
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
        "case_id": "case5_clone",
        "category": "零样本声音克隆 (In-Context 跨句克隆)",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "请用参考音色自然地说出这句话：测试需要同时关注内容、音色和稳定性。",
        "ip": None,
        "style": None,
        "ref_audio": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav",
        "prompt_text": "冬天对于田野和河流来说，是一个放松休息的季节。",
        "official_rel": None # Uses ref_audio as base comparison
    }
]

# Copy official wavs to web directory
import shutil
for c in CASES:
    if c['official_rel']:
        src_p = os.path.join(BASE_DIR, c['official_rel'])
        dst_p = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
        shutil.copy2(src_p, dst_p)
    elif c['ref_audio']:
        dst_p = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
        shutil.copy2(c['ref_audio'], dst_p)

# We will collect matrix: case -> { official, bf16, fp16, int8, int4 }
data_matrix = []

for c in CASES:
    print(f"\n==========================================")
    print(f"Processing Case: [{c['case_id']}] {c['category']}")
    
    off_wav_path = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
    off_info = sf.info(off_wav_path)
    
    case_record = {
        "case_id": c['case_id'],
        "category": c['category'],
        "text": c['text'],
        "official": {
            "path": f"audio/{c['case_id']}_official.wav",
            "duration": round(off_info.duration, 2),
            "samplerate": off_info.samplerate
        },
        "variants": {}
    }
    data_matrix.append(case_record)

for p in PRECISIONS:
    pkey = p['key']
    pname = p['name']
    pdir = p['dir']
    print(f"\n>>> Loading model for {pname} from {pdir}...")
    
    weights_path = os.path.join(pdir, 'mlx_weights.safetensors')
    size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    t_load = time.perf_counter()
    runner = MingMLX(pdir)
    load_time = round(time.perf_counter() - t_load, 2)
    
    # Warmup
    _ = runner.generate(DEFAULT_TTS_PROMPT, "热身", ip="灵小萌", max_decode_steps=5)
    mx.eval(_)
    
    for idx, c in enumerate(CASES):
        cid = c['case_id']
        out_fname = f"{cid}_{pkey}.wav"
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
        
        data_matrix[idx]['variants'][pkey] = {
            "name": pname,
            "path": f"audio/{out_fname}",
            "size_mb": size_mb,
            "load_time_s": load_time,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3)
        }
        print(f"  [{pkey}] Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/matrix_data.json', 'w', encoding='utf-8') as f:
    json.dump(data_matrix, f, indent=2, ensure_ascii=False)

print("\nAll 4 precisions generated and matrix data saved!")
