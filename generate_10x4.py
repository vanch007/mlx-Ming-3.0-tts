
import json, os, time, sys, shutil
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav

BASE_DIR = '/Users/vanch/mlx-Ming-omni-tts/benchmarks/ming-official-demo-listening-report-fixed'
WEB_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/audio'
os.makedirs(WEB_AUDIO_DIR, exist_ok=True)

# 10 cases covering all official sections
CASES = [
    {
        "case_id": "01_basic",
        "category_name": "基础属性控制 (语速: 慢速)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "导航开始，全程二十五公里，预计需要十二分钟。",
        "instruction_dict": {"语速": "慢速"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__basic__bs11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__basic__bs12.wav")
    },
    {
        "case_id": "02_dialect",
        "category_name": "方言口音控制 (广粤话)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "佢系头大冇脑脑大生草种",
        "instruction_dict": {"方言": "广粤话"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__dialect__dialect11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__dialect__dialect_1.wav")
    },
    {
        "case_id": "03_emotion",
        "category_name": "情感控制 (英文: 高兴)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "If these examinations are held orally, they may be known colloquially as \"orals\".",
        "instruction_dict": {"情感": "高兴"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__emotion__emo11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__emotion__emo12.wav")
    },
    {
        "case_id": "04_ip",
        "category_name": "内置高品质音色 (灵小甄 - 带货主播)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "instruction_dict": {"IP": "灵小甄"},
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__ip__ip9.wav")
    },
    {
        "case_id": "05_style",
        "category_name": "自然语言音色设计 (女童声急促抗议)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "人家从那走过，他们就说我故意偷听，还说我是小广播，我偏要广播，偏要广播偏。",
        "instruction_dict": {"风格": "性别: 女童声音. 音高: 音高尖锐，持续偏高. 语速: 语速迅捷，语气急促. 音量: 音量响亮，情绪饱满. 年龄: 学龄儿童. 清晰度: 吐字清晰，发音用力. 流畅度: 表达流畅，伴强调性重复. 口音: 标准普通话. 音色质感: 童声清亮，略显尖锐. 情绪: 激动委屈，带有抗议. 语调: 声调高昂，语势急切. 性格: 急躁率真，不甘示弱."},
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__style__st1.wav")
    },
    {
        "case_id": "06_clone",
        "category_name": "零样本声音克隆 (In-context 声学提示)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "请用参考音色自然地说出这句话：测试需要同时关注内容、音色和稳定性。",
        "instruction_dict": None,
        "ref_audio": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav",
        "prompt_text": "冬天对于田野和河流来说，是一个放松休息的季节。",
        "official_wav": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav"
    },
    {
        "case_id": "07_music",
        "category_name": "音乐伴奏生成 (Music: 迪斯科/电吉他)",
        "prompt": "Please generate music based on the following description.\n",
        "text": " Genre: 迪斯科. Mood: 活力四射 / 精力充沛. Instrument: 电吉他. Theme: 运动. Duration: 30s",
        "instruction_dict": None,
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__music__music1.wav")
    },
    {
        "case_id": "08_speech_music",
        "category_name": "语音/音乐单通道联合生成 (BGM: 电子舞曲)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "全神贯注，跟上这强劲的节奏，冲向终点吧！",
        "instruction_dict": {
            "BGM": {
                "Genre": "电子舞曲.",
                "Mood": "活力四射.",
                "Instrument": "合成铜管乐器.",
                "Theme": "运动.",
                "SNR": 5.0,
                "ENV": None
            }
        },
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__speech_music__sm11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__speech_music__sm12.wav")
    },
    {
        "case_id": "09_tta",
        "category_name": "文本到音频事件 (TTA: 马达加速换挡音效)",
        "prompt": "Please generate audio events based on given text.\n",
        "text": "A motor is revving and changing gears",
        "instruction_dict": None,
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__tta__tta1.wav")
    },
    {
        "case_id": "10_speech_sound",
        "category_name": "语音/环境音单通道联合生成 (鸟鸣环境音)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "副主任及以上号别就诊人次，为二百零八点二万。",
        "instruction_dict": {
            "BGM": {
                "ENV": "Birds chirping",
                "SNR": 10.0,
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None
            }
        },
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__speech_sound__ss11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__speech_sound__ss12.wav")
    }
]

# 1. Copy official WAVs
for c in CASES:
    target_off = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
    shutil.copy2(c['official_wav'], target_off)

# 2. Setup matrix records
all_data = []
for c in CASES:
    off_p = os.path.join(WEB_AUDIO_DIR, f"{c['case_id']}_official.wav")
    info = sf.info(off_p)
    rec = {
        "case_id": c['case_id'],
        "category": c['category_name'],
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
    all_data.append(rec)

# 3. Models
MODELS = [
    {"key": "bf16", "name": "MLX bf16 (全精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-bf16"},
    {"key": "int8", "name": "MLX 8-bit (整数量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit"},
    {"key": "int4", "name": "MLX 4-bit (超轻量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit"},
]

def format_instruction(d):
    if not d: return None
    if "IP" in d:
        from mlx_ming_tts.pipeline import build_ip_instruction
        return build_ip_instruction(d['IP'], is_chinese=True)
    elif "风格" in d:
        from mlx_ming_tts.pipeline import build_voice_design_instruction
        return build_voice_design_instruction(d['风格'], is_chinese=True)
    else:
        fields = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
        values = (1, "speaker_1", d.get("方言"), d.get("风格"), d.get("语速"), d.get("基频"), d.get("音量"), d.get("情感"))
        item = dict(zip(fields, values))
        item["BGM"] = d.get("BGM", {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None})
        item["IP"] = d.get("IP")
        return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)

for m in MODELS:
    mkey = m['key']
    mname = m['name']
    mdir = m['dir']
    print(f"\n==========================================")
    print(f"Loading {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, 'mlx_weights.safetensors')
    weight_size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    t_load = time.perf_counter()
    runner = MingMLX(mdir)
    load_time = round(time.perf_counter() - t_load, 2)
    
    # Warmup
    _ = runner.generate(CASES[0]['prompt'], "热身", ip="灵小萌", max_decode_steps=5)
    mx.eval(_)
    
    for idx, c in enumerate(CASES):
        cid = c['case_id']
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(WEB_AUDIO_DIR, out_fname)
        
        # If file already generated and valid, we can reuse or regenerate
        instr_str = format_instruction(c.get('instruction_dict'))
        
        t0 = time.perf_counter()
        wav = runner.generate(
            c['prompt'],
            c['text'],
            instruction=instr_str,
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
        
        all_data[idx]['mlx_versions'][mkey] = {
            "name": mname,
            "audio_file": f"audio/{out_fname}",
            "weight_size_mb": weight_size_mb,
            "load_time_s": load_time,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "size_kb": round(os.path.getsize(out_path) / 1024, 1)
        }
        print(f"  [{cid}] Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/full_10_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print("\nAll 10 features x 4 precisions generated and exported!")
