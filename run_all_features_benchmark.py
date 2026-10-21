
import json, os, time, sys
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, load_and_resample_prompt

BASE_DIR = '/Users/vanch/mlx-Ming-omni-tts/benchmarks/ming-official-demo-listening-report-fixed'
OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval'
os.makedirs(OUT_DIR, exist_ok=True)

# Use 8-bit model for fast, high-quality full test
ming = MingMLX('models/Ming-3.0-tts-0.5b-mlx-8bit')

cases = [
    {
        "category_id": "01_basic",
        "category_name": "基础属性控制 (语速: 慢速)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "导航开始，全程二十五公里，预计需要十二分钟。",
        "instruction_dict": {"语速": "慢速"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__basic__bs11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__basic__bs12.wav")
    },
    {
        "category_id": "02_dialect",
        "category_name": "方言口音控制 (广粤话)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "佢系头大冇脑脑大生草种",
        "instruction_dict": {"方言": "广粤话"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__dialect__dialect11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__dialect__dialect_1.wav")
    },
    {
        "category_id": "03_emotion",
        "category_name": "情感控制 (高兴)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "If these examinations are held orally, they may be known colloquially as \"orals\".",
        "instruction_dict": {"情感": "高兴"},
        "ref_audio": os.path.join(BASE_DIR, "refs/static__audio__emotion__emo11.wav"),
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__emotion__emo12.wav")
    },
    {
        "category_id": "04_ip",
        "category_name": "内置高品质音色 (灵小甄 - 带货主播)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "instruction_dict": {"IP": "灵小甄"},
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__ip__ip9.wav")
    },
    {
        "category_id": "05_style",
        "category_name": "自然语言音色设计 (女童声急促抗议)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "人家从那走过，他们就说我故意偷听，还说我是小广播，我偏要广播，偏要广播偏。",
        "instruction_dict": {"风格": "性别: 女童声音. 音高: 音高尖锐，持续偏高. 语速: 语速迅捷，语气急促. 音量: 音量响亮，情绪饱满. 年龄: 学龄儿童. 清晰度: 吐字清晰，发音用力. 流畅度: 表达流畅，伴强调性重复. 口音: 标准普通话. 音色质感: 童声清亮，略显尖锐. 情绪: 激动委屈，带有抗议. 语调: 声调高昂，语势急切. 性格: 急躁率真，不甘示弱."},
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__style__st1.wav")
    },
    {
        "category_id": "06_clone",
        "category_name": "零样本声音克隆 (In-context 声学提示)",
        "prompt": "Please generate speech based on the following description.\n",
        "text": "请用参考音色自然地说出这句话：测试需要同时关注内容、音色和稳定性。",
        "instruction_dict": None,
        "ref_audio": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav",
        "prompt_text": "冬天对于田野和河流来说，是一个放松休息的季节。",
        "official_wav": "/Users/vanch/tts-test-project/data/reference_audio/dots_seedtts_zh_ref.wav"
    },
    {
        "category_id": "07_music",
        "category_name": "音乐伴奏生成 (Music: 迪斯科/电吉他)",
        "prompt": "Please generate music based on the following description.\n",
        "text": " Genre: 迪斯科. Mood: 活力四射 / 精力充沛. Instrument: 电吉他. Theme: 运动. Duration: 30s",
        "instruction_dict": None,
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__music__music1.wav")
    },
    {
        "category_id": "08_speech_music",
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
        "category_id": "09_tta",
        "category_name": "文本到音频事件 (TTA: 马达加速换挡音效)",
        "prompt": "Please generate audio events based on given text.\n",
        "text": "A motor is revving and changing gears",
        "instruction_dict": None,
        "ref_audio": None,
        "prompt_text": None,
        "official_wav": os.path.join(BASE_DIR, "official/static__audio__tta__tta1.wav")
    },
    {
        "category_id": "10_speech_sound",
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

print("Warming up MLX JIT graph...")
_ = ming.generate(cases[0]['prompt'], "热身", ip="灵小萌", max_decode_steps=5)
mx.eval(_)

results = []
for c in cases:
    cid = c['category_id']
    cname = c['category_name']
    print(f"\n=== Testing [{cid}] {cname} ===")
    out_path = os.path.join(OUT_DIR, f"{cid}.wav")
    
    # Format instruction JSON if present
    instr_str = None
    if c.get('instruction_dict'):
        d = c['instruction_dict']
        if "IP" in d:
            from mlx_ming_tts.pipeline import build_ip_instruction
            instr_str = build_ip_instruction(d['IP'], is_chinese=True)
        elif "风格" in d:
            from mlx_ming_tts.pipeline import build_voice_design_instruction
            instr_str = build_voice_design_instruction(d['风格'], is_chinese=True)
        else:
            # structured instruction
            fields = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
            values = (1, "speaker_1", d.get("方言"), d.get("风格"), d.get("语速"), d.get("基频"), d.get("音量"), d.get("情感"))
            item = dict(zip(fields, values))
            item["BGM"] = d.get("BGM", {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None})
            item["IP"] = d.get("IP")
            instr_str = json.dumps({"audio_sequence": [item]}, ensure_ascii=False)
            
    t0 = time.perf_counter()
    wav = ming.generate(
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
    
    save_wav(wav, out_path, ming.sample_rate)
    duration = len(wav) / ming.sample_rate
    rtf = elapsed / duration
    
    off_info = sf.info(c['official_wav'])
    
    res = {
        "category_id": cid,
        "category_name": cname,
        "text": c['text'],
        "local_wav": out_path,
        "official_wav": c['official_wav'],
        "local_duration_s": round(duration, 2),
        "official_duration_s": round(off_info.duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 3)
    }
    results.append(res)
    print(f"Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nAll 10 feature cases completed!")
