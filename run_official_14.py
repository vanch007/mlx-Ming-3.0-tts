
import json, os, time, sys, shutil, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"
EMOTION_WAV = "/Users/vanch/Ming-3.0-tts/data/wavs/emotion_prompt.wav"
YUE_WAV = "/Users/vanch/Ming-3.0-tts/data/wavs/yue_prompt.wav"

OFFICIAL_CASES = [
    {
        "id": "case01_clone_zh",
        "category": "零样本声音克隆 (Zero-Shot Clone: 灵光)",
        "desc": "基于灵光参考音频克隆全新文本，声线特征自适应对齐",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "雨后的山谷很安静，远处亮起了一盏温暖的灯。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case02_paralinguistic_zh",
        "category": "中文副语言控制 ([Laughter] 行内笑声)",
        "desc": "行内笑声标签控制，原样透传不被文本归一化破坏",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "[Laughter] 你居然真的把那封信藏在旧钟后面了。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case03_paralinguistic_en",
        "category": "英文副语言控制 ([Sigh] 叹气 + [Uhm] 迟疑)",
        "desc": "英文行内复合副语言标签控制，叹气与语气助词平滑衔接",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "I checked the drawer twice [Sigh] and it was empty [Uhm] again.",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case04_multilingual_ja",
        "category": "官方多语言：日语 (Japanese)",
        "desc": "指定日语目标语种，绕过汉字归一化保持平假名/片假名纯正发音",
        "prompt": "Please generate speech based on the following description.\nTarget language: Japanese.\n",
        "text": "あなた、ご自分のことばかりですのね。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case05_multilingual_ko",
        "category": "官方多语言：韩语 (Korean)",
        "desc": "官方支持多语种测试：韩语音韵结构与声学合成",
        "prompt": "Please generate speech based on the following description.\nTarget language: Korean.\n",
        "text": "오늘 저녁에는 비가 올 것 같아요.",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case06_multilingual_fr",
        "category": "官方多语言：法语 (French)",
        "desc": "官方支持多语种测试：法语连音与欧系语言声学合成",
        "prompt": "Please generate speech based on the following description.\nTarget language: French.\n",
        "text": "Le vieux phare veille encore sur la baie.",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case07_freeform_en",
        "category": "英文自由描述指令 (Free-Form Instruction)",
        "desc": "纯自然语言指令定制：温暖、怀旧、舒缓亲切的说书人叙事质感",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "The old radio crackled softly as the storm approached.",
        "ip": None,
        "style": "Speak in a warm, intimate storytelling voice, with a gentle pace and a hint of nostalgia.",
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "case08_voice_design",
        "category": "自然语言音色设计 (Voice Design)",
        "desc": "纯自然语言定制：成熟沉稳的纪录片男声旁白，吐字舒缓清晰",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "冰川在漫长岁月中缓缓移动，留下深刻的痕迹。",
        "ip": None,
        "style": "成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓。",
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "case09_basic_fast",
        "category": "基础属性控制 (语速: 快速)",
        "desc": "显式基础声学控制：设置快速语速、中音高与中音量",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "今晚的观测将在云层散开后开始。",
        "instruction_dict": {"语速": "快速", "基频": "中", "音量": "中"},
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "case10_emotion_sad",
        "category": "情感控制：悲伤 (Emotion: Sad)",
        "desc": "跨情感控制，使用中性提示生成深沉悲伤、低落的语音色彩",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "所有的测试都通过了。",
        "instruction_dict": {"情感": "悲伤"},
        "ref_audio": EMOTION_WAV,
        "prompt_text": None
    },
    {
        "id": "case11_dialect_sichuan",
        "category": "方言控制：四川话 (Sichuanese Dialect)",
        "desc": "地道川渝方言调腔控制，发音纯正地道",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "今天天气巴适得很，我们一起出去转哈嘛。",
        "instruction_dict": {"方言": "川渝话"},
        "ref_audio": YUE_WAV,
        "prompt_text": None
    },
    {
        "id": "case12_builtin_ip_us_james",
        "category": "官方推荐英文内置音色 (US_James)",
        "desc": "官方推荐 5 大英文高品质原生音色之一，纯正美式成熟男声",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "The garden will remain open until sunset, provided the weather stays clear.",
        "ip": "US_James",
        "style": None,
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "case13_builtin_ip_lingxiaomeng",
        "category": "官方推荐中文内置音色 (灵小萌)",
        "desc": "官方 100+ 推荐音色，治愈系电台主播女声，吐字干练清脆",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "夜晚的微风吹散了白天的喧嚣，愿好梦常伴你左右。",
        "ip": "灵小萌",
        "style": None,
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "case14_text_norm",
        "category": "复杂电商混合文本归一化 (Text Normalization)",
        "desc": "订单号、日期、货币符号混合自动扩展与精准朗读",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "订单号 20250317，共 3 件商品，合计 ¥1,280.50，预计 3-5 天送达。",
        "ip": "灵小锋",
        "style": None,
        "ref_audio": None,
        "prompt_text": None
    }
]

MODELS = [
    {"key": "bf16", "name": "MLX bf16 (全精度)", "dir": "models/Ming-3.0-tts-0.5b-mlx-bf16"},
    {"key": "int8", "name": "MLX 8-bit (整数量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-8bit"},
    {"key": "int4", "name": "MLX 4-bit (超轻量化)", "dir": "models/Ming-3.0-tts-0.5b-mlx-4bit"},
]

def format_instruction(d):
    if not d: return None
    fields = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
    values = (1, "speaker_1", d.get("方言"), d.get("风格"), d.get("语速"), d.get("基频"), d.get("音量"), d.get("情感"))
    item = dict(zip(fields, values))
    item["BGM"] = {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}
    item["IP"] = d.get("IP")
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)

all_cases_data = []
for c in OFFICIAL_CASES:
    all_cases_data.append({
        "case_id": c['id'],
        "category": c['category'],
        "desc": c['desc'],
        "text": c['text'],
        "ref_audio": c['ref_audio'],
        "variants": {}
    })

OUT_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/official_cases_audio'
os.makedirs(OUT_AUDIO_DIR, exist_ok=True)

for m in MODELS:
    mkey = m['key']
    mname = m['name']
    mdir = m['dir']
    print(f"\n==========================================")
    print(f"Loading {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, 'mlx_weights.safetensors')
    weight_size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    runner = MingMLX(mdir)
    
    for idx, c in enumerate(OFFICIAL_CASES):
        cid = c['id']
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(OUT_AUDIO_DIR, out_fname)
        
        instr_str = format_instruction(c.get('instruction_dict'))
        
        t0 = time.perf_counter()
        wav = runner.generate(
            c['prompt'],
            c['text'],
            instruction=instr_str,
            ip=c.get('ip'),
            style=c.get('style'),
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
            
        all_cases_data[idx]['variants'][mkey] = {
            "name": mname,
            "weight_size_mb": weight_size_mb,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "b64": b64_str
        }
        print(f"  [{cid}] Elapsed: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/official_14_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(all_cases_data, f, indent=2, ensure_ascii=False)

print("\nAll 14 official cookbook cases generated successfully!")
