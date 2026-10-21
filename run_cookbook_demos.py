
import json, os, time, sys, shutil, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

# 10 Official Cookbook Capabilities of Ming-3.0-tts-0.5b:
COOKBOOK_CASES = [
    {
        "id": "cb01_clone",
        "category": "零样本声音克隆 (Zero-Shot Voice Cloning)",
        "desc": "基于灵光参考音频克隆全新文本，声线特征自适应对齐",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "雨后的山谷很安静，远处亮起了一盏温暖的灯。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "cb02_paralinguistic",
        "category": "副语言笑声标签 ([Laughter] Paralinguistic)",
        "desc": "行内副语言笑声与语气词控制，原样透传不被文本归一化破坏",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "[Laughter] 你居然真的把那封信藏在旧钟后面了。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "cb03_multilingual_ja",
        "category": "多语言拓展：日语 (Japanese)",
        "desc": "指定日语目标语种，绕过汉字归一化保持平假名/片假名纯正发音",
        "prompt": "Please generate speech based on the following description.\nTarget language: Japanese.\n",
        "text": "あなた、ご自己的ことばかりですのね。",
        "ip": None, "style": None,
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "cb04_freeform_en",
        "category": "英文自由描述指令 (Free-Form Instruction)",
        "desc": "通过纯自然语言自由指令塑造温馨亲切的说书人叙事质感",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "The old radio crackled softly as the storm approached.",
        "ip": None,
        "style": "Speak in a warm, intimate storytelling voice, with a gentle pace and a hint of nostalgia.",
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "cb05_voice_design",
        "category": "自然语言音色设计 (Voice Design)",
        "desc": "纯自然语言定制：成熟沉稳的纪录片男声旁白",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "冰川在漫长岁月中缓缓移动，留下深刻的痕迹。",
        "ip": None,
        "style": "成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓。",
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "cb06_basic_fast",
        "category": "基础属性控制 (语速: 快速)",
        "desc": "显式基础声学控制：设置快速语速、中音高与中音量",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "今晚的观测将在云层散开后开始。",
        "instruction_dict": {"语速": "快速", "基频": "中", "音量": "中"},
        "ref_audio": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT
    },
    {
        "id": "cb07_emotion_sad",
        "category": "情感控制：悲伤 (Emotion: Sad)",
        "desc": "情感维度控制：带有深沉悲伤、低落情绪的语音合成",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "所有的测试都通过了。",
        "instruction_dict": {"情感": "悲伤"},
        "ref_audio": "/Users/vanch/Ming-3.0-tts/data/wavs/emotion_prompt.wav",
        "prompt_text": None
    },
    {
        "id": "cb08_dialect_sichuan",
        "category": "方言控制：四川话 (Sichuanese Dialect)",
        "desc": "地道川渝方言俚语调腔控制，发音纯正地道",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "今天天气巴适得很，我们一起出去转哈嘛。",
        "instruction_dict": {"方言": "川渝话"},
        "ref_audio": "/Users/vanch/Ming-3.0-tts/data/wavs/yue_prompt.wav",
        "prompt_text": None
    },
    {
        "id": "cb09_builtin_ip",
        "category": "内置品质音色 (Built-in IP: 灵小萌)",
        "desc": "官方精调 100+ 内置音色，治愈系元气电台主播声线",
        "prompt": DEFAULT_TTS_PROMPT,
        "text": "夜晚的微风吹散了白天的喧嚣，愿好梦常伴你左右。",
        "ip": "灵小萌",
        "style": None,
        "ref_audio": None,
        "prompt_text": None
    },
    {
        "id": "cb10_text_norm",
        "category": "复杂电商混合文本归一化 (Text Normalization)",
        "desc": "订单号、货币符号、日期、数字混合自动扩展与朗读",
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

# Collect data
all_cases_data = []
for c in COOKBOOK_CASES:
    all_cases_data.append({
        "case_id": c['id'],
        "category": c['category'],
        "desc": c['desc'],
        "text": c['text'],
        "variants": {}
    })

OUT_AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/cookbook_audio'
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
    
    for idx, c in enumerate(COOKBOOK_CASES):
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
        
        # Read base64
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

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/cookbook_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(all_cases_data, f, indent=2, ensure_ascii=False)

print("\nAll official cookbook cases generated successfully!")
