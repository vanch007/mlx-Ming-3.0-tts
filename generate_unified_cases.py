
import json, os, time, sys, base64
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
import mlx.core as mx
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

CB_JSON = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/official_14_comparison.json'
with open(CB_JSON, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Add case 15: podcast (Tiny exclusive)
cases.append({
    "case_id": "case15_podcast",
    "category": "双人对话剧本 (Podcast Dialogue - Tiny独家)",
    "desc": "7.9B MoE 独家双人角色交替对话剧本（0.5b 不支持此任务）",
    "text": "speaker_1: 你最近是不是又去那家面馆了？\nspeaker_2: 对啊，排了半个小时的队，不过真的值得。",
    "ref_audio": "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "prompt_text": "诶，你今天有空吗？我们能不能聊聊这件事啊？",
    "prompt": DEFAULT_TTS_PROMPT,
    "ip": None,
    "style": None,
    "variants": {}  # 0.5b has no audio for podcast
})

# Let's inspect the fields of each case
PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_audio'
os.makedirs(OUT_DIR, exist_ok=True)

TINY_MODELS = [
    {"key": "tiny_8bit", "name": "Tiny 8-bit (推荐主力)", "dir": "models/Ming-3.0-tts-tiny-mlx-8bit"},
    {"key": "tiny_4bit", "name": "Tiny 4-bit (超轻量)", "dir": "models/Ming-3.0-tts-tiny-mlx-4bit"},
    {"key": "tiny_bf16", "name": "Tiny bf16 (全精度)", "dir": "models/Ming-3.0-tts-tiny-mlx-bf16"},
]

# We will store results in unified_matrix
unified_data = []
for c in cases:
    # 0.5b variants from existing json if present
    existing_vars = c.get("variants", {})
    rec = {
        "case_id": c["case_id"],
        "category": c["category"],
        "desc": c.get("desc", ""),
        "text": c["text"],
        "ref_audio": c.get("ref_audio"),
        "dense_0_5b": {
            "int8": existing_vars.get("int8"),
            "bf16": existing_vars.get("bf16"),
            "int4": existing_vars.get("int4"),
        },
        "tiny_moe": {}
    }
    unified_data.append(rec)

for m in TINY_MODELS:
    mkey = m["key"]
    mname = m["name"]
    mdir = m["dir"]
    print(f"\n==========================================")
    print(f"Loading {mname} ({mdir})...")
    
    weights_path = os.path.join(mdir, "mlx_weights.safetensors")
    size_mb = round(os.path.getsize(weights_path) / 1024 / 1024, 1)
    
    t_load = time.perf_counter()
    runner = MingMLX(mdir)
    load_time = round(time.perf_counter() - t_load, 2)
    
    for idx, c in enumerate(cases):
        cid = c["case_id"]
        out_fname = f"{cid}_{mkey}.wav"
        out_path = os.path.join(OUT_DIR, out_fname)
        
        prompt = c.get("prompt", DEFAULT_TTS_PROMPT)
        text = c["text"]
        ip = c.get("ip")
        style = c.get("style")
        ref_audio = c.get("ref_audio")
        prompt_text = PROMPT_TEXT if ref_audio else None
        
        # Check instruction dict if present in run_official_14 definitions
        instr_str = None
        if "fast" in cid:
            from mlx_ming_tts.pipeline import build_voice_design_instruction
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": None, "风格": None, "语速": "快速", "基频": "中", "音量": "中", "情感": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
        elif "sad" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": None, "风格": None, "语速": None, "基频": None, "音量": None, "情感": "悲伤", "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
        elif "sichuan" in cid:
            instr_str = json.dumps({"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": "川渝话", "风格": None, "语速": None, "基频": None, "音量": None, "情感": None, "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None}, "IP": None}]}, ensure_ascii=False)
            
        t0 = time.perf_counter()
        wav = runner.generate(
            prompt,
            text,
            instruction=instr_str,
            ip=ip,
            style=style,
            ref_audio=ref_audio,
            prompt_text=prompt_text,
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
            
        unified_data[idx]["tiny_moe"][mkey] = {
            "name": mname,
            "weight_size_mb": size_mb,
            "elapsed_s": round(elapsed, 2),
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "b64": b64
        }
        print(f"  [{cid}] Elapsed: {elapsed:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.3f}")

with open('/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_same_cases_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(unified_data, f, indent=2, ensure_ascii=False)

print("\nAll cases across 0.5b and Tiny generated successfully!")
