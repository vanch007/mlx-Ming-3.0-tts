
import subprocess, json, os

targets = [
    ("16-bit (bfloat16)", "/Users/vanch/mlx-Ming-3.0-tts/output/compare_bf16.wav", "/Users/vanch/mlx-Ming-3.0-tts/output/asr_bf16_eval"),
    ("8-bit (int8)", "/Users/vanch/mlx-Ming-3.0-tts/output/compare_8bit.wav", "/Users/vanch/mlx-Ming-3.0-tts/output/asr_8bit_eval2"),
    ("4-bit (int4)", "/Users/vanch/mlx-Ming-3.0-tts/output/compare_4bit.wav", "/Users/vanch/mlx-Ming-3.0-tts/output/asr_4bit_eval2"),
]

for name, wav, out_dir in targets:
    cmd = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{wav}' --language zh --format json --output-dir '{out_dir}' --run"
    subprocess.run(cmd, shell=True, check=True)
    base = os.path.basename(wav).replace(".wav", ".confucius.json")
    with open(os.path.join(out_dir, base), 'r', encoding='utf-8') as f:
        t = json.load(f).get('text')
    print(f"[{name}] ASR: {t}")
