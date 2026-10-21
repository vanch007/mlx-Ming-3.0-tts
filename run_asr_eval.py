
import json, os, subprocess

with open('/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval/summary.json') as f:
    items = json.load(f)

for item in items:
    cid = item['id']
    print(f"Evaluating ASR for: {cid}")
    dir_loc = f"/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval/asr_{cid}_local"
    dir_off = f"/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval/asr_{cid}_official"
    
    cmd_local = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['local_wav']}' --language zh --format json --output-dir '{dir_loc}' --run"
    cmd_off = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['official_wav']}' --language zh --format json --output-dir '{dir_off}' --run"
    
    subprocess.run(cmd_local, shell=True, check=True)
    subprocess.run(cmd_off, shell=True, check=True)
    
    fname_loc = os.path.basename(item['local_wav']).replace(".wav", ".confucius.json")
    fname_off = os.path.basename(item['official_wav']).replace(".wav", ".confucius.json")
    
    with open(os.path.join(dir_loc, fname_loc), 'r', encoding='utf-8') as f:
        item['local_asr'] = json.load(f).get('text')
    with open(os.path.join(dir_off, fname_off), 'r', encoding='utf-8') as f:
        item['official_asr'] = json.load(f).get('text')

with open('/Users/vanch/mlx-Ming-3.0-tts/output/official_demo_eval/summary.json', 'w', encoding='utf-8') as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print("ASR evaluations completed!")
