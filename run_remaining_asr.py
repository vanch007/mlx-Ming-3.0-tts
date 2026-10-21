
import json, os, subprocess

with open('/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/summary.json') as f:
    items = json.load(f)

for item in items:
    cid = item['category_id']
    if item.get('local_asr'):
        continue
    dir_loc = f"/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/asr_{cid}_loc"
    dir_off = f"/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/asr_{cid}_off"
    
    cmd_loc = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['local_wav']}' --language zh --format json --output-dir '{dir_loc}' --run"
    cmd_off = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['official_wav']}' --language zh --format json --output-dir '{dir_off}' --run"
    
    subprocess.run(cmd_loc, shell=True, check=True)
    subprocess.run(cmd_off, shell=True, check=True)
    
    fn_loc = os.path.basename(item['local_wav']).replace(".wav", ".confucius.json")
    fn_off = os.path.basename(item['official_wav']).replace(".wav", ".confucius.json")
    
    item['local_asr'] = json.load(open(os.path.join(dir_loc, fn_loc))).get('text')
    item['official_asr'] = json.load(open(os.path.join(dir_off, fn_off))).get('text')

with open('/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/summary.json', 'w') as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print("Remaining ASR finished!")
