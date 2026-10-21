
import json, os, subprocess

with open('/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/summary.json') as f:
    items = json.load(f)

for item in items:
    cid = item['category_id']
    # Skip music and tta from speech ASR evaluation
    if any(k in cid for k in ('07_music', '09_tta')):
        item['local_asr'] = '[非纯人声音乐/音效]'
        item['official_asr'] = '[非纯人声音乐/音效]'
        continue
        
    print(f"Running ASR on: {cid}")
    dir_loc = f"/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/asr_{cid}_loc"
    dir_off = f"/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/asr_{cid}_off"
    
    lang = "en" if "03_emotion" in cid else "zh"
    cmd_loc = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['local_wav']}' --language {lang} --format json --output-dir '{dir_loc}' --run"
    cmd_off = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{item['official_wav']}' --language {lang} --format json --output-dir '{dir_off}' --run"
    
    subprocess.run(cmd_loc, shell=True, check=True)
    subprocess.run(cmd_off, shell=True, check=True)
    
    fn_loc = os.path.basename(item['local_wav']).replace(".wav", ".confucius.json")
    fn_off = os.path.basename(item['official_wav']).replace(".wav", ".confucius.json")
    
    try:
        with open(os.path.join(dir_loc, fn_loc)) as f:
            item['local_asr'] = json.load(f).get('text')
    except Exception as e:
        item['local_asr'] = str(e)
        
    try:
        with open(os.path.join(dir_off, fn_off)) as f:
            item['official_asr'] = json.load(f).get('text')
    except Exception as e:
        item['official_asr'] = str(e)

with open('/Users/vanch/mlx-Ming-3.0-tts/output/all_features_demo_eval/summary.json', 'w') as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print("Batch ASR evaluation finished!")
