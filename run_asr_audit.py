
import json, os, subprocess

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_same_cases_comparison.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    cases = json.load(f)

AUDIO_DIR = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_audio'
ASR_OUT_DIR = '/Users/vanch/mlx-Ming-3.0-tts/output/asr_audit_tiny'
os.makedirs(ASR_OUT_DIR, exist_ok=True)

audit_results = []

for idx, c in enumerate(cases):
    cid = c['case_id']
    cat = c['category']
    target_text = c['text']
    
    # Determine language
    if 'ja' in cid:
        lang = 'ja'
    elif 'ko' in cid:
        lang = 'ko'
    elif 'fr' in cid:
        lang = 'fr'
    elif 'en' in cid or 'james' in cid:
        lang = 'en'
    else:
        lang = 'zh'
        
    print(f"\n=== [{idx+1:02d}/15] Auditing {cid} ({cat}) [Lang: {lang}] ===")
    print(f"Target Text: {target_text[:60]}")
    
    case_audit = {
        "case_id": cid,
        "category": cat,
        "target_text": target_text,
        "language": lang,
        "variants": {}
    }
    
    for vkey in ['tiny_8bit', 'tiny_4bit', 'tiny_bf16']:
        wav_file = os.path.join(AUDIO_DIR, f"{cid}_{vkey}.wav")
        if not os.path.exists(wav_file):
            print(f"  {vkey}: [WAV file missing]")
            case_audit["variants"][vkey] = {"status": "missing_file", "text": ""}
            continue
            
        run_out_dir = os.path.join(ASR_OUT_DIR, f"{cid}_{vkey}_{lang}")
        os.makedirs(run_out_dir, exist_ok=True)
        
        # Route: if zh/en use confucius (transcribe), if ja/ko/fr use whisper language-id / transcribe
        if lang in ('zh', 'en'):
            cmd = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --audio '{wav_file}' --language {lang} --format json --output-dir '{run_out_dir}' --run"
            json_name = f"{cid}_{vkey}.confucius.json"
        else:
            cmd = f"python3 /Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py --task transcribe --language {lang} --audio '{wav_file}' --format json --output-dir '{run_out_dir}' --run"
            json_name = f"{cid}_{vkey}.json" # whisper or moss
            
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Find json output
        recognized = ""
        for fname in os.listdir(run_out_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(run_out_dir, fname), 'r', encoding='utf-8') as jf:
                        res_data = json.load(jf)
                        recognized = res_data.get('text', '') or res_data.get('raw_text', '')
                except:
                    pass
                break
                
        print(f"  [{vkey}] ASR Output: {repr(recognized)}")
        case_audit["variants"][vkey] = {
            "status": "recognized" if recognized.strip() else "empty_or_hallucinated",
            "asr_text": recognized
        }
        
    audit_results.append(case_audit)

with open('/Users/vanch/mlx-Ming-3.0-tts/output/tiny_asr_full_audit.json', 'w', encoding='utf-8') as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

print("\nFull ASR Audit Completed and saved to output/tiny_asr_full_audit.json!")
