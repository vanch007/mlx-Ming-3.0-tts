
import json, os, time, sys, shutil, gc
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
from mlx.utils import tree_flatten

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
from mlx_ming_tts.models.lingv3_tts import LingV3TTS

SRC_DIR = "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-tiny-mlx-bf16"

def quantize_and_save(out_dir, bits=8, group_size=64):
    os.makedirs(out_dir, exist_ok=True)
    out_weights_path = os.path.join(out_dir, "mlx_weights.safetensors")
    print(f"\nQuantizing Tiny to {bits}-bit -> {out_weights_path}...")
    
    tok = AutoTokenizer.from_pretrained(SRC_DIR, trust_remote_code=True)
    model = LingV3TTS(SRC_DIR, tok)
    print("Loading bf16 weights into skeleton...")
    model.load_mlx_weights(os.path.join(SRC_DIR, "mlx_weights.safetensors"))
    
    def predicate(path, m):
        if not isinstance(m, nn.Linear):
            return False
        if any(k in path for k in ('head_out', 'stop_head', 'spk_head')):
            return False
        if m.weight.shape[-1] % group_size != 0:
            return False
        return True

    print(f"Applying {bits}-bit quantization (group_size={group_size})...")
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    
    print("Extracting quantized parameters...")
    weights = dict(tree_flatten(model.parameters()))
    print(f"Saving {len(weights)} quantized tensors to {out_weights_path}...")
    t0 = time.perf_counter()
    mx.save_safetensors(out_weights_path, weights)
    sz_gb = os.path.getsize(out_weights_path) / 1e9
    print(f"Saved {sz_gb:.2f} GB in {time.perf_counter()-t0:.1f}s!")
    
    # Copy aux files
    for fname in os.listdir(SRC_DIR):
        src_f = os.path.join(SRC_DIR, fname)
        if os.path.isfile(src_f) and not fname.endswith(".safetensors"):
            shutil.copy2(src_f, os.path.join(out_dir, fname))
            
    # Write mlx_model_config.json
    cfg_data = {
        "model_type": "lingv3_tts",
        "architecture": "tiny_moe",
        "weights_file": "mlx_weights.safetensors",
        "quantization": {
            "bits": bits,
            "group_size": group_size
        },
        "sample_rate": 44100
    }
    with open(os.path.join(out_dir, "mlx_model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)
    print(f"Done exporting {bits}-bit model to {out_dir}!")
    del model, weights
    gc.collect()

if __name__ == "__main__":
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    d = sys.argv[2] if len(sys.argv) > 2 else f"/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-tiny-mlx-{b}bit"
    quantize_and_save(d, bits=b)
