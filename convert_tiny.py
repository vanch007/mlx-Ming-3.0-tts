
import json, os, time, sys, gc
import mlx.core as mx
import shutil

SOURCE_DIR = "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-tiny-hf"
INDEX_PATH = os.path.join(SOURCE_DIR, "model.safetensors.index.json")

def convert_tiny_to_mlx(out_dir, dtype=mx.bfloat16):
    os.makedirs(out_dir, exist_ok=True)
    out_weights_path = os.path.join(out_dir, "mlx_weights.safetensors")
    print(f"\nConverting Tiny HF shards to {out_weights_path} with dtype={dtype}...")
    
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
        
    shards = sorted(set(weight_map.values()))
    print(f"Total shards to process: {len(shards)}")
    
    converted = {}
    
    for s_idx, sname in enumerate(shards):
        t0 = time.perf_counter()
        shard_path = os.path.join(SOURCE_DIR, sname)
        shard_weights = mx.load(shard_path)
        
        for k, v in shard_weights.items():
            dst_k = k
            if dst_k == "audio.decoder.head.out.weight":
                dst_k = "audio.decoder.head_out.weight"
            elif dst_k == "audio.decoder.head.out.bias":
                dst_k = "audio.decoder.head_out.bias"
            
            arr = v
            if arr.dtype != dtype and arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                if "norm" not in dst_k and "A_log" not in dst_k and "dt_bias" not in dst_k and "expert_bias" not in dst_k:
                    arr = arr.astype(dtype)
            converted[dst_k] = arr
            
        elapsed = time.perf_counter() - t0
        print(f"[{s_idx+1:02d}/{len(shards)}] Loaded {sname} ({len(converted)} total keys, {elapsed:.2f}s)")
        del shard_weights
        gc.collect()

    print(f"\nSaving all {len(converted)} tensors to {out_weights_path}...")
    t_save = time.perf_counter()
    mx.save_safetensors(out_weights_path, converted)
    print(f"Saved {os.path.getsize(out_weights_path)/1e9:.2f} GB in {time.perf_counter()-t_save:.1f}s!")
    
    # Copy auxiliary files
    for fname in os.listdir(SOURCE_DIR):
        src_f = os.path.join(SOURCE_DIR, fname)
        if os.path.isfile(src_f) and not fname.endswith(".safetensors") and not fname.endswith(".incomplete"):
            shutil.copy2(src_f, os.path.join(out_dir, fname))
            
    # Write mlx_model_config.json
    mlx_config = {
        "model_type": "lingv3_tts",
        "architecture": "tiny_moe",
        "weights_file": "mlx_weights.safetensors",
        "dtype": "bfloat16" if dtype == mx.bfloat16 else str(dtype),
        "sample_rate": 44100
    }
    with open(os.path.join(out_dir, "mlx_model_config.json"), "w", encoding="utf-8") as f:
        json.dump(mlx_config, f, indent=2, ensure_ascii=False)
    print(f"Done creating MLX bundle at {out_dir}!")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-tiny-mlx-bf16"
    convert_tiny_to_mlx(out_dir)
