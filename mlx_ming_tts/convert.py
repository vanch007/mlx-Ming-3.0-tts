"""Offline weight conversion: HF safetensors -> MLX safetensors.

Reads HF weights via mx.load (bfloat16/float32 supported natively) and remaps
key names to the MLX module layout. Supports both component-level export and
full end-to-end model conversion with mlx_model_config.json.
"""

import argparse
import glob
import json
import os
import shutil

import mlx.core as mx


def _load_source(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return mx.load(single)

    shards = sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    merged = {}
    for s in shards:
        merged.update(mx.load(s))
    return merged


def llm_key_map(num_layers, tie_embed=True):
    m = {
        "model.embed_tokens.weight": "model.model.embed_tokens.weight",
        "model.norm.weight": "model.model.norm.weight",
    }
    if not tie_embed:
        m["lm_head.weight"] = "model.lm_head.weight"

    for i in range(num_layers):
        src = f"model.model.layers.{i}"
        dst = f"model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj"):
            m[f"{dst}.self_attn.{proj}.weight"] = f"{src}.self_attn.{proj}.weight"
            m[f"{dst}.self_attn.{proj}.bias"] = f"{src}.self_attn.{proj}.bias"
        m[f"{dst}.self_attn.o_proj.weight"] = f"{src}.self_attn.o_proj.weight"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            m[f"{dst}.mlp.{proj}.weight"] = f"{src}.mlp.{proj}.weight"
        m[f"{dst}.input_layernorm.weight"] = f"{src}.input_layernorm.weight"
        m[f"{dst}.post_attention_layernorm.weight"] = f"{src}.post_attention_layernorm.weight"
    return m


def convert_llm(model_dir, num_layers, dtype=mx.float32, tie_embed=True):
    src = _load_source(model_dir)
    available = set(src.keys())
    key_map = llm_key_map(num_layers, tie_embed=tie_embed)

    missing = [s for s in key_map.values() if s not in available]
    if missing:
        raise KeyError(f"{len(missing)} source keys missing, e.g. {missing[:5]}")

    out = {}
    for dst, s in key_map.items():
        out[dst] = src[s].astype(dtype)
    if tie_embed:
        out["lm_head.weight"] = src["model.model.embed_tokens.weight"].astype(dtype)
    return out


def convert_by_prefix(model_dir, prefix, mlx_param_names, dtype=mx.float32):
    src = _load_source(model_dir)
    available = set(src.keys())
    out = {}
    missing = []
    for name in mlx_param_names:
        key = f"{prefix}.{name}"
        if key not in available:
            missing.append(key)
            continue
        out[name] = src[key].astype(dtype)
    if missing:
        raise KeyError(f"{len(missing)} source keys missing under '{prefix}', e.g. {missing[:5]}")
    return out


def moe_key_map(cfg):
    n_layers = cfg["num_hidden_layers"]
    n_experts = cfg["num_experts"]
    has_shared = bool(cfg.get("num_shared_experts"))
    use_qkv_bias = cfg.get("use_qkv_bias", False)
    use_bias = cfg.get("use_bias", False)

    m = {
        "model.word_embeddings.weight": "model.model.word_embeddings.weight",
        "model.norm.weight": "model.model.norm.weight",
        "lm_head.weight": "model.lm_head.weight",
    }
    for i in range(n_layers):
        src = f"model.model.layers.{i}"
        dst = f"model.layers.{i}"
        m[f"{dst}.input_layernorm.weight"] = f"{src}.input_layernorm.weight"
        m[f"{dst}.post_attention_layernorm.weight"] = f"{src}.post_attention_layernorm.weight"
        m[f"{dst}.attention.query_key_value.weight"] = f"{src}.attention.query_key_value.weight"
        m[f"{dst}.attention.dense.weight"] = f"{src}.attention.dense.weight"
        if use_qkv_bias:
            m[f"{dst}.attention.query_key_value.bias"] = f"{src}.attention.query_key_value.bias"
        if use_bias:
            m[f"{dst}.attention.dense.bias"] = f"{src}.attention.dense.bias"
        m[f"{dst}.mlp.gate.weight"] = f"{src}.mlp.gate.weight"
        m[f"{dst}.mlp.experts_gate_proj_weight"] = [
            f"{src}.mlp.experts.{e}.gate_proj.weight" for e in range(n_experts)
        ]
        m[f"{dst}.mlp.experts_up_proj_weight"] = [
            f"{src}.mlp.experts.{e}.up_proj.weight" for e in range(n_experts)
        ]
        m[f"{dst}.mlp.experts_down_proj_weight"] = [
            f"{src}.mlp.experts.{e}.down_proj.weight" for e in range(n_experts)
        ]
        if has_shared:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                m[f"{dst}.mlp.shared_experts.{proj}.weight"] = f"{src}.mlp.shared_experts.{proj}.weight"
    return m


def convert_moe(model_dir, cfg, dtype=mx.bfloat16):
    src = _load_source(model_dir)
    available = set(src.keys())
    key_map = moe_key_map(cfg)
    source_keys = []
    for value in key_map.values():
        source_keys.extend(value if isinstance(value, list) else [value])
    missing = [s for s in source_keys if s not in available]
    if missing:
        raise KeyError(f"{len(missing)} MoE source keys missing, e.g. {missing[:5]}")
    out = {}
    for dst, s in key_map.items():
        if isinstance(s, list):
            out[dst] = mx.stack([src[key].astype(dtype) for key in s], axis=0)
        else:
            out[dst] = src[s].astype(dtype)
    return out


def convert_vae(model_dir, mlx_param_names, dtype=mx.float32):
    src = _load_source(model_dir)
    available = set(src.keys())
    out = {}
    missing = []
    for name in mlx_param_names:
        if name == "decoder.head_out.weight":
            key = "audio.decoder.head.out.weight"
        elif name == "decoder.head_out.bias":
            key = "audio.decoder.head.out.bias"
        else:
            key = f"audio.{name}"
        if key not in available:
            missing.append(key)
            continue
        out[name] = src[key].astype(dtype)
    if missing:
        raise KeyError(f"{len(missing)} VAE source keys missing, e.g. {missing[:5]}")
    return out


def convert_full_model(hf_dir, out_dir, dtype=mx.bfloat16):
    """Convert the full Ming-3.0-tts model to standard MLX safetensors."""
    from mlx.utils import tree_flatten
    from transformers import AutoTokenizer
    from mlx_ming_tts.models.bailingmm import BailingMM

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(hf_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. Build model skeleton
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = BailingMM(hf_dir, tok)
    model.load_all_weights(dtype=dtype)

    # 2. Extract weights
    weights = dict(tree_flatten(model.parameters()))

    # 3. Save safetensors
    weights_filename = "mlx_weights.safetensors"
    weights_path = os.path.join(out_dir, weights_filename)
    mx.save_safetensors(weights_path, weights)
    print(f"Saved {len(weights)} MLX parameters to {weights_path}")

    # 4. Copy config files, tokenizer files, etc.
    for fname in os.listdir(hf_dir):
        src_f = os.path.join(hf_dir, fname)
        if os.path.isfile(src_f) and not fname.endswith(".safetensors") and not fname.endswith(".bin"):
            shutil.copy2(src_f, os.path.join(out_dir, fname))

    # 5. Write mlx_model_config.json
    mlx_config = {
        "model_type": "ming_3.0_tts",
        "weights_file": weights_filename,
        "dtype": "bfloat16" if dtype == mx.bfloat16 else ("float16" if dtype == mx.float16 else "float32"),
        "sample_rate": cfg.get("audio_tokenizer_config", {}).get("sample_rate", 44100),
    }
    with open(os.path.join(out_dir, "mlx_model_config.json"), "w", encoding="utf-8") as f:
        json.dump(mlx_config, f, indent=2, ensure_ascii=False)
    print(f"Generated {os.path.join(out_dir, 'mlx_model_config.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    # legacy args
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--component", default="llm", choices=["llm", "moe"])
    ap.add_argument("--num-layers", type=int, default=24)
    args = ap.parse_args()

    dtype_map = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
    chosen_dtype = dtype_map[args.dtype]

    if args.hf_dir and args.out_dir:
        convert_full_model(args.hf_dir, args.out_dir, dtype=chosen_dtype)
    elif args.model_dir and args.out:
        if args.component == "llm":
            weights = convert_llm(args.model_dir, args.num_layers, chosen_dtype)
        elif args.component == "moe":
            with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)["llm_config"]
            weights = convert_moe(args.model_dir, cfg, chosen_dtype)
        mx.save_safetensors(args.out, weights)
        print(f"saved {len(weights)} tensors -> {args.out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
