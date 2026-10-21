"""Stage 1 acceptance: MLX Qwen2 vs torch HF Qwen2.

Loads the dense LLM weights into both backends, feeds identical token ids, and
compares per-layer hidden states + final logits. Acceptance gate: cosine > 0.99
at every node.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import mlx.core as mx
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mlx_ming_tts.models.qwen2 import Qwen2ForCausalLM as MLXQwen2
from mlx_ming_tts.convert import convert_llm
from mlx_ming_tts.verify import compare, report

MODEL_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")


def _load_torch_ref(llm_cfg):
    cfg = Qwen2Config(
        hidden_size=llm_cfg["hidden_size"],
        num_hidden_layers=llm_cfg["num_hidden_layers"],
        num_attention_heads=llm_cfg["num_attention_heads"],
        num_key_value_heads=llm_cfg["num_key_value_heads"],
        intermediate_size=llm_cfg["intermediate_size"],
        vocab_size=llm_cfg["vocab_size"],
        rms_norm_eps=llm_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=llm_cfg.get("rope_theta", 1000000.0),
        max_position_embeddings=llm_cfg.get("max_position_embeddings", 32768),
        tie_word_embeddings=llm_cfg.get("tie_word_embeddings", True),
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg)
    # Load weights via mx.load (handles bf16), convert to float32 torch tensors.
    src = mx.load(os.path.join(MODEL_DIR, "model.safetensors"))
    sd = {}
    for k, v in src.items():
        arr = np.array(v.astype(mx.float32), copy=False)
        if k == "model.lm_head.weight":
            sd["lm_head.weight"] = torch.from_numpy(arr)
        elif k.startswith("model.model."):
            sd[k[len("model."):]] = torch.from_numpy(arr)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tie_word_embeddings may leave lm_head as "missing" if tied; acceptable.
    model = model.float().eval()
    return model


def test_qwen2_alignment():
    config = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    llm_cfg = config["llm_config"]

    # MLX model
    mlx_model = MLXQwen2(llm_cfg)
    weights = convert_llm(MODEL_DIR, llm_cfg["num_hidden_layers"], dtype=mx.float32)
    mlx_model.load_weights(list(weights.items()))
    mlx_model.eval()

    # torch reference
    torch_model = _load_torch_ref(llm_cfg)

    # identical input
    rng = np.random.default_rng(0)
    ids = rng.integers(0, llm_cfg["vocab_size"], size=(1, 12)).astype(np.int64)

    mlx_logits, mlx_hidden = mlx_model(input_ids=mx.array(ids))
    mx.eval(mlx_logits)

    with torch.no_grad():
        out = torch_model(torch.from_numpy(ids), output_hidden_states=True)
    torch_logits = out.logits
    torch_hidden = out.hidden_states  # tuple: embeddings + each layer output

    results = []
    # MLX hidden: [emb, L0..L23, final_norm] (n_layers+2 entries).
    # HF hidden:  [emb, after-L0..after-L22, final_norm] (n_layers+1 entries) --
    #   it appends before each layer then once after the final norm, so it omits
    #   L23's raw output and ends with the normed output.
    n_layers = llm_cfg["num_hidden_layers"]
    for i in range(n_layers):  # emb + after-L0..after-L22 align 1:1
        results.append(compare(f"hidden[{i}]", mlx_hidden[i], torch_hidden[i]))
    results.append(compare("final_norm", mlx_hidden[-1], torch_hidden[-1]))
    results.append(compare("logits", mlx_logits, torch_logits))

    all_pass = report(results)
    assert all_pass, "Qwen2 alignment failed at one or more nodes"


if __name__ == "__main__":
    test_qwen2_alignment()
    print("qwen2 alignment: PASS")
