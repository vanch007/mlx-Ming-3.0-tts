"""Configuration loading and dtype strategy for the MLX pipeline.

dtype strategy (Apple Silicon, see project .ai_memory.md):
  - Default to float32 on MLX. Qwen2 RMSNorm/attention overflow float16 -> NaN.
  - ISTFT exp(mag) must stay float32 (float16 max ~65504 overflows).
  - bfloat16 is reserved for CUDA, which is not the target platform here.
"""

import json
import os

import mlx.core as mx


def default_dtype():
    """Return the safe default compute dtype for this platform.

    float32 on Apple Silicon to avoid the float16 NaN issues documented for
    Qwen2 RMSNorm/attention in this project.
    """
    return mx.float32


def load_config(model_dir):
    """Load the top-level model config.json from a model directory."""
    path = os.path.join(model_dir, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_llm_config(config):
    """Extract the LLM sub-config (Qwen2 dense or Bailing MoE)."""
    return config.get("llm_config", {})


def is_moe(config):
    """True when the top-level model is the MoE variant."""
    llm = get_llm_config(config)
    return config.get("model_type") == "moe" or bool(llm.get("num_experts"))
