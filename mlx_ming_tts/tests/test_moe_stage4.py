"""Stage 4 MoE smoke checks.

These tests avoid loading the full 16.8B weights. They validate the structural
contract that must hold before running the expensive alignment benchmark.
"""

import json
import os

import mlx.core as mx

from mlx_ming_tts.convert import moe_key_map, _load_source
from mlx_ming_tts.ops.moe import SparseMoeBlock
from mlx_ming_tts.pipeline import resolve_dtype


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOE_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-16.8B-A3B")
DENSE_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")


def test_moe_key_map_matches_checkpoint():
    cfg = json.load(open(os.path.join(MOE_DIR, "config.json")))["llm_config"]
    source = set(_load_source(MOE_DIR).keys())
    mapped = moe_key_map(cfg)
    source_keys = []
    for value in mapped.values():
        source_keys.extend(value if isinstance(value, list) else [value])
    missing = [src for src in source_keys if src not in source]
    assert len(mapped) == 311
    assert len(source_keys) == 5603
    assert not missing


def test_pipeline_auto_dtype():
    assert resolve_dtype(DENSE_DIR) == mx.float32
    assert resolve_dtype(MOE_DIR) == mx.bfloat16


def test_sparse_moe_block_forward_smoke():
    block = SparseMoeBlock(
        hidden=8,
        moe_intermediate=4,
        num_experts=3,
        top_k=2,
        num_shared_experts=1,
    )
    out = block(mx.random.normal((2, 5, 8)))
    mx.eval(out)
    assert out.shape == (2, 5, 8)
    assert not bool(mx.isnan(out).any())
