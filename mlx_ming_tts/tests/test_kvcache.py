"""Stage 1: KVCache incremental decoding must equal a single full forward.

Validates the cache path used by the AR loop: prefill a prompt, then decode
one token at a time, and check the per-step logits match the full-sequence
forward at the corresponding positions.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import mlx.core as mx

from mlx_ming_tts.models.qwen2 import Qwen2ForCausalLM as MLXQwen2
from mlx_ming_tts.convert import convert_llm
from mlx_ming_tts.verify import cosine_similarity

MODEL_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")


def test_kvcache_matches_full_forward():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))["llm_config"]
    model = MLXQwen2(cfg)
    weights = convert_llm(MODEL_DIR, cfg["num_hidden_layers"], dtype=mx.float32)
    model.load_weights(list(weights.items()))
    model.eval()

    rng = np.random.default_rng(1)
    ids = rng.integers(0, cfg["vocab_size"], size=(1, 10)).astype(np.int64)
    ids_mx = mx.array(ids)

    # Full forward
    full_logits, _ = model(input_ids=ids_mx)
    mx.eval(full_logits)

    # Cached: prefill first 6 tokens, then decode the rest one at a time
    caches = model.make_caches()
    prefill = 6
    step_logits = []
    pre_logits, _ = model(input_ids=ids_mx[:, :prefill], caches=caches)
    mx.eval(pre_logits)
    step_logits.append(pre_logits[:, -1:, :])
    for t in range(prefill, ids.shape[1]):
        lg, _ = model(input_ids=ids_mx[:, t : t + 1], caches=caches)
        mx.eval(lg)
        step_logits.append(lg)

    # Compare last-token logits at each decoded position
    cos_prefill = cosine_similarity(step_logits[0], full_logits[:, prefill - 1 : prefill, :])
    print(f"prefill last-token cosine: {cos_prefill:.6f}")
    assert cos_prefill > 0.99

    for i, t in enumerate(range(prefill, ids.shape[1]), start=1):
        cos = cosine_similarity(step_logits[i], full_logits[:, t : t + 1, :])
        print(f"step t={t} cosine: {cos:.6f}")
        assert cos > 0.99, f"step {t} cosine {cos} below 0.99"


if __name__ == "__main__":
    test_kvcache_matches_full_forward()
    print("kvcache: PASS")
