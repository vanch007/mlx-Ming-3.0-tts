"""Attention ops: scaled dot-product attention with GQA and sliding window support."""

import math

import mlx.core as mx


def repeat_kv(x, n_rep):
    """Expand KV heads for grouped-query attention.

    x: (B, n_kv_heads, T, head_dim) -> (B, n_kv_heads * n_rep, T, head_dim).
    """
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    x = mx.expand_dims(x, axis=2)
    x = mx.broadcast_to(x, (B, n_kv, n_rep, T, D))
    return x.reshape(B, n_kv * n_rep, T, D)


def sdpa(q, k, v, scale=None, mask=None):
    """Scaled dot-product attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def causal_mask(T, dtype=mx.float32, sliding_window=None):
    """Additive causal mask with optional sliding window."""
    idx_q = mx.arange(T)[:, None]
    idx_k = mx.arange(T)[None, :]
    causal = idx_k <= idx_q
    if sliding_window is not None:
        window = idx_k >= (idx_q - sliding_window)
        keep = causal & window
    else:
        keep = causal
    return mx.where(keep, mx.array(0.0, dtype=dtype), mx.array(-float("inf"), dtype=dtype))
