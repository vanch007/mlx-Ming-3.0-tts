"""Rotary position embedding ops.

Standard RoPE for Qwen2 (dense). YARN / 3D-RoPE variants used by the MoE LLM
are added in the MoE stage; this module provides the base rotation that matches
the HF Qwen2 reference (rotate-half convention).
"""

import mlx.core as mx


def precompute_freqs(head_dim, positions, theta=1000000.0):
    """Return (cos, sin) tables for the given integer positions.

    positions: 1-D int array of length T.
    Returns cos, sin each of shape (T, head_dim).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (mx.arange(0, half, dtype=mx.float32) / half))
    pos = positions.astype(mx.float32)
    freqs = mx.outer(pos, inv_freq)  # (T, half)
    emb = mx.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary embedding to query and key.

    q, k: (B, n_heads, T, head_dim). cos, sin: (T, head_dim).
    Uses the HF rotate-half convention to match the torch reference.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


# --- x_transformers-style interleaved RoPE (used by DiT / Aggregator) ---
# This convention differs from HF Qwen2 above: pairs are adjacent (r=2)
# rather than split-half, and freqs are built by interleaving (stack) rather
# than concatenation. Must match x_transformers exactly for Flow alignment.


def xt_rope_freqs(head_dim, seq_len, base=10000.0):
    """Replicate x_transformers RotaryEmbedding.forward_from_seq_len.

    Returns freqs of shape (seq_len, head_dim), where head_dim == dim (full),
    built as interleave(freqs, freqs). scale is fixed 1.0 (no xpos).
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)  # (T, head_dim/2)
    # stack((freqs, freqs), dim=-1) then flatten last two dims -> interleaved
    freqs = mx.stack([freqs, freqs], axis=-1).reshape(seq_len, head_dim)
    return freqs


def _rotate_half_interleaved(x):
    # rearrange(x, '... (d r) -> ... d r', r=2); (-x2, x1); flatten back
    *lead, d = x.shape
    x = x.reshape(*lead, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out = mx.stack([-x2, x1], axis=-1)
    return out.reshape(*lead, d)


def apply_xt_rope(q, k, freqs):
    """Apply interleaved RoPE to q,k of shape (B, H, T, head_dim).

    freqs: (T, head_dim). Mirrors x_transformers.apply_rotary_pos_emb with
    scale=1 and full rotary dim.
    """
    cos = mx.cos(freqs)[None, None, :, :]
    sin = mx.sin(freqs)[None, None, :, :]
    q_out = q * cos + _rotate_half_interleaved(q) * sin
    k_out = k * cos + _rotate_half_interleaved(k) * sin
    return q_out, k_out
