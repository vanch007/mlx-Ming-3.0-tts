"""Qwen2 dense LLM in MLX (also reused as the AudioVAE backbone).

Mirrors the HF Qwen2 reference: QKV with bias, o_proj without bias, SwiGLU MLP,
RMSNorm, GQA, rotate-half RoPE. KVCache replaces torch DynamicCache.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_ming_tts.ops.norm import RMSNorm
from mlx_ming_tts.ops.rope import precompute_freqs, apply_rope
from mlx_ming_tts.ops.attention import sdpa, repeat_kv, causal_mask


class KVCache:
    """Pre-allocated, growing key/value cache (mlx-lm style).

    Stores keys/values per layer and grows by step along the time axis to
    avoid reallocating every decode step.
    """

    def __init__(self, step=256):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys, values):
        B, H, T, D = keys.shape
        if self.keys is None or (self.offset + T) > self.keys.shape[2]:
            new_cap = ((self.offset + T + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, new_cap, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H, new_cap, D), dtype=values.dtype)
            if self.keys is not None:
                new_k[..., : self.offset, :] = self.keys[..., : self.offset, :]
                new_v[..., : self.offset, :] = self.values[..., : self.offset, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., self.offset : self.offset + T, :] = keys
        self.values[..., self.offset : self.offset + T, :] = values
        self.offset += T
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class Qwen2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.rope_theta = cfg.get("rope_theta", 1000000.0)

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def __call__(self, x, positions, mask=None, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = precompute_freqs(self.head_dim, positions, self.rope_theta)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = sdpa(q, k, v, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("rms_norm_eps", 1e-6)
        self.self_attn = Qwen2Attention(cfg)
        self.mlp = Qwen2MLP(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps)

    def __call__(self, x, positions, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), positions, mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    """Qwen2 transformer trunk (embeddings -> layers -> final norm)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [Qwen2DecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
        self.sliding_window = cfg.get("sliding_window") if cfg.get("use_sliding_window") else None

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, T, _ = inputs_embeds.shape

        if positions is None:
            offset = caches[0].offset if caches is not None else 0
            positions = mx.arange(offset, offset + T)

        mask = causal_mask(T, dtype=inputs_embeds.dtype, sliding_window=self.sliding_window) if T > 1 else None

        hidden_states = [inputs_embeds]
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, positions, mask, cache)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states.append(x)
        return x, hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        x, hidden_states = self.model(input_ids, inputs_embeds, positions, caches)
        logits = self.lm_head(x)
        return logits, hidden_states

    def make_caches(self):
        return [KVCache() for _ in range(self.cfg["num_hidden_layers"])]
