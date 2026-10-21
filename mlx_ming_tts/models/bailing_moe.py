"""Bailing MoE LLM in MLX (16.8B-A3B).

Mirrors modeling_bailing_moe.py for text/audio generation. Fused QKV attention
(no bias), GQA, RMSNorm, 64 routed SwiGLU experts with top-6 routing, and 2
shared experts. Uses standard half-split RoPE: the checkpoint declares
rope_scaling type "3D", but this TTS path passes no image/video grids, so
get_rope_index gives identical positions on all three axes and 3D mRoPE
collapses to standard RoPE (theta=600000). Every layer is MoE
(first_k_dense_replace=0).
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_ming_tts.ops.norm import RMSNorm
from mlx_ming_tts.ops.rope import precompute_freqs, apply_rope
from mlx_ming_tts.ops.attention import sdpa, repeat_kv, causal_mask
from mlx_ming_tts.ops.moe import SparseMoeBlock
from mlx_ming_tts.models.qwen2 import KVCache


class MoeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.hidden // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        self.rope_theta = cfg.get("rope_theta", 600000.0)
        qkv_out = (self.n_heads + 2 * self.n_kv) * self.head_dim
        use_qkv_bias = cfg.get("use_qkv_bias", False)
        use_bias = cfg.get("use_bias", False)
        self.query_key_value = nn.Linear(self.hidden, qkv_out, bias=use_qkv_bias)
        self.dense = nn.Linear(self.n_heads * self.head_dim, self.hidden, bias=use_bias)

    def __call__(self, x, positions, mask=None, cache=None):
        B, T, _ = x.shape
        qkv = self.query_key_value(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = precompute_freqs(self.head_dim, positions, self.rope_theta)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = sdpa(q, k, v, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.dense(out)


class MoeDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg.get("rms_norm_eps", 1e-5)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.attention = MoeAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.mlp = SparseMoeBlock(
            hidden=cfg["hidden_size"], moe_intermediate=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"], top_k=cfg["num_experts_per_tok"],
            num_shared_experts=cfg.get("num_shared_experts") or 0,
            norm_topk_prob=cfg.get("norm_topk_prob", True),
        )

    def __call__(self, x, positions, mask=None, cache=None):
        x = x + self.attention(self.input_layernorm(x), positions, mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class BailingMoeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [MoeDecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-5))

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        B, T, _ = inputs_embeds.shape
        if positions is None:
            offset = caches[0].offset if caches is not None else 0
            positions = mx.arange(offset, offset + T)
        mask = causal_mask(T, dtype=inputs_embeds.dtype) if T > 1 else None

        hidden_states = [inputs_embeds]
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, positions, mask, cache)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states.append(x)
        return x, hidden_states


class BailingMoeForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = BailingMoeModel(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        x, hidden = self.model(input_ids, inputs_embeds, positions, caches)
        return self.lm_head(x), hidden

    def make_caches(self):
        return [KVCache() for _ in range(self.cfg["num_hidden_layers"])]
