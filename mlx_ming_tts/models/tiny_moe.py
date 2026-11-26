"""Bailing MoE v3 (LingV3 TTS Tiny) MLX implementation with persistent step caching.

Contains:
1. KDACache for persistent conv1d and recurrent state tracking
2. MLACache for KV caching in MLA attention layers
3. ShortConv1d with state caching
4. KimiDeltaAttention with recurrent state caching
5. MultiLatentAttention with MLACache
6. BailingMoeGate with group-limited topk and expert_bias
7. BailingMoeV3SparseMoeBlock with 128 routed experts + 1 shared expert
8. BailingMoeV3Model and BailingMoeV3ForCausalLM with make_caches()
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple

from mlx_ming_tts.ops.norm import RMSNorm
from mlx_ming_tts.ops.rope import precompute_freqs, apply_rope
from mlx_ming_tts.ops.attention import sdpa


class KDACache:
    """Persistent state cache for KimiDeltaAttention layer."""
    def __init__(self):
        self.conv_q = None
        self.conv_k = None
        self.conv_v = None
        self.recurrent_state = None  # S: (B, H, D, D)


class MLACache:
    """KV cache for MultiLatentAttention layer."""
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
            new_v = mx.zeros((B, H, new_cap, values.shape[-1]), dtype=values.dtype)
            if self.keys is not None:
                new_k[..., : self.offset, :] = self.keys[..., : self.offset, :]
                new_v[..., : self.offset, :] = self.values[..., : self.offset, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., self.offset : self.offset + T, :] = keys
        self.values[..., self.offset : self.offset + T, :] = values
        self.offset += T
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class ShortConv1d(nn.Module):
    """Depthwise 1D causal convolution with step caching."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, 1, kernel_size))

    def __call__(self, x, cache=None):
        B, T, C = x.shape
        W = self.kernel_size
        w = self.weight[:, 0, :]  # (C, W)

        if cache is not None and cache.shape[-1] == (W - 1) and T == 1:
            x_step = x.transpose(0, 2, 1)  # (B, C, 1)
            full = mx.concatenate([cache, x_step], axis=-1)  # (B, C, W)
            new_cache = full[:, :, 1:]
            out = mx.sum(full * w[None, :, :], axis=-1, keepdims=True)  # (B, C, 1)
            out = nn.silu(out.transpose(0, 2, 1))
            return out, new_cache
        else:
            pad = mx.zeros((B, W - 1, C), dtype=x.dtype)
            x_padded = mx.concatenate([pad, x], axis=1)  # (B, T + W - 1, C)
            windows = mx.stack([x_padded[:, t : t + W, :] for t in range(T)], axis=1)
            out = mx.sum(windows * w[None, None, :, :].transpose(0, 1, 3, 2), axis=2)
            final_cache = x_padded[:, - (W - 1):, :].transpose(0, 2, 1) if W > 1 else None
            return nn.silu(out), final_cache


class KimiDeltaAttention(nn.Module):
    """KDA Linear Attention with state caching."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.head_dim = cfg.get("head_dim", 128)
        self.num_heads = cfg["num_attention_heads"]
        self.layer_idx = layer_idx
        self.conv_size = cfg.get("short_conv_kernel_size", 4)
        self.lower_bound = cfg.get("kda_lower_bound", -5.0)

        proj_size = self.head_dim * self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, proj_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, proj_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, proj_size, bias=False)

        self.q_conv1d = ShortConv1d(proj_size, self.conv_size)
        self.k_conv1d = ShortConv1d(proj_size, self.conv_size)
        self.v_conv1d = ShortConv1d(proj_size, self.conv_size)

        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((proj_size,))

        self.f_proj = nn.Linear(self.hidden_size, proj_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, proj_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.o_norm = nn.RMSNorm(self.head_dim, eps=cfg.get("rms_norm_eps", 1e-6))
        self.o_proj = nn.Linear(proj_size, self.hidden_size, bias=False)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, cache: Optional[KDACache] = None):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)

        cq = cache.conv_q if cache is not None else None
        ck = cache.conv_k if cache is not None else None
        cv = cache.conv_v if cache is not None else None
        S = cache.recurrent_state if cache is not None and cache.recurrent_state is not None else mx.zeros((B, H, D, D), dtype=mx.float32)

        q, new_cq = self.q_conv1d(q_raw, cq)
        k, new_ck = self.k_conv1d(k_raw, ck)
        v, new_cv = self.v_conv1d(v_raw, cv)

        f = self.f_proj(x)
        beta = mx.sigmoid(self.b_proj(x))

        q = q.reshape(B, T, H, D)
        k = k.reshape(B, T, H, D)
        # Apply L2-Norm on q and k before scaling
        q = (q / mx.sqrt(mx.sum(q**2, axis=-1, keepdims=True) + 1e-6)) * self.scale
        k = k / mx.sqrt(mx.sum(k**2, axis=-1, keepdims=True) + 1e-6)
        v = v.reshape(B, T, H, D)
        f = f.reshape(B, T, H, D)

        dt_bias = self.dt_bias.reshape(H, D)
        A_val = mx.exp(self.A_log)[None, None, :, None]
        g = self.lower_bound * mx.sigmoid(A_val * (f + dt_bias[None, None, :, :]))

        o_list = []
        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = g[:, t]
            b_t = beta[:, t]

            S = S * mx.exp(g_t[..., None])
            k_proj_S = mx.sum(k_t[..., None] * S, axis=-2)
            kv_diff = v_t - k_proj_S
            S = S + b_t[..., None, None] * k_t[..., None] * kv_diff[..., None, :]
            out_t = mx.sum(q_t[..., None] * S, axis=-2)
            o_list.append(out_t[:, None, :, :])

        o = mx.concatenate(o_list, axis=1)

        g_out = mx.sigmoid(self.g_proj(x).reshape(B, T, H, D))
        o = self.o_norm(o) * g_out
        o = o.reshape(B, T, H * D)
        out = self.o_proj(o)

        if cache is not None:
            cache.conv_q = new_cq
            cache.conv_k = new_ck
            cache.conv_v = new_cv
            cache.recurrent_state = S

        return out


class MultiLatentAttention(nn.Module):
    """MLA layer with LoRA projection and head-wise gating."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.qk_nope_head_dim = cfg.get("qk_nope_head_dim", 128)
        self.qk_rope_head_dim = cfg.get("qk_rope_head_dim", 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg.get("v_head_dim", 128)
        self.q_lora_rank = cfg.get("q_lora_rank", 256)
        self.kv_lora_rank = cfg.get("kv_lora_rank", 512)
        self.layer_idx = layer_idx
        self.rope_theta = cfg.get("rope_theta", 10000.0)
        self.rope_interleave = bool(cfg.get("rope_interleave", False))

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=cfg.get("rms_norm_eps", 1e-6))
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=cfg.get("rms_norm_eps", 1e-6))
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.dense = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.scaling = self.qk_head_dim ** -0.5

    @staticmethod
    def _deinterleave(x):
        b, h, t, d = x.shape
        return x.reshape(b, h, t, d // 2, 2).swapaxes(-1, -2).reshape(b, h, t, d)

    def __call__(self, x, positions, mask=None, cache: Optional[MLACache] = None):
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, T, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_pass = q[..., :self.qk_nope_head_dim]
        q_rot = q[..., self.qk_nope_head_dim:]

        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass_raw = compressed_kv[..., :self.kv_lora_rank]
        k_rot = compressed_kv[..., self.kv_lora_rank:]

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass_raw))
        k_pass = k_pass.reshape(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(0, 2, 1, 3)
        k_pass_nope = k_pass[..., :self.qk_nope_head_dim]
        v = k_pass[..., self.qk_nope_head_dim:]

        k_rot = k_rot.reshape(B, T, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        if self.rope_interleave:
            # Official apply_rotary_pos_emb_interleave: regroup adjacent pairs (view d/2,2 ->
            # transpose -> reshape) before the rotate-half rotation.
            q_rot = self._deinterleave(q_rot)
            k_rot = self._deinterleave(k_rot)
        cos, sin = precompute_freqs(self.qk_rope_head_dim, positions, self.rope_theta)
        q_rot, k_rot = apply_rope(q_rot, k_rot, cos, sin)
        k_rot = mx.broadcast_to(k_rot, (B, self.num_heads, T, self.qk_rope_head_dim))

        q_full = mx.concatenate([q_pass, q_rot], axis=-1)
        k_full = mx.concatenate([k_pass_nope, k_rot], axis=-1)

        if cache is not None:
            k_full, v = cache.update_and_fetch(k_full, v)

        if mask is None and T > 1:
            # The official MLA is causal (is_causal=True); without a mask MLX attends both ways
            # over the whole prefill. Query i sits at absolute position (Tk - T + i).
            Tk = k_full.shape[2]
            q_pos = mx.arange(Tk - T, Tk)[:, None]
            k_pos = mx.arange(Tk)[None, :]
            mask = mx.where(k_pos <= q_pos, mx.array(0.0, dtype=q_full.dtype), mx.array(-float("inf"), dtype=q_full.dtype))

        attn_out = sdpa(q_full, k_full, v, scale=self.scaling, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3)

        gate = mx.sigmoid(self.g_proj(x))[:, :, :, None]
        attn_out = attn_out * gate
        attn_out = attn_out.reshape(B, T, -1)
        return self.dense(attn_out)


class BailingMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingMoeGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.n_group = cfg.get("n_group", 8)
        self.topk_group = cfg.get("topk_group", 4)
        self.routed_scaling_factor = cfg.get("routed_scaling_factor", 2.5)
        self.hidden_size = cfg["hidden_size"]

        self.weight = mx.zeros((self.num_experts, self.hidden_size))
        self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, x):
        B, T, C = x.shape
        flat_x = x.reshape(-1, C)
        # Official gate: logits, sigmoid and expert_bias in fp32 (router_dtype=fp32); bf16 logits
        # can flip Top-K expert choices.
        logits = flat_x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        scores_for_routing = scores + self.expert_bias.astype(mx.float32)[None, :]

        N = flat_x.shape[0]
        group_size = self.num_experts // self.n_group
        grouped = scores_for_routing.reshape(N, self.n_group, group_size)
        sorted_in_group = mx.sort(grouped, axis=-1)[:, :, -2:]
        group_scores = mx.sum(sorted_in_group, axis=-1)

        group_idx = mx.argpartition(-group_scores, kth=self.topk_group - 1, axis=-1)[:, :self.topk_group]

        mask = mx.zeros((N, self.num_experts), dtype=mx.bool_)
        for n in range(N):
            for g_id in group_idx[n].tolist():
                start = g_id * group_size
                mask[n, start : start + group_size] = True

        masked_scores = mx.where(mask, scores_for_routing, mx.array(-float("inf"), dtype=scores.dtype))
        topk_idx = mx.argpartition(-masked_scores, kth=self.top_k - 1, axis=-1)[:, :self.top_k]

        topk_scores = mx.take_along_axis(scores, topk_idx, axis=-1)
        weights = topk_scores / (mx.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return topk_idx, weights


class BailingSparseMoeBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.gate = BailingMoeGate(cfg)
        h = cfg["hidden_size"]
        inter = cfg["moe_intermediate_size"]

        self.experts = [BailingMLP(h, inter) for _ in range(self.num_experts)]
        self.shared_experts = BailingMLP(h, cfg["moe_shared_expert_intermediate_size"])

    def __call__(self, x):
        B, T, C = x.shape
        flat_x = x.reshape(-1, C)
        N = flat_x.shape[0]

        shared_out = self.shared_experts(flat_x)
        topk_idx, weights = self.gate(x)

        out = mx.zeros_like(flat_x)
        topk_idx_list = topk_idx.tolist()
        weights_list = weights.tolist()

        for n in range(N):
            x_n = flat_x[n:n+1]
            token_out = mx.zeros((1, C), dtype=x.dtype)
            for k in range(self.top_k):
                e_id = topk_idx_list[n][k]
                w = weights_list[n][k]
                token_out = token_out + w * self.experts[e_id](x_n)
            out[n:n+1] = token_out

        out = out + shared_out
        return out.reshape(B, T, C)


class TinyDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mla = (layer_idx + 1) % cfg.get("layer_group_size", 4) == 0
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))

        if self.is_mla:
            self.attention = MultiLatentAttention(cfg, layer_idx)
        else:
            self.attention = KimiDeltaAttention(cfg, layer_idx)

        if layer_idx < cfg.get("first_k_dense_replace", 1):
            self.mlp = BailingMLP(cfg["hidden_size"], cfg["intermediate_size"])
        else:
            self.mlp = BailingSparseMoeBlock(cfg)

    def __call__(self, x, positions, mask=None, cache=None):
        normed = self.input_layernorm(x)
        if self.is_mla:
            attn_out = self.attention(normed, positions, mask, cache=cache)
        else:
            attn_out = self.attention(normed, cache=cache)

        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class BailingMoeV3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [TinyDecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        self.norm = RMSNorm(cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6))

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        B, T, _ = inputs_embeds.shape

        if positions is None:
            offset = caches[3].offset if (caches is not None and hasattr(caches[3], 'offset')) else 0
            positions = mx.arange(offset, offset + T)

        x = inputs_embeds
        hidden_states = [x]
        for i, layer in enumerate(self.layers):
            c = caches[i] if caches is not None else None
            x = layer(x, positions, cache=c)
            hidden_states.append(x)
        x = self.norm(x)
        hidden_states.append(x)
        return x, hidden_states

    def make_caches(self):
        caches = []
        for i, layer in enumerate(self.layers):
            if layer.is_mla:
                caches.append(MLACache())
            else:
                caches.append(KDACache())
        return caches


class BailingMoeV3ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = BailingMoeV3Model(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def __call__(self, input_ids=None, inputs_embeds=None, positions=None, caches=None):
        x, hidden_states = self.model(input_ids, inputs_embeds, positions, caches)
        logits = self.lm_head(x)
        return logits, hidden_states

    def make_caches(self):
        return self.model.make_caches()
