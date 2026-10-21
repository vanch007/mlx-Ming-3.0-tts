"""Flow Matching in MLX: DiT backbone + CFM ODE solver.

Mirrors fm/dit.py, fm/modules.py, fm/CFM.py. Module/param names match the torch
checkpoint (blocks.N.attn.to_q, mlp.ff.0.0, t_embedder.time_mlp.0/2, ...).

The DiT uses x_transformers interleaved RoPE (see ops/rope.apply_xt_rope), which
differs from the HF Qwen2 half-split RoPE used by the LLM.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_ming_tts.ops.norm import RMSNorm
from mlx_ming_tts.ops.rope import xt_rope_freqs, apply_xt_rope
from mlx_ming_tts.ops.attention import sdpa


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        # nn.Sequential(Linear, SiLU, Linear) -> indices 0 and 2 carry weights
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]

    def __call__(self, timestep):
        h = self.time_embed(timestep)
        for layer in self.time_mlp:
            h = layer(h)
        return h


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def __call__(self, llm_cond):
        # inference: no dropout
        return self.cond_embedder(llm_cond)


class FeedForward(nn.Module):
    """nn.Sequential(Sequential(Linear, GELU), Dropout, Linear).

    Param layout: ff.0.0 (first linear), ff.2 (second linear). GELU uses
    approximate='tanh' in the DiT.
    """

    def __init__(self, dim, mult=4, dim_out=None):
        super().__init__()
        inner = int(dim * mult)
        dim_out = dim_out or dim
        self.ff = [[nn.Linear(dim, inner), nn.GELU(approx="tanh")], None, nn.Linear(inner, dim_out)]

    def __call__(self, x):
        x = self.ff[0][0](x)
        x = self.ff[0][1](x)
        x = self.ff[2](x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        inner = dim_head * heads
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.to_out = [nn.Linear(inner, dim)]

    def __call__(self, x, freqs):
        B, T, _ = x.shape
        hd = x.shape[-1] // self.heads
        q = self.to_q(x).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
        k = self.to_k(x).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
        q, k = apply_xt_rope(q, k, freqs)
        out = sdpa(q, k, v)  # non-causal, no mask (attn_mask_enabled=false)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.heads * hd)
        return self.to_out[0](out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, hidden_size // num_heads)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio)

    def __call__(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x):
        return self.linear(self.norm_final(x))


class DiT(nn.Module):
    def __init__(self, in_channels=64, hidden_size=1024, depth=8, num_heads=16,
                 mlp_ratio=4.0, llm_cond_dim=896):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size)
        self.blocks = [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def __call__(self, x, t, c, latent_history):
        t_emb = self.t_embedder(t)[:, None, :]
        x_now = self.x_embedder(x)
        x_hist = self.x_embedder(latent_history)
        x = mx.concatenate([x_hist, x_now], axis=1)
        c_emb = self.c_embedder(c)
        y = t_emb + c_emb
        x = mx.concatenate([y, x], axis=1)
        freqs = xt_rope_freqs(self.head_dim, x.shape[1])
        for block in self.blocks:
            x = block(x, freqs)
        return self.final_layer(x)

    def forward_with_cfg(self, x, t, c, cfg_scale, latent_history, patch_size):
        if cfg_scale != 1:
            x = mx.concatenate([x, x], axis=0)
            latent_history = mx.concatenate([latent_history, latent_history], axis=0)
            fake = mx.zeros_like(c)
            c = mx.concatenate([c, fake], axis=0)
        if t.ndim == 0:
            t = mx.broadcast_to(t, (x.shape[0],))
        out = self(x, t, c, latent_history)
        return out[:, -patch_size:, :]


def get_epss_timesteps(n):
    dt = 1 / 32
    predefined = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined.get(n, [])
    if not t:
        return mx.linspace(0, 1, n + 1)
    return dt * mx.array(t, dtype=mx.float32)


class Solver:
    """First-order Euler ODE solver with stochastic noise shift.

    Mirrors fm/CFM.py:Solver. Noise can be injected for deterministic testing.
    """

    def __init__(self, func, y0, sigma=0.25, temperature=1.5):
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t, noise_fn=None):
        n = len(t)
        solution = [None] * n
        solution[0] = self.y0
        j = 1
        y0 = self.y0
        t_list = [float(x) for x in t]
        for idx in range(n - 1):
            t0, t1 = t_list[idx], t_list[idx + 1]
            dt = t1 - t0
            f0 = self.func(mx.array(t0), y0)
            y1 = y0 + dt * f0
            while j < n and t1 >= t_list[j]:
                solution[j] = self._interp(t0, t1, y0, y1, t_list[j])
                j += 1
            if noise_fn is not None:
                noise = noise_fn(y0.shape, idx)
            else:
                noise = mx.random.normal(y0.shape)
            shift = self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift
        return solution

    def _interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


class CFM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def sample(self, noise, c, latent_history, steps=10, cfg_scale=1.0,
               sway_sampling_coef=-1.0, use_epss=True, patch_size=1,
               sigma=0.25, temperature=1.5, noise_fn=None):
        def fn(t, x):
            pred_cfg = self.model.forward_with_cfg(
                x=x, t=t, c=c, latent_history=latent_history,
                cfg_scale=cfg_scale, patch_size=patch_size,
            )
            if cfg_scale < 1e-5:
                return pred_cfg
            pred, null_pred = mx.split(pred_cfg, 2, axis=0)
            return null_pred + cfg_scale * (pred - null_pred)

        y0 = noise.transpose(0, 2, 1)
        if use_epss:
            t = get_epss_timesteps(steps)
        else:
            t = mx.linspace(0, 1, steps + 1)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (mx.cos(math.pi / 2 * t) - 1 + t)
        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t, noise_fn=noise_fn)
        return trajectory[-1], trajectory


class Aggregator(nn.Module):
    """Projects audio latents back into the LLM input space (linear_proj_audio).

    Mirrors fm/dit.py:Aggregator. Prepends a learned cls token, runs DiTBlocks
    with interleaved RoPE, and returns the projected cls position.
    """

    def __init__(self, in_channels=64, hidden_size=1024, depth=8, num_heads=16,
                 mlp_ratio=4.0, llm_input_dim=896):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.blocks = [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def __call__(self, x):
        x = self.x_embedder(x)
        cls = self.word_embedder(mx.zeros((x.shape[0], 1), dtype=mx.int32))
        x = mx.concatenate([cls, x], axis=1)
        freqs = xt_rope_freqs(self.head_dim, x.shape[1])
        for block in self.blocks:
            x = block(x, freqs)
        x = self.final_layer(x)
        return x[:, :1, :]


class FlowLoss(nn.Module):
    """Wraps CFM(DiT). sample() mirrors fm/flowloss.py."""

    def __init__(self, z_channels, llm_cond_dim, hidden_size=1024, depth=8,
                 num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(DiT(in_channels=z_channels, hidden_size=hidden_size, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio, llm_cond_dim=llm_cond_dim))

    def sample(self, z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
               temperature=0, steps=10, noise=None, noise_fn=None):
        if noise is None:
            noise = mx.random.normal((z.shape[0], self.z_channels, patch_size))
        return self.cfm.sample(noise=noise, c=z, latent_history=latent_history,
                               cfg_scale=cfg, patch_size=patch_size, sigma=sigma,
                               temperature=temperature, steps=steps, noise_fn=noise_fn)
