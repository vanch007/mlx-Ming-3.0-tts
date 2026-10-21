"""Normalization ops."""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm with float32 variance accumulation.

    Variance is computed in float32 then cast back, matching the torch
    reference and avoiding the float16 overflow -> NaN seen for Qwen2 on MPS.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x.astype(input_dtype)
