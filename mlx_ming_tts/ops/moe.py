"""Mixture-of-Experts ops for the Bailing MoE LLM.

Routing mirrors BailingMoeGate: softmax(fp32) over expert logits, top-k, then
normalize the top-k weights by their sum (norm_topk_prob). Expert dispatch uses
stacked expert weights and batched gather/einsum so only the routed top-k
experts run. This is the critical path for the 16.8B model.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ExpertMLP(nn.Module):
    """SwiGLU expert: down(silu(gate(x)) * up(x)). No bias."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Top-k routed experts + shared experts, summed residually.

    Mirrors BailingMoeSparseMoeBlock inference (moe_infer + shared_experts). The
    image/audio gates exist in the checkpoint (multi_gate) but are unused for
    text/audio generation without modality masks, so only the text gate runs.
    """

    def __init__(self, hidden, moe_intermediate, num_experts, top_k,
                 num_shared_experts, norm_topk_prob=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts_gate_proj_weight = mx.zeros((num_experts, moe_intermediate, hidden))
        self.experts_up_proj_weight = mx.zeros((num_experts, moe_intermediate, hidden))
        self.experts_down_proj_weight = mx.zeros((num_experts, hidden, moe_intermediate))
        if num_shared_experts and num_shared_experts > 0:
            self.shared_experts = ExpertMLP(hidden, moe_intermediate * num_shared_experts)
        else:
            self.shared_experts = None

    def __call__(self, x):
        # x: (B, T, H) -> flatten to (N, H)
        B, T, H = x.shape
        xf = x.reshape(B * T, H)

        logits = self.gate(xf)  # (N, E)
        scores = mx.softmax(logits.astype(mx.float32), axis=-1)

        # top-k indices/weights (order within top-k is irrelevant for the sum).
        # mx.topk currently returns values only, so argpartition gives indices.
        topk_idx = mx.argpartition(scores, kth=self.num_experts - self.top_k, axis=-1)[:, -self.top_k:]
        topk_w = mx.take_along_axis(scores, topk_idx, axis=-1)  # (N, k)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_w = topk_w / mx.sum(topk_w, axis=-1, keepdims=True)

        if B * T == 1:
            # Decode path: one token at a time. Avoid top-k weight gather because
            # it materializes large expert tensors; slicing one expert at a time
            # keeps memory traffic lower despite a small Python loop.
            mx.eval(topk_idx, topk_w)
            idx = np.array(topk_idx, copy=False)[0].tolist()
            out = mx.zeros((1, H), dtype=x.dtype)
            for j, expert_idx in enumerate(idx):
                gate_w = self.experts_gate_proj_weight[expert_idx]
                up_w = self.experts_up_proj_weight[expert_idx]
                down_w = self.experts_down_proj_weight[expert_idx]
                gate = mx.matmul(xf, gate_w.T)
                up = mx.matmul(xf, up_w.T)
                routed = nn.silu(gate) * up
                expert_out = mx.matmul(routed, down_w.T)
                out = out + topk_w[:, j:j + 1].astype(expert_out.dtype) * expert_out
        else:
            # Prefill path: many tokens. Avoid per-token expert-weight gathers,
            # which duplicate huge tensors; compute all experts in stacked form.
            onehot = (topk_idx[:, :, None] == mx.arange(self.num_experts)[None, None, :])
            full_w = mx.sum(topk_w[:, :, None] * onehot.astype(topk_w.dtype), axis=1)
            gate = mx.einsum("nh,eih->nei", xf, self.experts_gate_proj_weight)
            up = mx.einsum("nh,eih->nei", xf, self.experts_up_proj_weight)
            routed = nn.silu(gate) * up
            expert_out = mx.einsum("nei,ehi->neh", routed, self.experts_down_proj_weight)
            out = mx.sum(expert_out * full_w[:, :, None].astype(expert_out.dtype), axis=1)

        if self.shared_experts is not None:
            out = out + self.shared_experts(xf)

        return out.reshape(B, T, H)
