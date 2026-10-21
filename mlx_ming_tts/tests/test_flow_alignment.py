"""Stage 2a acceptance: MLX DiT vs torch DiT (deterministic vector field).

Flow sampling is stochastic (randn in the Solver), so cross-framework RNG can't
match. We instead verify the deterministic DiT forward -- the ODE vector field
that the solver integrates -- with identical inputs. Acceptance: cosine > 0.99.

We also check forward_with_cfg (the cond/uncond batched path).
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import mlx.core as mx
import torch

from fm.flowloss import FlowLoss as TorchFlowLoss
from mlx_ming_tts.models.flow import FlowLoss as MLXFlowLoss
from mlx_ming_tts.convert import convert_by_prefix
from mlx_ming_tts.verify import compare, report
from mlx.utils import tree_flatten

MODEL_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")


def _build_models():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    dc = cfg["ditar_config"]
    z_channels = 64
    llm_cond = cfg["llm_config"]["hidden_size"]

    mlx_fl = MLXFlowLoss(z_channels=z_channels, llm_cond_dim=llm_cond,
                         hidden_size=dc["hidden_size"], depth=dc["depth"],
                         num_heads=dc["num_heads"], mlp_ratio=dc["mlp_ratio"])
    names = [k for k, _ in tree_flatten(mlx_fl.parameters())]
    weights = convert_by_prefix(MODEL_DIR, "flowloss", names, dtype=mx.float32)
    mlx_fl.load_weights(list(weights.items()))
    mlx_fl.eval()

    torch_fl = TorchFlowLoss(z_channels=z_channels, llm_cond_dim=llm_cond,
                             hidden_size=dc["hidden_size"], depth=dc["depth"],
                             num_heads=dc["num_heads"], mlp_ratio=dc["mlp_ratio"])
    src = mx.load(os.path.join(MODEL_DIR, "model.safetensors"))
    sd = {}
    for k, v in src.items():
        if k.startswith("flowloss."):
            sd[k[len("flowloss."):]] = torch.from_numpy(np.array(v.astype(mx.float32), copy=False))
    torch_fl.load_state_dict(sd, strict=False)
    torch_fl = torch_fl.float().eval()
    return mlx_fl, torch_fl, dc, z_channels, llm_cond


def test_dit_forward_alignment():
    mlx_fl, torch_fl, dc, z_channels, llm_cond = _build_models()

    rng = np.random.default_rng(0)
    patch = dc["patch_size"]
    hist = dc["history_patch_size"]
    x = rng.standard_normal((1, patch, z_channels)).astype(np.float32)
    lat_hist = rng.standard_normal((1, hist, z_channels)).astype(np.float32)
    c = rng.standard_normal((1, 1, llm_cond)).astype(np.float32)
    t = np.array([0.3], dtype=np.float32)

    with torch.no_grad():
        t_out = torch_fl.cfm.model(
            x=torch.from_numpy(x), t=torch.from_numpy(t), c=torch.from_numpy(c),
            latent_history=torch.from_numpy(lat_hist), mask=None,
        )
    m_out = mlx_fl.cfm.model(mx.array(x), mx.array(t), mx.array(c), mx.array(lat_hist))
    mx.eval(m_out)

    results = [compare("dit_forward", m_out, t_out)]
    assert report(results), "DiT forward alignment failed"


def test_dit_cfg_alignment():
    mlx_fl, torch_fl, dc, z_channels, llm_cond = _build_models()

    rng = np.random.default_rng(1)
    patch = dc["patch_size"]
    hist = dc["history_patch_size"]
    x = rng.standard_normal((1, z_channels, patch)).astype(np.float32)  # pre-transpose
    x = np.transpose(x, (0, 2, 1))  # (1, patch, z) as solver feeds after transpose
    lat_hist = rng.standard_normal((1, hist, z_channels)).astype(np.float32)
    c = rng.standard_normal((1, 1, llm_cond)).astype(np.float32)
    t = np.array(0.5, dtype=np.float32)

    with torch.no_grad():
        t_out = torch_fl.cfm.model.forward_with_cfg(
            x=torch.from_numpy(x), t=torch.tensor(0.5), c=torch.from_numpy(c),
            cfg_scale=2.0, latent_history=torch.from_numpy(lat_hist), patch_size=patch,
        )
    m_out = mlx_fl.cfm.model.forward_with_cfg(
        mx.array(x), mx.array(0.5), mx.array(c), cfg_scale=2.0,
        latent_history=mx.array(lat_hist), patch_size=patch,
    )
    mx.eval(m_out)

    results = [compare("dit_forward_with_cfg", m_out, t_out)]
    assert report(results), "DiT CFG alignment failed"


if __name__ == "__main__":
    test_dit_forward_alignment()
    print("dit forward: PASS")
    test_dit_cfg_alignment()
    print("dit cfg: PASS")
