"""Numerical alignment utilities: compare MLX outputs against PyTorch reference.

Cross-framework parity is never bit-exact (FFT, softmax, reduction order all
differ), so we check cosine similarity and relative error at key nodes. The goal
is to localize which layer diverged, not to demand identical bits.
"""

import numpy as np


def _to_numpy(x):
    """Best-effort conversion of mx.array / torch.Tensor / np.ndarray to numpy."""
    if isinstance(x, np.ndarray):
        return x
    # torch.Tensor
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    # mx.array
    try:
        import mlx.core as mx

        if isinstance(x, mx.array):
            return np.array(x, copy=False)
    except ImportError:
        pass
    return np.asarray(x)


def cosine_similarity(a, b):
    """Flattened cosine similarity between two arrays."""
    a = _to_numpy(a).astype(np.float64).ravel()
    b = _to_numpy(b).astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def rel_error(a, b):
    """Max relative error, robust to near-zero reference values."""
    a = _to_numpy(a).astype(np.float64)
    b = _to_numpy(b).astype(np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))


def has_nan_inf(x):
    arr = _to_numpy(x)
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


def compare(name, mlx_out, torch_out, threshold=0.99):
    """Compare one node and return a result dict.

    Reports cosine similarity, relative error, NaN/Inf flags, and pass/fail
    against the cosine threshold.
    """
    nan_mlx = has_nan_inf(mlx_out)
    nan_torch = has_nan_inf(torch_out)
    cos = cosine_similarity(mlx_out, torch_out)
    rel = rel_error(mlx_out, torch_out)
    passed = (cos >= threshold) and not nan_mlx and not nan_torch
    return {
        "name": name,
        "cosine": cos,
        "rel_error": rel,
        "nan_mlx": nan_mlx,
        "nan_torch": nan_torch,
        "passed": passed,
        "threshold": threshold,
    }


def report(results):
    """Pretty-print a list of compare() results; return True if all passed."""
    all_passed = True
    print(f"{'node':<40} {'cosine':>10} {'rel_err':>12} {'status':>8}")
    print("-" * 74)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        flag = ""
        if r["nan_mlx"] or r["nan_torch"]:
            flag = " [NaN/Inf]"
        print(f"{r['name']:<40} {r['cosine']:>10.5f} {r['rel_error']:>12.3e} {status:>8}{flag}")
        all_passed = all_passed and r["passed"]
    print("-" * 74)
    print("ALL PASS" if all_passed else "SOME FAILED")
    return all_passed
