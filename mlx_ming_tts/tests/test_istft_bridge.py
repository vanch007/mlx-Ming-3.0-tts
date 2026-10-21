"""Stage 0 acceptance: numpy ISTFT bridge vs torch reference.

Acceptance gate: cosine similarity > 0.99 for non-streaming reconstruction and
for the streaming path (chunked) reassembled against a single-shot torch run.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402

from audio_tokenizer.istft import ISTFT as TorchISTFT  # noqa: E402
from mlx_ming_tts.bridge.istft import ISTFT as NpISTFT  # noqa: E402
from mlx_ming_tts.verify import cosine_similarity  # noqa: E402

N_FFT = 1280
HOP = 320
WIN = 1280


def _random_spec(T, seed=0):
    rng = np.random.default_rng(seed)
    n_bins = N_FFT // 2 + 1
    real = rng.standard_normal((1, n_bins, T)).astype(np.float32)
    imag = rng.standard_normal((1, n_bins, T)).astype(np.float32)
    return real + 1j * imag


def test_window_matches_torch():
    from mlx_ming_tts.bridge.istft import hann_window

    np_win = hann_window(WIN)
    torch_win = torch.hann_window(WIN).numpy()
    assert cosine_similarity(np_win, torch_win) > 0.999999
    assert np.max(np.abs(np_win - torch_win)) < 1e-5


def test_non_streaming_reconstruction():
    spec = _random_spec(T=20, seed=1)

    torch_istft = TorchISTFT(N_FFT, HOP, WIN, padding="same")
    y_torch, _, _ = torch_istft(torch.from_numpy(spec), streaming=False)

    np_istft = NpISTFT(N_FFT, HOP, WIN, padding="same")
    y_np, _, _ = np_istft(spec, streaming=False)

    cos = cosine_similarity(y_np, y_torch)
    print(f"non-streaming cosine: {cos:.6f}")
    assert cos > 0.99, f"cosine {cos} below 0.99"


def test_streaming_reconstruction():
    # Three chunks, processed with carry-over buffers, then concatenated.
    specs = [_random_spec(T=8, seed=s) for s in (2, 3, 4)]

    torch_istft = TorchISTFT(N_FFT, HOP, WIN, padding="same")
    np_istft = NpISTFT(N_FFT, HOP, WIN, padding="same")

    ta, tw = None, None
    na, nw = None, None
    torch_outs, np_outs = [], []
    for i, spec in enumerate(specs):
        last = i == len(specs) - 1
        yt, ta, tw = torch_istft(torch.from_numpy(spec), audio_buffer=ta,
                                 window_buffer=tw, streaming=True, last_chunk=last)
        yn, na, nw = np_istft(spec, audio_buffer=na, window_buffer=nw,
                              streaming=True, last_chunk=last)
        torch_outs.append(yt.numpy())
        np_outs.append(yn)

    y_torch = np.concatenate(torch_outs, axis=1)
    y_np = np.concatenate(np_outs, axis=1)
    cos = cosine_similarity(y_np, y_torch)
    print(f"streaming cosine: {cos:.6f}")
    assert cos > 0.99, f"streaming cosine {cos} below 0.99"


if __name__ == "__main__":
    test_window_matches_torch()
    print("window: PASS")
    test_non_streaming_reconstruction()
    print("non-streaming: PASS")
    test_streaming_reconstruction()
    print("streaming: PASS")
