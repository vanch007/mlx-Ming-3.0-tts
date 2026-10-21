"""Speaker-embedding bridge: wraps the original ONNX CampPlus extractor.

CampPlus ships as an ONNX model (not PyTorch weights), and speaker embedding is
computed once at the start of generation, not in the AR loop. Re-implementing it
in MLX has poor cost/benefit, so we reuse the original extractor verbatim and
only convert its output to an mx.array at the boundary.
"""

import sys
import os

import mlx.core as mx
import numpy as np

# Reuse the original extractor (ONNX Runtime, CPU) without modification.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spkemb_extractor import SpkembExtractor  # noqa: E402


class SpeakerEmbedder:
    """Thin wrapper returning speaker embeddings as mx.array."""

    def __init__(self, campplus_model, target_sr=16000):
        self._extractor = SpkembExtractor(campplus_model, target_sr=target_sr)

    def __call__(self, waveform):
        """waveform: torch.Tensor (as the original extractor expects).

        Returns an mx.array of shape (1, emb_dim).
        """
        emb = self._extractor(waveform)  # torch.Tensor (1, emb_dim)
        return mx.array(np.asarray(emb.cpu().numpy(), dtype=np.float32))
