"""Stage 0 acceptance: speaker bridge output matches the original extractor.

The bridge only converts the ONNX extractor's output to mx.array, so values must
be element-wise identical to the original SpkembExtractor.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402

from spkemb_extractor import SpkembExtractor  # noqa: E402
from mlx_ming_tts.bridge.speaker import SpeakerEmbedder  # noqa: E402

CAMPPLUS = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B", "campplus.onnx")


def test_speaker_bridge_matches_original():
    assert os.path.exists(CAMPPLUS), f"campplus.onnx not found at {CAMPPLUS}"

    # 1 second of fake 16kHz audio, shape (1, T) as kaldi.fbank expects.
    rng = np.random.default_rng(0)
    waveform = torch.from_numpy(rng.standard_normal((1, 16000)).astype(np.float32))

    original = SpkembExtractor(CAMPPLUS)
    bridge = SpeakerEmbedder(CAMPPLUS)

    emb_orig = original(waveform).cpu().numpy()
    emb_bridge = np.array(bridge(waveform), copy=False)

    print(f"orig shape {emb_orig.shape}, bridge shape {emb_bridge.shape}")
    assert emb_orig.shape == emb_bridge.shape
    max_diff = float(np.max(np.abs(emb_orig - emb_bridge)))
    print(f"max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-5, f"max diff {max_diff} too large"


if __name__ == "__main__":
    test_speaker_bridge_matches_original()
    print("speaker bridge: PASS")
