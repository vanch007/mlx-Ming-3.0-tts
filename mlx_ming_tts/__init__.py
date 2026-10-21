"""MLX implementation of the Ming-omni-tts inference pipeline.

The original PyTorch project remains the ground-truth reference. This package
mirrors its module structure with MLX implementations, while keeping ISTFT and
speaker-embedding extraction as numpy/ONNX bridges (see `bridge/`).
"""
