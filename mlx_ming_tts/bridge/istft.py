"""ISTFT bridge: numpy port of audio_tokenizer/istft.py.

MLX has no irfft / fold, so the final spectrogram -> waveform step runs in numpy.
This is not a performance bottleneck (it runs once per decoded chunk on a small
spectrogram) and keeps the numerically delicate FFT path on a mature backend.

The ISTFTHead linear projection (dim -> n_fft+2) stays in MLX; only the complex
spectrogram -> waveform reconstruction is bridged here. Inputs/outputs at the
bridge boundary are numpy arrays.
"""

import numpy as np


def hann_window(win_length):
    """Periodic Hann window, matching torch.hann_window(periodic=True).

    torch.hann_window defaults to periodic=True, which equals
    scipy.signal.windows.hann(N, sym=False). np.hanning is symmetric and would
    NOT match, so the window is constructed explicitly here.
    """
    n = np.arange(win_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / win_length)).astype(np.float32)


class ISTFT:
    """Numpy ISTFT with "same" padding and streaming overlap-add buffers.

    Mirrors audio_tokenizer/istft.py:ISTFT. Custom because torch.istft (and a
    plain irfft) can't do "same" padding for neural vocoding; the NOLA check
    fails at the edges, which we sidestep by trimming padded samples.
    """

    def __init__(self, n_fft, hop_length, win_length, padding="same"):
        if padding not in ("center", "same"):
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = hann_window(win_length)
        self.buffer_len = win_length - hop_length

    def _buffer_process(self, x, buffer, pad, last_chunk=False, streaming=False):
        # x: (B, L). Mirrors ISTFT.__buffer_process in the torch version.
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            else:
                x = x.copy()
                x[:, : self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len :]
            if not last_chunk:
                x = x[:, : -self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def _overlap_add(self, frames, output_size):
        """Overlap-add equivalent of F.fold for a single 1-D signal per batch.

        frames: (B, win_length, T) -> (B, output_size)
        """
        B, W, T = frames.shape
        out = np.zeros((B, output_size), dtype=np.float64)
        for t in range(T):
            start = t * self.hop_length
            out[:, start : start + W] += frames[:, :, t]
        return out

    def __call__(
        self,
        spec,
        audio_buffer=None,
        window_buffer=None,
        streaming=False,
        last_chunk=False,
    ):
        """Reconstruct waveform from a complex spectrogram.

        spec: complex numpy array (B, N, T), N = n_fft // 2 + 1.
        Returns (waveform (B, L), audio_buffer, window_buffer).
        """
        if self.padding == "center":
            raise NotImplementedError("center padding not used in this pipeline")
        pad = (self.win_length - self.hop_length) // 2

        assert spec.ndim == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT along the frequency axis. norm="backward" matches torch.
        ifft = np.fft.irfft(spec, n=self.n_fft, axis=1, norm="backward").astype(np.float64)
        ifft = ifft * self.window[None, :, None]

        output_size = (T - 1) * self.hop_length + self.win_length
        y = self._overlap_add(ifft, output_size)

        y, audio_buffer = self._buffer_process(
            y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming
        )

        # Window envelope (COLA normalization).
        window_sq = np.broadcast_to(
            (self.window**2)[:, None], (self.win_length, T)
        ).astype(np.float64)
        window_envelope = self._overlap_add(window_sq[None, :, :], output_size)
        window_envelope, window_buffer = self._buffer_process(
            window_envelope, window_buffer, pad, last_chunk=last_chunk, streaming=streaming
        )

        assert (window_envelope > 1e-11).all()
        y = y / window_envelope
        return y.astype(np.float32), audio_buffer, window_buffer


def reconstruct_from_mag_phase(mag, phase, n_fft, hop_length, win_length,
                               audio_buffer=None, window_buffer=None,
                               streaming=False, last_chunk=False, istft=None):
    """Build complex spectrogram from magnitude/phase and run ISTFT.

    mag, phase: numpy arrays (B, N, T). Mirrors the tail of ISTFTHead.forward in
    the torch version (after the linear projection and chunk).
    """
    if istft is None:
        istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding="same")
    spec = mag * (np.cos(phase) + 1j * np.sin(phase))
    audio, audio_buffer, window_buffer = istft(
        spec, audio_buffer=audio_buffer, window_buffer=window_buffer,
        streaming=streaming, last_chunk=last_chunk,
    )
    return audio, audio_buffer, window_buffer
