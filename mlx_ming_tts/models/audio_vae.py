"""AudioVAE in pure MLX: encoder/decoder around the Qwen2 trunk + ISTFT head.

Mirrors audio_tokenizer/{modeling_audio_vae,vae_modules}.py for the config in
this project (semantic_module=None, patch_size=4). Both the linear upsample
and ISTFT reconstruction (mx.fft.irfft + OLA) run completely in MLX.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_ming_tts.models.qwen2 import Qwen2Model


def linear_upsample(x, scale_factor, align_corners=False):
    """Pure MLX match for torch nn.Upsample(mode='linear', align_corners=False).

    x: (B, C, L) -> (B, C, L*scale_factor).
    """
    B, C, L = x.shape
    out_len = L * scale_factor
    out_idx = mx.arange(out_len, dtype=mx.float32)
    src = (out_idx + 0.5) / scale_factor - 0.5
    src = mx.clip(src, 0.0, float(L - 1))
    left = mx.floor(src).astype(mx.int32)
    right = mx.minimum(left + 1, L - 1)
    frac = src - left.astype(mx.float32)
    out_left = x[:, :, left]
    out_right = x[:, :, right]
    return out_left * (1.0 - frac) + out_right * frac


def istft_reconstruct(mag, phase, n_fft, hop_length, win_length, padding="same"):
    """Pure MLX reconstruction from complex spectrogram to waveform.

    mag, phase: (B, N, T)
    """
    B, N, T = mag.shape
    pad = (win_length - hop_length) // 2

    n = mx.arange(win_length, dtype=mx.float32)
    window = 0.5 - 0.5 * mx.cos(2.0 * 3.141592653589793 * n / win_length)

    real = mag * mx.cos(phase)
    imag = mag * mx.sin(phase)
    spec = real + 1j * imag

    ifft = mx.fft.irfft(spec, n=n_fft, axis=1) * window[None, :, None]
    output_size = (T - 1) * hop_length + win_length
    y = mx.zeros((B, output_size), dtype=mx.float32)
    for t in range(T):
        st = t * hop_length
        y[:, st : st + win_length] = y[:, st : st + win_length] + ifft[:, :, t]
    y = y[:, pad:-pad]

    window_sq = mx.broadcast_to((window ** 2)[:, None], (win_length, T))
    env = mx.zeros((1, output_size), dtype=mx.float32)
    for t in range(T):
        st = t * hop_length
        env[:, st : st + win_length] = env[:, st : st + win_length] + window_sq[:, t]
    env = env[:, pad:-pad]

    y = y / mx.maximum(env, 1e-11)
    return y.squeeze()


class Encoder(nn.Module):
    def __init__(self, backbone_cfg, input_dim=882, hop_size=320, latent_dim=64,
                 patch_size=4):
        super().__init__()
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.fc1 = nn.Linear(input_dim, backbone_cfg["hidden_size"], bias=False)
        self.fc2 = nn.Linear(backbone_cfg["hidden_size"], backbone_cfg["hidden_size"])
        self.fc3 = nn.Linear(backbone_cfg["hidden_size"], latent_dim * 2)
        self.norm = nn.LayerNorm(backbone_cfg["hidden_size"])
        self.encoder = Qwen2Model(backbone_cfg)
        if patch_size != -1:
            agg_cfg = dict(backbone_cfg)
            agg_cfg["num_hidden_layers"] = 4
            self.aggregator = Qwen2Model(agg_cfg)
            self.cls_embed = mx.zeros((1, 1, backbone_cfg["hidden_size"]))

    def get_frames(self, x):
        # x: (B, T_wav) -> (B, num_frames, input_dim) via unfold(size, step)
        B, Lw = x.shape
        num_frames = (Lw + self.hop_size - 1) // self.hop_size
        expected = (num_frames - 1) * self.hop_size + self.input_dim
        pad = expected - Lw
        if pad > 0:
            x = mx.concatenate([x, mx.zeros((B, pad))], axis=1)
        frames = mx.stack(
            [x[:, i * self.hop_size : i * self.hop_size + self.input_dim] for i in range(num_frames)],
            axis=1,
        )
        return frames

    def pad_patch_insert_cls(self, x):
        bsz, num_frame, dim = x.shape
        r = num_frame % self.patch_size
        pad_num = (self.patch_size - r) if r else 0
        if pad_num:
            x = mx.concatenate([x, mx.zeros((bsz, pad_num, dim))], axis=1)
        x = x.reshape(-1, self.patch_size, dim)
        cls = mx.broadcast_to(self.cls_embed, (x.shape[0], 1, dim))
        x = mx.concatenate([x, cls], axis=1)
        x = x.reshape(bsz, -1, dim)
        return x

    def __call__(self, waveform):
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        x, _ = self.encoder(inputs_embeds=x)
        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            x, _ = self.aggregator(inputs_embeds=x)
            bsz = waveform.shape[0]
            dim = x.shape[-1]
            x = x.reshape(-1, self.patch_size + 1, dim)
            x = x[:, -1:, :].reshape(bsz, -1, dim)
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, backbone_cfg, output_dim=882, latent_dim=64, patch_size=4):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hop_length = output_dim
        self.n_fft = output_dim * 4
        self.fc1 = nn.Linear(latent_dim, backbone_cfg["hidden_size"])
        self.decoder = Qwen2Model(backbone_cfg)
        self.head_out = nn.Linear(backbone_cfg["hidden_size"], self.n_fft + 2)

    def project_spectrogram(self, latent, upsample=True):
        x = self.fc1(latent)
        if self.patch_size != -1 and upsample:
            x = linear_upsample(x.transpose(0, 2, 1), self.patch_size).transpose(0, 2, 1)
        x, _ = self.decoder(inputs_embeds=x)
        x_pred = self.head_out(x)
        x_pred = x_pred.transpose(0, 2, 1)
        n = (self.n_fft + 2) // 2
        mag = mx.exp(x_pred[:, :n, :])
        mag = mx.clip(mag, a_min=None, a_max=1e2)
        phase = x_pred[:, n:, :]
        return mag, phase


class AudioVAE(nn.Module):
    """Encoder + Decoder with pure MLX audio synthesis."""

    def __init__(self, atc):
        super().__init__()
        enc_bb = atc["enc_kwargs"]["backbone"]
        dec_bb = atc["dec_kwargs"]["backbone"]
        patch = atc["patch_size"]
        self.encoder = Encoder(
            enc_bb, input_dim=atc["enc_kwargs"]["input_dim"],
            hop_size=atc["enc_kwargs"].get("hop_size", 320),
            latent_dim=atc["enc_kwargs"]["latent_dim"], patch_size=patch,
        )
        self.decoder = Decoder(
            dec_bb, output_dim=atc["dec_kwargs"]["output_dim"],
            latent_dim=atc["dec_kwargs"]["latent_dim"], patch_size=patch,
        )
        self.input_dim = atc["enc_kwargs"]["input_dim"]
        self.latent_dim = atc["enc_kwargs"]["latent_dim"]
        self.patch_size = patch

    def encode_latent(self, waveform, noise=None):
        frame_num = mx.ceil(mx.array([waveform.shape[1]], dtype=mx.float32) / self.input_dim).astype(mx.int32)
        if self.patch_size != -1:
            frame_num = mx.ceil(frame_num.astype(mx.float32) / self.patch_size).astype(mx.int32)
        h = self.encoder(waveform).transpose(0, 2, 1)
        mean, scale = mx.split(h, 2, axis=1)
        std = nn.softplus(scale) + 1e-4
        if noise is None:
            noise = mx.random.normal(mean.shape, dtype=mean.dtype)
        latent = mean + std * noise
        return latent.transpose(0, 2, 1), frame_num

    def decode_full(self, latent):
        """Non-streaming decode of a full latent sequence -> waveform (MLX array 1-D)."""
        mag, phase = self.decoder.project_spectrogram(latent)
        return istft_reconstruct(
            mag,
            phase,
            n_fft=self.decoder.n_fft,
            hop_length=self.decoder.hop_length,
            win_length=self.decoder.n_fft,
            padding="same",
        )
