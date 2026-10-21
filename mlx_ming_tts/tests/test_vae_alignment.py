"""Stage 2b acceptance: MLX AudioVAE vs torch AudioVAE.

Encoder output is a Gaussian distribution whose sampling is stochastic, so we
compare the deterministic distribution parameters (fc3 output, pre-sampling) and
the decoder spectrogram (mag/phase, deterministic given a latent). Acceptance:
cosine > 0.99.
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
from mlx.utils import tree_flatten

from audio_tokenizer.modeling_audio_vae import AudioVAE as TorchAudioVAE
from audio_tokenizer.configuration_audio_vae import AudioVAEconfig
from mlx_ming_tts.models.audio_vae import AudioVAE as MLXAudioVAE
from mlx_ming_tts.convert import convert_vae
from mlx_ming_tts.verify import compare, report

MODEL_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")


def _build():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    atc = cfg["audio_tokenizer_config"]

    mlx_vae = MLXAudioVAE(atc)
    names = [k for k, _ in tree_flatten(mlx_vae.parameters())]
    weights = convert_vae(MODEL_DIR, names, dtype=mx.float32)
    mlx_vae.load_weights(list(weights.items()))
    mlx_vae.eval()

    torch_cfg = AudioVAEconfig(**atc)
    torch_vae = TorchAudioVAE(torch_cfg)
    src = mx.load(os.path.join(MODEL_DIR, "model.safetensors"))
    sd = {}
    for k, v in src.items():
        if k.startswith("audio."):
            sd[k[len("audio."):]] = torch.from_numpy(np.array(v.astype(mx.float32), copy=False))
    torch_vae.load_state_dict(sd, strict=False)
    torch_vae = torch_vae.float().eval()
    return mlx_vae, torch_vae, atc


def test_encoder_distribution_alignment():
    mlx_vae, torch_vae, atc = _build()
    rng = np.random.default_rng(0)
    # 8 frames worth of waveform (< sliding_window 64)
    input_dim = atc["enc_kwargs"]["input_dim"]
    hop = atc["enc_kwargs"].get("hop_size", 320)
    n_frames = 8
    wav_len = (n_frames - 1) * hop + input_dim
    wav = rng.standard_normal((1, wav_len)).astype(np.float32)

    with torch.no_grad():
        t_h, _ = torch_vae.encoder(torch.from_numpy(wav))
    m_h = mlx_vae.encoder(mx.array(wav))
    mx.eval(m_h)

    results = [compare("encoder_dist_params", m_h, t_h)]
    assert report(results), "encoder distribution alignment failed"


def test_decoder_spectrogram_alignment():
    mlx_vae, torch_vae, atc = _build()
    rng = np.random.default_rng(1)
    latent_dim = atc["dec_kwargs"]["latent_dim"]
    n_frames = 6
    latent = rng.standard_normal((1, n_frames, latent_dim)).astype(np.float32)

    # torch: run decoder up to ISTFTHead, capture mag/phase
    with torch.no_grad():
        x = torch_vae.decoder.fc1(torch.from_numpy(latent))
        if torch_vae.decoder.patch_size != -1:
            x = torch_vae.decoder.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)
        x = torch_vae.decoder.decoder(inputs_embeds=x).last_hidden_state
        x_pred = torch_vae.decoder.head.out(x).transpose(1, 2)
        t_mag, t_phase = x_pred.chunk(2, dim=1)
        t_mag = torch.clip(torch.exp(t_mag), max=1e2)

    m_mag, m_phase = mlx_vae.decoder.project_spectrogram(mx.array(latent))
    mx.eval(m_mag, m_phase)

    results = [
        compare("decoder_mag", m_mag, t_mag),
        compare("decoder_phase", m_phase, t_phase),
    ]
    assert report(results), "decoder spectrogram alignment failed"


def test_full_decode_with_bridge():
    """MLX decoder spectrogram -> numpy ISTFT bridge vs torch end-to-end decode."""
    from mlx_ming_tts.bridge.istft import ISTFT

    mlx_vae, torch_vae, atc = _build()
    rng = np.random.default_rng(2)
    latent_dim = atc["dec_kwargs"]["latent_dim"]
    latent = rng.standard_normal((1, 6, latent_dim)).astype(np.float32)

    with torch.no_grad():
        wav_t, _, _ = torch_vae.decode(torch.from_numpy(latent), use_cache=False)
    wav_t = wav_t.squeeze().numpy()

    mag, phase = mlx_vae.decoder.project_spectrogram(mx.array(latent))
    mag = np.array(mag.astype(mx.float32), copy=False)
    phase = np.array(phase.astype(mx.float32), copy=False)
    spec = mag * (np.cos(phase) + 1j * np.sin(phase))
    n_fft = mlx_vae.decoder.n_fft
    hop = mlx_vae.decoder.hop_length
    istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft, padding="same")
    wav_m, _, _ = istft(spec, streaming=False)
    wav_m = wav_m.squeeze()

    L = min(len(wav_t), len(wav_m))
    from mlx_ming_tts.verify import cosine_similarity
    cos = cosine_similarity(wav_m[:L], wav_t[:L])
    print(f"end-to-end decode cosine: {cos:.6f}")
    assert cos > 0.99, f"end-to-end decode cosine {cos} below 0.99"


if __name__ == "__main__":
    test_encoder_distribution_alignment()
    print("encoder: PASS")
    test_decoder_spectrogram_alignment()
    print("decoder: PASS")
    test_full_decode_with_bridge()
    print("full decode + bridge: PASS")
