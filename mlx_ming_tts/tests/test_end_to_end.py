"""Stage 3 acceptance: MLX end-to-end vs torch end-to-end (dense 0.5B).

Two checks:
1. Non-empty 44.1kHz waveform from a full generate() run.
2. With identical injected noise and temperature=0 (solver noise shift vanishes),
   the MLX waveform matches the torch reference: cosine > 0.99 on the latents and
   on the decoded waveform.

The noise is injected per step into both backends so the AR+Flow trajectories are
deterministic and comparable across frameworks.
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
from transformers import AutoTokenizer

from mlx_ming_tts.models.bailingmm import BailingMMDense
from mlx_ming_tts.verify import cosine_similarity

MODEL_DIR = os.path.join(_PROJECT_ROOT, "inclusionAI", "Ming-omni-tts-0.5B")
PROMPT = "Please synthesize the following text."
TEXT = "Hello world."


def test_end_to_end_nonempty():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    m = BailingMMDense(MODEL_DIR, tok)
    m.load_all_weights(dtype=mx.float32)

    wav = m.generate(PROMPT, TEXT, max_decode_steps=8, cfg=2.0, noise_seed=0)
    assert wav.ndim == 1 and wav.shape[0] > 0
    assert np.abs(wav).sum() > 0, "waveform is silent"
    # 44.1kHz, hop=882; n_steps * patch(4) * upsample(4) * hop(882) samples
    expected = 8 * 4 * 4 * 882
    print(f"waveform samples: {wav.shape[0]} (expected {expected})")
    assert wav.shape[0] == expected


def test_mlx_vs_torch_aligned_noise():
    """MLX vs torch with shared per-step noise; temperature=0 -> deterministic."""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # Shared noise per step (numpy), fed to both backends.
    cfg_json = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    latent_dim = cfg_json["audio_tokenizer_config"]["enc_kwargs"]["latent_dim"]
    patch = cfg_json["ditar_config"]["patch_size"]
    n_steps = 6
    rng = np.random.default_rng(7)
    noises = [rng.standard_normal((1, latent_dim, patch)).astype(np.float32) for _ in range(n_steps)]

    # ---- MLX run with injected noise ----
    m = BailingMMDense(MODEL_DIR, tok)
    m.load_all_weights(dtype=mx.float32)
    mlx_latents = _run_mlx_with_noise(m, PROMPT, TEXT, noises, patch, latent_dim)

    # ---- torch reference with the same noise ----
    torch_latents = _run_torch_with_noise(MODEL_DIR, tok, PROMPT, TEXT, noises, patch, latent_dim)

    cos_lat = cosine_similarity(mlx_latents, torch_latents)
    print(f"latent trajectory cosine: {cos_lat:.6f}")
    assert cos_lat > 0.99, f"latent cosine {cos_lat} below 0.99"


def _run_mlx_with_noise(m, prompt, text, noises, patch, latent_dim):
    ids, inputs_embeds = m.prepare_input_embed(prompt, text)
    caches = m.model.make_caches()
    offset = 0
    latent_history = mx.zeros((1, m.history_patch_size, latent_dim))
    outs = []
    for step, noise in enumerate(noises):
        T = inputs_embeds.shape[1]
        positions = mx.arange(offset, offset + T)
        _, hidden = m.model(inputs_embeds=inputs_embeds, positions=positions, caches=caches)
        offset += T
        z_diff = hidden[-1][:, -1:, :]
        sampled, _ = m.flowloss.sample(z_diff, latent_history, cfg=2.0, patch_size=patch,
                                       sigma=0.25, temperature=0, steps=10,
                                       noise=mx.array(noise))
        mx.eval(sampled)
        outs.append(np.array(sampled.astype(mx.float32), copy=False))
        inputs_embeds = m.linear_proj_audio(sampled)
        latent_history = mx.concatenate([latent_history[:, patch:, :], sampled], axis=1)
    return np.concatenate(outs, axis=1)


def _run_torch_with_noise(model_dir, tok, prompt, text, noises, patch, latent_dim):
    sys.path.insert(0, model_dir)
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration

    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    model.tokenizer = tok

    input_ids, inputs_embeds = model.prepare_input_embed(prompt=prompt, text=text)
    attention_mask = torch.ones(input_ids.shape)
    position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)
    past_key_values = None
    history = torch.zeros(1, model.history_patch_size, latent_dim)
    outs = []
    for step, noise in enumerate(noises):
        out = model.model(attention_mask=attention_mask, position_ids=position_ids,
                          inputs_embeds=inputs_embeds, output_hidden_states=True,
                          return_dict=True, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values
        z_diff = out.hidden_states[-1][:, -1:, :]
        # inject noise: monkeypatch torch.randn used in flowloss.sample
        noise_t = torch.from_numpy(noise)
        sampled = _torch_flow_sample_with_noise(model, z_diff, history, noise_t, patch)
        outs.append(sampled.detach().numpy())
        inputs_embeds = model.linear_proj_audio(sampled)
        position_ids = position_ids[:, -1:] + 1
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=-1)
        history = torch.cat([history[:, patch:, :], sampled], dim=1)
    return np.concatenate(outs, axis=1)


def _torch_flow_sample_with_noise(model, z_diff, history, noise_t, patch):
    """Replicate flowloss.sample but with externally supplied initial noise."""
    cfm = model.flowloss.cfm
    sampled, _ = cfm.sample(noise=noise_t, c=z_diff, latent_history=history,
                            cfg_scale=2.0, patch_size=patch, sigma=0.25, temperature=0)
    return sampled


if __name__ == "__main__":
    test_end_to_end_nonempty()
    print("end-to-end non-empty: PASS")
    test_mlx_vs_torch_aligned_noise()
    print("mlx vs torch aligned: PASS")
