from .prompt_contract import build_tts_prompt_ids
"""LingV3 TTS (Ming-3.0-tts-tiny) MLX end-to-end model.

Wraps BailingMoeV3ForCausalLM + AudioVAE + FlowLoss (depth=16) + Aggregator.
Supports:
- single-speaker tasks (seed, clone, paralinguistic, dialect, emotion, basic, ip, voice_design)
- multi-speaker podcast dialog (tiny unique capability)
"""

import json
import os
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple

from mlx_ming_tts.models.tiny_moe import BailingMoeV3ForCausalLM
from mlx_ming_tts.models.audio_vae import AudioVAE
from mlx_ming_tts.models.flow import FlowLoss, Aggregator


class LingV3TTS(nn.Module):
    def __init__(self, model_dir, tokenizer):
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        self.config = cfg
        self.llm_cfg = cfg["llm_config"]
        self.audio_cfg = cfg["audio_config"]
        self.atc = self.audio_cfg
        self.flow_cfg = cfg["flow_config"]
        self.agg_cfg = cfg["aggregator_config"]
        self.tts_cfg = cfg["tts_config"]

        hidden = self.llm_cfg["hidden_size"]  # 1536
        latent_dim = self.tts_cfg.get("latent_dim", 64)
        self.patch_size = self.tts_cfg.get("patch_size", 1)
        self.history_size = self.tts_cfg.get("history_size", 1)

        # 1. LLM Trunk
        self.model = BailingMoeV3ForCausalLM(self.llm_cfg)

        # 2. Audio VAE
        self.audio = AudioVAE(self.audio_cfg)

        # 3. Aggregator (linear_proj_audio)
        self.linear_proj_audio = Aggregator(
            in_channels=latent_dim,
            hidden_size=self.agg_cfg["hidden_size"],
            depth=self.agg_cfg["depth"],
            num_heads=self.agg_cfg["num_heads"],
            mlp_ratio=self.agg_cfg["mlp_ratio"],
            llm_input_dim=hidden
        )

        # 4. Flow Matching DiT
        self.flowloss = FlowLoss(
            z_channels=latent_dim,
            llm_cond_dim=hidden,
            hidden_size=self.flow_cfg["hidden_size"],
            depth=self.flow_cfg["depth"],
            num_heads=self.flow_cfg["num_heads"],
            mlp_ratio=self.flow_cfg["mlp_ratio"]
        )

        # 5. Heads
        self.stop_head = nn.Linear(hidden, 2, bias=True)
        self.spk_head = nn.Linear(self.tts_cfg.get("speaker_embedding_dim", 192), hidden, bias=True)

    def load_mlx_weights(self, weights_path):
        weights = mx.load(weights_path)
        # Filter out non-parameter constant buffers from PyTorch checkpoint
        ignored = ('istft.window', 'rotary_embed.inv_freq')
        filtered = [(k, v) for k, v in weights.items() if not any(ig in k for ig in ignored)]
        self.load_weights(filtered)
        self.eval()

    def token_embeddings(self, input_ids):
        return self.model.model.word_embeddings(input_ids)

    def prepare_input_embed(self, prompt, text, spk_emb=None, instruction=None,
                            prompt_latent=None, prompt_text=None,
                            target_language="Chinese"):
        """Tiny prompt assembly using official build_tts_prompt_ids."""
        tok = self.tokenizer
        
        prompt_latent_tokens = 0
        p_hidden = None
        if prompt_latent is not None and prompt_text is not None:
            bsz = prompt_latent.shape[0]
            n_tokens = prompt_latent.shape[1] // self.patch_size
            grouped = prompt_latent.reshape(-1, self.patch_size, 64)
            p_hidden = self.linear_proj_audio(grouped)
            p_hidden = p_hidden.reshape(bsz, n_tokens, -1)
            prompt_latent_tokens = n_tokens

        # Check if multispeaker podcast dialogue
        is_podcast = "speaker_1:" in text and "speaker_2:" in text
        if is_podcast:
            speaker_count = 2
            canon_script = canonicalize_podcast_script(text, speaker_count)
            prompt_transcript = prompt_text if (isinstance(prompt_text, str) and "speaker_1:" in prompt_text) else build_podcast_prompt_transcript([
                prompt_text if isinstance(prompt_text, str) else "诶，你今天有空吗？我们能不能聊聊这件事啊？",
                "感谢你的认可。"
            ])
            prompt_ids, speaker_positions = build_multispeaker_tts_prompt_ids(
                tok,
                canon_script,
                self.tts_cfg,
                speaker_count=speaker_count,
                prompt_text=prompt_transcript,
                prompt_latent_tokens=prompt_latent_tokens,
                target_language=target_language,
            )
            input_ids = prompt_ids
            inputs_embeds = self.token_embeddings(mx.array([input_ids]))
            for i, pos in enumerate(speaker_positions):
                inputs_embeds[0, pos] = 0.0
        else:
            prompt_ids, speaker_pos = build_tts_prompt_ids(
                tok,
                text,
                self.tts_cfg,
                prompt_text=prompt_text,
                prompt_latent_tokens=prompt_latent_tokens,
            )
            input_ids = prompt_ids
            inputs_embeds = self.token_embeddings(mx.array([input_ids]))
            if speaker_pos is not None:
                if spk_emb is not None:
                    se_proj = self.spk_head(spk_emb[0])
                    inputs_embeds[0, speaker_pos] = se_proj[0]
                else:
                    # 官方标准规范：Voice Design / IP / 无参考音频时，必须直接将 speaker position 槽位置为 0
                    # 严禁将全0传给带 bias 的 spk_head，否则注入的 bias 向量会严重污染声学自回归导致音色断崖漂移
                    inputs_embeds[0, speaker_pos] = 0.0

        # Inject prompt acoustic latents
        if p_hidden is not None:
            start = len(input_ids) - prompt_latent_tokens
            inputs_embeds[0, start:start+prompt_latent_tokens, :] = p_hidden[0]

        return input_ids, inputs_embeds

    def sample(self, prompt, text, spk_emb=None, instruction=None,
               prompt_waveform=None, prompt_text=None,
               target_language="Chinese",
               max_decode_steps=200, cfg=2.0, sigma=0.15, temperature=0,
               steps=10, noise_seed=42):
        if prompt_waveform is not None and prompt_text is not None:
            prompt_latent, _ = self.audio.encode_latent(prompt_waveform)
        else:
            prompt_latent = None

        input_ids, inputs_embeds = self.prepare_input_embed(
            prompt, text, spk_emb=spk_emb, instruction=instruction,
            prompt_latent=prompt_latent, prompt_text=prompt_text,
            target_language=target_language
        )

        caches = self.model.make_caches()
        offset = 0
        latent_history = mx.zeros((1, self.history_size, 64))
        if prompt_latent is not None:
            if prompt_latent.shape[1] >= self.history_size:
                latent_history[:, :, :] = prompt_latent[:, -self.history_size:, :]
            else:
                latent_history[:, -prompt_latent.shape[1]:, :] = prompt_latent

        rng_key = mx.random.key(noise_seed) if noise_seed is not None else None

        for step in range(max_decode_steps):
            T = inputs_embeds.shape[1]
            positions = mx.arange(offset, offset + T)
            logits, hidden = self.model(inputs_embeds=inputs_embeds, positions=positions, caches=caches)
            offset += T
            z_diff = hidden[-1][:, -1:, :]

            noise = None
            if rng_key is not None:
                rng_key, sub = mx.random.split(rng_key)
                noise = mx.random.normal((z_diff.shape[0], 64, self.patch_size), key=sub)

            sampled, _ = self.flowloss.sample(
                z_diff, latent_history, cfg=cfg, patch_size=self.patch_size,
                sigma=sigma, temperature=temperature, steps=steps, noise=noise,
            )
            mx.eval(sampled)

            stop_logits = self.stop_head(z_diff)[0][0]
            stop_prob = mx.softmax(stop_logits, axis=-1)[1]
            # Official ming_tiny old_shift decode: the patch is sampled first, then a stop is
            # allowed once len(generated) >= tts_config.minimum_decode_steps (13) and
            # stop_p > stop_threshold. The 0.5B rule (step > 3) cut clips to ~0.2 s.
            min_steps = int(self.tts_cfg.get("minimum_decode_steps", 4))
            threshold = float(self.tts_cfg.get("stop_threshold", 0.5))
            is_last = bool(stop_prob > threshold) and (step + 1) >= min_steps
            yield sampled, is_last
            if is_last:
                break

            inputs_embeds = self.linear_proj_audio(sampled)
            latent_history = mx.concatenate(
                [latent_history[:, self.patch_size:, :], sampled], axis=1
            )

    def generate(self, prompt, text, spk_emb=None, instruction=None,
                 prompt_waveform=None, prompt_text=None,
                 target_language="Chinese",
                 max_decode_steps=200, cfg=2.0, sigma=0.15, temperature=0,
                 steps=10, noise_seed=42):
        latents = []
        for sampled, last in self.sample(
            prompt, text, spk_emb=spk_emb, instruction=instruction,
            prompt_waveform=prompt_waveform, prompt_text=prompt_text,
            target_language=target_language,
            max_decode_steps=max_decode_steps, cfg=cfg, sigma=sigma,
            temperature=temperature, steps=steps, noise_seed=noise_seed
        ):
            latents.append(sampled)
        full = mx.concatenate(latents, axis=1)
        mx.eval(full)
        return self.audio.decode_full(full)
