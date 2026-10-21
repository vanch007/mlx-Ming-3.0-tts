"""Main orchestration: AR (LLM) + Flow Matching alternating loop in pure MLX.

Mirrors modeling_bailingmm.py: prepare_input_embed, the AR+Flow sample loop,
and full-sequence audio decode. Dense 0.5B uses Qwen2; 16.8B-A3B uses the
Bailing MoE LLM. Flow/VAE/Aggregator stay config-parameterized and shared.
"""

import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_ming_tts.models.qwen2 import Qwen2ForCausalLM
from mlx_ming_tts.models.bailing_moe import BailingMoeForCausalLM
from mlx_ming_tts.models.flow import FlowLoss, Aggregator
from mlx_ming_tts.models.audio_vae import AudioVAE
from mlx_ming_tts import convert


class BailingMM(nn.Module):
    """MLX text-to-speech orchestrator for dense and MoE Ming checkpoints."""

    def __init__(self, model_dir, tokenizer):
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.config = cfg
        self.llm_cfg = cfg["llm_config"]
        self.is_moe = bool(self.llm_cfg.get("num_experts"))
        self.atc = cfg["audio_tokenizer_config"]
        self.ditar = cfg["ditar_config"]
        self.agg_cfg = cfg["aggregator_config"]

        self.latent_dim = self.atc["enc_kwargs"]["latent_dim"]
        self.patch_size = self.ditar["patch_size"]
        self.history_patch_size = self.ditar.get("history_patch_size", self.patch_size)
        hidden = self.llm_cfg["hidden_size"]

        if self.is_moe:
            self.model = BailingMoeForCausalLM(self.llm_cfg)
        else:
            self.model = Qwen2ForCausalLM(self.llm_cfg)
        self.audio = AudioVAE(self.atc)
        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim, hidden_size=self.agg_cfg["hidden_size"],
            depth=self.agg_cfg["depth"], num_heads=self.agg_cfg["num_heads"],
            mlp_ratio=self.agg_cfg["mlp_ratio"], llm_input_dim=hidden,
        )
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim, llm_cond_dim=hidden,
            hidden_size=self.ditar["hidden_size"], depth=self.ditar["depth"],
            num_heads=self.ditar["num_heads"], mlp_ratio=self.ditar["mlp_ratio"],
        )
        self.stop_head = nn.Linear(hidden, 2, bias=True)
        self.spk_head = nn.Linear(192, hidden, bias=True)

    # ---- weight loading -------------------------------------------------
    def load_all_weights(self, dtype=mx.float32):
        """Load every component from the HF checkpoint via convert.py."""
        if self.is_moe:
            llm_w = convert.convert_moe(self.model_dir, self.llm_cfg, dtype)
        else:
            n_layers = self.llm_cfg["num_hidden_layers"]
            tie_embed = self.llm_cfg.get("tie_word_embeddings", True)
            llm_w = convert.convert_llm(self.model_dir, n_layers, dtype, tie_embed=tie_embed)
        self.model.load_weights([(k, v) for k, v in llm_w.items()])

        vae_names = [k for k, _ in tree_flatten(self.audio.parameters())]
        vae_w = convert.convert_vae(self.model_dir, vae_names, dtype)
        self.audio.load_weights(list(vae_w.items()))

        agg_names = [k for k, _ in tree_flatten(self.linear_proj_audio.parameters())]
        agg_w = convert.convert_by_prefix(self.model_dir, "linear_proj_audio", agg_names, dtype)
        self.linear_proj_audio.load_weights(list(agg_w.items()))

        flow_names = [k for k, _ in tree_flatten(self.flowloss.parameters())]
        flow_w = convert.convert_by_prefix(self.model_dir, "flowloss", flow_names, dtype)
        self.flowloss.load_weights(list(flow_w.items()))

        src = convert._load_source(self.model_dir)
        self.stop_head.load_weights([
            ("weight", src["stop_head.weight"].astype(dtype)),
            ("bias", src["stop_head.bias"].astype(dtype)),
        ])
        self.spk_head.load_weights([
            ("weight", src["spk_head.weight"].astype(dtype)),
            ("bias", src["spk_head.bias"].astype(dtype)),
        ])
        self.eval()

    def load_mlx_weights(self, weights_path):
        """Load preconverted MLX weights from a flat safetensors file."""
        weights = mx.load(weights_path)
        self.load_weights(list(weights.items()))
        self.eval()

    def token_embeddings(self, input_ids):
        if self.is_moe:
            return self.model.model.word_embeddings(input_ids)
        return self.model.model.embed_tokens(input_ids)

    # ---- input preparation ---------------------------------------------
    def prepare_input_embed(self, prompt, text, spk_emb=None, instruction=None,
                            prompt_latent=None, prompt_text=None,
                            use_zero_spk_emb=False):
        """Dense-model input assembly. Returns (input_ids list, inputs_embeds mx)."""
        tok = self.tokenizer
        spk_emb_prompt = []
        spk_vectors = []
        if spk_emb is None and use_zero_spk_emb:
            spk_emb = [mx.zeros((1, 192), dtype=self.spk_head.weight.dtype)]
        if spk_emb is not None:
            for i, se in enumerate(spk_emb):
                se_proj = self.spk_head(se)
                if use_zero_spk_emb:
                    se_proj = mx.zeros_like(se_proj)
                spk_vectors.append(se_proj)
                if self.is_moe:
                    spk_emb_prompt.extend(
                        tok.encode(f"  speaker_{i+1}:")
                        + tok.encode("<spk>")
                        + tok.encode("<audioPatch>")
                        + tok.encode("</spk>" + chr(10))
                    )
                else:
                    spk_emb_prompt.extend(
                        tok.encode(f"  speaker_{i+1}:")
                        + tok.encode("<|vision_start|>")
                        + tok.encode("<|vision_pad|>")
                        + tok.encode("<|vision_end|>" + chr(10))
                    )

        instruction_prompt = []
        if instruction is not None:
            instruction_prompt = tok.encode(instruction) + tok.encode("<|endoftext|>" if self.is_moe else "<|im_end|>")

        prompt_text_token = []
        prompt_latent_token = []
        if prompt_latent is not None and prompt_text is not None:
            bsz = prompt_latent.shape[0]
            prompt_latent = prompt_latent.reshape(-1, self.patch_size, self.latent_dim)
            prompt_latent = self.linear_proj_audio(prompt_latent)
            prompt_latent = prompt_latent.reshape(bsz, -1, prompt_latent.shape[-1])
            prompt_text_token = tok.encode(prompt_text)
            patch_id = tok.convert_tokens_to_ids("<audioPatch>")
            prompt_latent_token = [patch_id] * prompt_latent.shape[1]

        prompt2 = tok.encode(" Text input:" + chr(10))
        if all(k in text for k in ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")):
            prompt2 = []

        if self.is_moe:
            input_part = (
                tok.encode("<role>HUMAN</role>")
                + tok.encode(prompt)
                + spk_emb_prompt
                + prompt2
                + prompt_text_token
                + tok.encode(text)
                + tok.encode("<role>ASSISTANT</role>")
                + instruction_prompt
                + tok.encode("<audio>")
                + prompt_latent_token
            )
        else:
            input_part = (
                tok.encode("<|im_start|>system" + chr(10) + "You are a helpful assistant.<|im_end|>" + chr(10))
                + tok.encode("<|im_start|>user" + chr(10))
                + tok.encode(prompt)
                + spk_emb_prompt
                + prompt2
                + prompt_text_token
                + tok.encode(text)
                + tok.encode("<|im_end|>" + chr(10))
                + tok.encode("<|im_start|>assistant" + chr(10))
                + instruction_prompt
                + tok.encode("<audio>")
                + prompt_latent_token
            )

        input_ids = input_part
        inputs_embeds = self.token_embeddings(mx.array([input_ids]))

        if spk_emb is not None:
            spk_token_id = tok.convert_tokens_to_ids("<spk>" if self.is_moe else "<|vision_start|>")
            spk_indices = [idx for idx, t_id in enumerate(input_ids) if t_id == spk_token_id]
            assert len(spk_indices) > 0, "no speaker start token found"
            for i, se in enumerate(spk_vectors):
                pos = spk_indices[i] + 1
                inputs_embeds[0, pos] = se[0]

        if prompt_latent is not None:
            audio_token_id = tok.convert_tokens_to_ids("<audio>")
            audio_indices = [idx for idx, t_id in enumerate(input_ids) if t_id == audio_token_id]
            assert len(audio_indices) > 0, "no <audio> token found"
            start = audio_indices[0] + 1
            end = start + prompt_latent.shape[1]
            inputs_embeds[0, start:end, :] = prompt_latent[0]

        return input_ids, inputs_embeds

    # ---- AR + Flow sampling loop ---------------------------------------
    def sample(self, prompt, text, spk_emb=None, instruction=None,
               prompt_waveform=None, prompt_text=None,
               max_decode_steps=200, cfg=2.0, sigma=0.25, temperature=0,
               steps=10, use_zero_spk_emb=False, noise_seed=None):
        if prompt_waveform is not None and prompt_text is not None:
            prompt_latent, _ = self.audio.encode_latent(prompt_waveform)
        else:
            prompt_latent = None

        input_ids, inputs_embeds = self.prepare_input_embed(
            prompt, text, spk_emb=spk_emb, instruction=instruction,
            prompt_latent=prompt_latent, prompt_text=prompt_text,
            use_zero_spk_emb=use_zero_spk_emb,
        )

        caches = self.model.make_caches()
        offset = 0
        latent_history = mx.zeros((1, self.history_patch_size, self.latent_dim))
        if prompt_latent is not None:
            if prompt_latent.shape[1] >= self.history_patch_size:
                latent_history[:, :, :] = prompt_latent[:, -self.history_patch_size:, :]
            else:
                start = self.history_patch_size - prompt_latent.shape[1]
                latent_history[:, start:, :] = prompt_latent
        rng_key = mx.random.key(noise_seed) if noise_seed is not None else None

        for step in range(max_decode_steps):
            T = inputs_embeds.shape[1]
            positions = mx.arange(offset, offset + T)
            logits, hidden = self.model(inputs_embeds=inputs_embeds, positions=positions,
                                        caches=caches)
            offset += T
            z_diff = hidden[-1][:, -1:, :]

            noise = None
            if rng_key is not None:
                rng_key, sub = mx.random.split(rng_key)
                noise = mx.random.normal((z_diff.shape[0], self.latent_dim, self.patch_size), key=sub)

            sampled, _ = self.flowloss.sample(
                z_diff, latent_history, cfg=cfg, patch_size=self.patch_size,
                sigma=sigma, temperature=temperature, steps=steps, noise=noise,
            )
            mx.eval(sampled)

            stop_logits = self.stop_head(z_diff)[0][0]
            stop_prob = mx.softmax(stop_logits, axis=-1)[1]
            is_last = bool(stop_prob > 0.5) and step > 3
            yield sampled, is_last
            if is_last:
                break

            inputs_embeds = self.linear_proj_audio(sampled)
            latent_history = mx.concatenate(
                [latent_history[:, self.patch_size:, :], sampled], axis=1
            )

    def generate(self, prompt, text, spk_emb=None, instruction=None,
                 prompt_waveform=None, prompt_text=None,
                 max_decode_steps=200, cfg=2.0, sigma=0.25, temperature=0,
                 steps=10, use_zero_spk_emb=False, noise_seed=None):
        """Full text-to-speech: returns a 1-D MLX array waveform."""
        latents = []
        for sampled, last in self.sample(
            prompt, text, spk_emb=spk_emb, instruction=instruction,
            prompt_waveform=prompt_waveform, prompt_text=prompt_text,
            max_decode_steps=max_decode_steps, cfg=cfg, sigma=sigma,
            temperature=temperature, steps=steps, use_zero_spk_emb=use_zero_spk_emb,
            noise_seed=noise_seed,
        ):
            latents.append(sampled)
        full = mx.concatenate(latents, axis=1)
        mx.eval(full)
        return self.audio.decode_full(full)


BailingMMDense = BailingMM
