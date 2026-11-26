"""Tokenizer and prompt contracts used by LingV3 TTS checkpoints."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional


LEGACY_ANTMMF_PROTOCOL = "legacy_antmmf"
THINKING_V1_PROTOCOL = "thinking_v1"
NATIVE_FP32_SAMPLING = "native_fp32"
DEVICE_DTYPE_SAMPLING = "device_dtype"
OLD_SHIFT_DECODE_ALIGNMENT = "old_shift"
UNSHIFTED_DECODE_ALIGNMENT = "unshifted"

# Keep this order identical to the native training-side multimodal token list.
# The step270000 checkpoint learned these exact embedding rows.
LEGACY_MULTIMODAL_TOKENS = (
    "<imagePatch>",
    "<image>",
    "</image>",
    "<video>",
    "</video>",
    "<gen_imagePatch>",
    "<gen_image>",
    "</gen_image>",
    "<imageHere>",
    "<end_of_chunk>",
    "<end_of_audio>",
    "<audioPatch>",
    "<audio>",
    "</audio>",
    "<gen_audioPatch>",
    "<gen_audio>",
    "</gen_audio>",
    "<audioHere>",
    "<framePatch>",
    "<text>",
    "<asr>",
    "<tts>",
    "<spk>",
    "</spk>",
)
LEGACY_BASE_TOKENIZER_SIZE = 157153
LEGACY_TOKEN_IDS = {
    token: LEGACY_BASE_TOKENIZER_SIZE + index
    for index, token in enumerate(LEGACY_MULTIMODAL_TOKENS)
}
LEGACY_TTS_TOKEN_IDS = {
    "<audioPatch>": 157164,
    "<audio>": 157165,
    "</audio>": 157166,
    "<spk>": 157175,
    "</spk>": 157176,
}

LEGACY_USER_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
    "  speaker_1:"
)
LEGACY_USER_PREFIX_NO_RML = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
LEGACY_TEXT_PREFIX = "\n Text input:\n"
LEGACY_TEXT_PREFIX_NO_RML = " Text input:\n"
LEGACY_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

THINKING_SYSTEM_PREFIX = "<role>SYSTEM</role>detailed thinking off"
THINKING_USER_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
    "  speaker_1:"
)
THINKING_USER_PREFIX_NO_RML = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
THINKING_TEXT_PREFIX = "\n Text input:\n"
THINKING_TEXT_PREFIX_NO_RML = " Text input:\n"
THINKING_ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n<think></think>"


def protocol_version(tts_config: Mapping[str, object]) -> str:
    value = tts_config.get("protocol_version")
    if value is None:
        value = (
            THINKING_V1_PROTOCOL
            if tts_config.get("audio_patch_token_id") == 157168
            else LEGACY_ANTMMF_PROTOCOL
        )
    value = str(value)
    if value not in {LEGACY_ANTMMF_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            f"Unsupported LingV3 TTS protocol_version={value!r}; expected "
            f"{LEGACY_ANTMMF_PROTOCOL!r} or {THINKING_V1_PROTOCOL!r}"
        )
    return value


def prompt_format(tts_config: Mapping[str, object]) -> str:
    """Return the chat/prompt layout independently of token row names.

    Older bundles used ``protocol_version`` for both dimensions. Newer
    Megatron tasks can deliberately use the Bailing V3 runtime while their
    ``config_vl`` tokenizer and prompt retain the AntMMF multimodal rows.
    """

    value = str(
        tts_config.get("prompt_format", protocol_version(tts_config))
    )
    if value not in {LEGACY_ANTMMF_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            f"Unsupported LingV3 TTS prompt_format={value!r}; expected "
            f"{LEGACY_ANTMMF_PROTOCOL!r} or {THINKING_V1_PROTOCOL!r}"
        )
    return value


def tokenizer_contract(tts_config: Mapping[str, object]) -> str:
    """Return the multimodal token vocabulary used during training."""

    value = str(
        tts_config.get("tokenizer_contract", protocol_version(tts_config))
    )
    if value not in {LEGACY_ANTMMF_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            "Unsupported LingV3 TTS tokenizer_contract="
            f"{value!r}; expected {LEGACY_ANTMMF_PROTOCOL!r} or "
            f"{THINKING_V1_PROTOCOL!r}"
        )
    return value


def sampling_contract(tts_config: Mapping[str, object]) -> str:
    """Return the Flow RNG/dtype contract independently of prompt layout.

    Old bundles overloaded ``protocol_version`` for this choice. A mixed
    Megatron contract can use the Bailing V3 cache/stop order while the rc1.6
    Flow sampler still creates fp32 history and CPU fp32 initial noise with
    the global RNG streams.
    """

    default = (
        NATIVE_FP32_SAMPLING
        if protocol_version(tts_config) == LEGACY_ANTMMF_PROTOCOL
        else DEVICE_DTYPE_SAMPLING
    )
    value = str(tts_config.get("sampling_contract", default))
    if value not in {NATIVE_FP32_SAMPLING, DEVICE_DTYPE_SAMPLING}:
        raise ValueError(
            "Unsupported LingV3 TTS sampling_contract="
            f"{value!r}; expected {NATIVE_FP32_SAMPLING!r} or "
            f"{DEVICE_DTYPE_SAMPLING!r}"
        )
    return value


def decode_alignment(tts_config: Mapping[str, object]) -> str:
    """Return the training label shift used by autoregressive TTS decode.

    Historical bundles coupled this choice to ``protocol_version``.  Newer
    Megatron jobs may use the modern cache/runtime protocol while retaining
    ``shift_tts_labels_like_old=true``.  Keep the explicit dimension optional
    so existing converted bundles retain their exact behavior.
    """

    default = (
        OLD_SHIFT_DECODE_ALIGNMENT
        if protocol_version(tts_config) == LEGACY_ANTMMF_PROTOCOL
        else UNSHIFTED_DECODE_ALIGNMENT
    )
    value = str(tts_config.get("decode_alignment", default))
    if value not in {
        OLD_SHIFT_DECODE_ALIGNMENT,
        UNSHIFTED_DECODE_ALIGNMENT,
    }:
        raise ValueError(
            "Unsupported LingV3 TTS decode_alignment="
            f"{value!r}; expected {OLD_SHIFT_DECODE_ALIGNMENT!r} or "
            f"{UNSHIFTED_DECODE_ALIGNMENT!r}"
        )
    return value


def task_decode_alignments(
    tts_config: Mapping[str, object],
) -> Dict[str, str]:
    """Validate task-specific label-shift provenance from conversion.

    Mixed-task checkpoints can contain a task whose training label shift is
    different from the bundle-wide default.  Keep this metadata optional for
    backward compatibility; a task that needs it must fail closed at runtime.
    """

    raw = tts_config.get("task_decode_alignment", {})
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("task_decode_alignment must be a JSON object")
    result: Dict[str, str] = {}
    for raw_task, raw_alignment in raw.items():
        task = str(raw_task).strip().lower().replace("-", "_")
        if not task:
            raise ValueError("task_decode_alignment contains an empty task")
        alignment = str(raw_alignment).strip()
        if alignment not in {
            OLD_SHIFT_DECODE_ALIGNMENT,
            UNSHIFTED_DECODE_ALIGNMENT,
        }:
            raise ValueError(
                "Unsupported task decode alignment for "
                f"{task!r}: {alignment!r}"
            )
        if task in result:
            raise ValueError(
                f"task_decode_alignment contains duplicate task {task!r}"
            )
        result[task] = alignment
    return result


def with_rml(tts_config: Mapping[str, object]) -> bool:
    """Return whether the training prompt contains speaker RML.

    Historical bundles predate this explicit field and all used
    ``with_RML=True``.  Defaulting to ``True`` therefore preserves their
    prompt and speaker-injection behavior byte-for-byte.  A no-RML bundle must
    opt in explicitly with the JSON boolean ``with_rml=false``.
    """

    value = tts_config.get("with_rml", True)
    if not isinstance(value, bool):
        raise ValueError(
            f"TTS with_rml must be a JSON boolean, got {value!r}"
        )
    return value


def encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def ensure_legacy_antmmf_tokenizer(tokenizer) -> int:
    """Add the native multimodal tokens in their training-time order.

    A later LingV3 tokenizer cannot be repaired by appending aliases: its
    occupied embedding rows already have different meanings.  Fail with an
    actionable error in that case.
    """

    actual = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in LEGACY_MULTIMODAL_TOKENS
    }
    if actual == LEGACY_TOKEN_IDS:
        return 0

    absent = all(token_id is None for token_id in actual.values())
    if absent and len(tokenizer) == LEGACY_BASE_TOKENIZER_SIZE:
        added = tokenizer.add_tokens(
            list(LEGACY_MULTIMODAL_TOKENS), special_tokens=True
        )
        if added != len(LEGACY_MULTIMODAL_TOKENS):
            raise RuntimeError(
                "Failed to add the complete native multimodal token list: "
                f"added {added}/{len(LEGACY_MULTIMODAL_TOKENS)}"
            )
        repaired = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in LEGACY_MULTIMODAL_TOKENS
        }
        if repaired != LEGACY_TOKEN_IDS:
            raise RuntimeError(
                "Tokenizer assigned unexpected native token IDs: "
                f"{repaired}"
            )
        return added

    mismatches = {
        token: (LEGACY_TOKEN_IDS[token], actual[token])
        for token in LEGACY_MULTIMODAL_TOKENS
        if actual[token] != LEGACY_TOKEN_IDS[token]
    }
    preview = ", ".join(
        f"{token}: expected {want}, got {got}"
        for token, (want, got) in list(mismatches.items())[:8]
    )
    raise ValueError(
        "This tokenizer does not have the legacy AntMMF token rows and cannot "
        "be repaired by appending tokens. Load the base tokenizer from "
        "the checkpoint's training-side for_infer directory (or convert the "
        "model with --tokenizer-dir). "
        + preview
    )


def _expected_tts_tokens(
    tts_config: Mapping[str, object],
) -> Dict[str, int]:
    if tokenizer_contract(tts_config) == LEGACY_ANTMMF_PROTOCOL:
        token_names = {
            "<audioPatch>": "audio_patch_token_id",
            "<audio>": "audio_start_token_id",
            "</audio>": "audio_end_token_id",
            "<spk>": "speaker_start_token_id",
            "</spk>": "speaker_end_token_id",
        }
    else:
        token_names = {
            "<|audio_pad|>": "audio_patch_token_id",
            "<|audio_start|>": "audio_start_token_id",
            "<|audio_end|>": "audio_end_token_id",
            "<spk>": "speaker_start_token_id",
            "</spk>": "speaker_end_token_id",
        }
    return {
        token: int(tts_config[config_name])
        for token, config_name in token_names.items()
    }


def validate_tts_tokenizer(
    tokenizer,
    tts_config: Mapping[str, object],
    config_eos,
) -> Dict[str, int]:
    expected = _expected_tts_tokens(tts_config)
    actual = {
        token: tokenizer.convert_tokens_to_ids(token) for token in expected
    }
    mismatches = {
        token: (expected[token], actual[token])
        for token in expected
        if actual[token] != expected[token]
    }
    if mismatches:
        details = ", ".join(
            f"{token}: expected {want}, got {got}"
            for token, (want, got) in mismatches.items()
        )
        raise ValueError(
            "Tokenizer IDs do not match LingV3 TTS training: " + details
        )
    for token, token_id in expected.items():
        encoded = encode(tokenizer, token)
        if encoded != [token_id]:
            raise ValueError(
                f"{token!r} must encode to [{token_id}], got {encoded}"
            )

    config_eos_ids = (
        set(config_eos)
        if isinstance(config_eos, (list, tuple, set))
        else {config_eos}
    )
    if tokenizer.eos_token_id not in config_eos_ids:
        raise ValueError(
            "Tokenizer EOS does not match the LingV3 config: "
            f"{tokenizer.eos_token_id} not in {sorted(config_eos_ids)}"
        )
    expected_prompt_eos = tts_config.get("prompt_eos_token_id")
    if (
        expected_prompt_eos is not None
        and tokenizer.eos_token_id != int(expected_prompt_eos)
    ):
        raise ValueError(
            "Tokenizer EOS does not match the training prompt separator: "
            f"expected {int(expected_prompt_eos)}, got "
            f"{tokenizer.eos_token_id}"
        )
    return actual


def _build_legacy_antmmf_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
) -> tuple[List[int], Optional[int]]:
    # Exact sample_instruct_tts(with_RML=True, RML_length=1,
    # with_caption=True, caption="") sequence.  There is deliberately no
    # SYSTEM prefix, EOS separator, or <think></think>.
    rml_enabled = with_rml(tts_config)
    ids = encode(
        tokenizer,
        LEGACY_USER_PREFIX if rml_enabled else LEGACY_USER_PREFIX_NO_RML,
    )
    speaker_embedding_position: Optional[int] = None
    if rml_enabled:
        ids.append(int(tts_config["speaker_start_token_id"]))
        speaker_embedding_position = len(ids)
        ids.append(int(tts_config["audio_patch_token_id"]))
        ids.append(int(tts_config["speaker_end_token_id"]))
    ids.extend(
        encode(
            tokenizer,
            LEGACY_TEXT_PREFIX if rml_enabled else LEGACY_TEXT_PREFIX_NO_RML,
        )
    )
    if prompt_text is not None:
        ids.extend(encode(tokenizer, prompt_text))
    ids.extend(encode(tokenizer, target_text))
    ids.extend(encode(tokenizer, LEGACY_ASSISTANT_PREFIX))
    ids.append(int(tts_config["audio_start_token_id"]))
    ids.extend(
        [int(tts_config["audio_patch_token_id"])]
        * prompt_latent_tokens
    )
    return ids, speaker_embedding_position


def _build_thinking_v1_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
) -> tuple[List[int], Optional[int]]:
    ids = encode(tokenizer, THINKING_SYSTEM_PREFIX)
    ids.append(tokenizer.eos_token_id)
    rml_enabled = with_rml(tts_config)
    ids.extend(
        encode(
            tokenizer,
            THINKING_USER_PREFIX if rml_enabled else THINKING_USER_PREFIX_NO_RML,
        )
    )
    speaker_embedding_position: Optional[int] = None
    if rml_enabled:
        ids.append(int(tts_config["speaker_start_token_id"]))
        speaker_embedding_position = len(ids)
        ids.append(int(tts_config["audio_patch_token_id"]))
        ids.append(int(tts_config["speaker_end_token_id"]))
    ids.extend(
        encode(
            tokenizer,
            THINKING_TEXT_PREFIX if rml_enabled else THINKING_TEXT_PREFIX_NO_RML,
        )
    )
    if prompt_text is not None:
        ids.extend(encode(tokenizer, prompt_text))
    ids.extend(encode(tokenizer, target_text))
    ids.append(tokenizer.eos_token_id)
    ids.extend(encode(tokenizer, THINKING_ASSISTANT_PREFIX))
    ids.append(int(tts_config["audio_start_token_id"]))
    ids.extend(
        [int(tts_config["audio_patch_token_id"])]
        * prompt_latent_tokens
    )
    return ids, speaker_embedding_position


def build_tts_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
) -> tuple[List[int], Optional[int]]:
    """Build the LingV2-compatible zero-shot prompt token sequence.

    When reference conditioning is present, the reference transcript is
    placed immediately before the target transcript and one audio placeholder
    is appended for every projected prompt-latent patch.  Passing neither
    keeps the historical speaker-only/no-reference prompt byte-for-byte.
    When ``with_rml=false`` is declared by the converted model, the complete
    ``speaker_1:<spk><audioPatch></spk>`` input line is omitted and the
    returned speaker position is ``None``.
    """

    if prompt_latent_tokens < 0:
        raise ValueError("prompt_latent_tokens must be non-negative")
    has_prompt_text = prompt_text is not None
    has_prompt_latent = prompt_latent_tokens > 0
    if has_prompt_text != has_prompt_latent:
        raise ValueError(
            "prompt_text and prompt latent tokens must be provided together"
        )
    if prompt_text == "":
        raise ValueError("prompt_text must not be empty")

    if prompt_format(tts_config) == LEGACY_ANTMMF_PROTOCOL:
        return _build_legacy_antmmf_prompt_ids(
            tokenizer,
            target_text,
            tts_config,
            prompt_text=prompt_text,
            prompt_latent_tokens=prompt_latent_tokens,
        )
    return _build_thinking_v1_prompt_ids(
        tokenizer,
        target_text,
        tts_config,
        prompt_text=prompt_text,
        prompt_latent_tokens=prompt_latent_tokens,
    )
