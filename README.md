# mlx-Ming-3.0-tts

Pure Apple Silicon MLX port of [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) (Qwen2 / BailingMoE + Flow Matching + AudioVAE).

## 核心特性

- **纯 MLX 原生运行时**：
  - 声学解码器与 ISTFT 逆短时傅里叶变换完全由 `mx.fft.irfft` 与 MLX 数组切片重叠相加（OLA）原生实现，完全消除 NumPy/PyTorch 运行时依赖。
  - 音频保存基于 Python 标准库 `wave`，直接输出 44.1kHz 16-bit 单声道 PCM WAV。
- **长文本自然连贯度优化**：
  - 严格匹配 Qwen2 AudioVAE 解码器的 `sliding_window=64` 滑动窗口注意力机制，消除长文本后半段音质退化。
  - 严格对齐官方 Dense 0.5B 的 ChatML Token 契约（指令终止符 `<|im_end|>`），消除自回归先验漂移与发音断续。
  - 支持官方 100+ 预置 IP 音色（如 `灵小萌`、`灵小锋`、`US_Emma`）及自然语言音色设计。

## 目录结构

```text
mlx-Ming-3.0-tts/
├── mlx_ming_tts/
│   ├── models/
│   │   ├── audio_vae.py       # 纯 MLX AudioVAE 编解码器与 ISTFT 重构头
│   │   ├── bailingmm.py       # 文本-声学自回归与 Flow Matching 协调调度
│   │   ├── flow.py            # FlowLoss、DiT 与 Euler ODE 求解器
│   │   ├── qwen2.py           # 支持滑动窗口注意力与 KV Cache 的 Qwen2 模型
│   │   └── bailing_moe.py     # Bailing MoE 架构支持
│   ├── ops/
│   │   ├── attention.py       # 包含 sliding_window 的因果掩码与 SDPA
│   │   ├── norm.py            # RMSNorm 算子
│   │   └── rope.py            # 旋转位置编码 RoPE
│   ├── convert.py             # HF 权重一键转标准 MLX safetensors
│   └── pipeline.py            # 高层语音合成与音频导出入口
└── models/
    ├── Ming-3.0-tts-0.5b/     # 原始 HF 格式权重
    └── Ming-3.0-tts-0.5b-mlx/ # 转换后的纯 MLX safetensors 权重与配置文件
```

## 快速使用

### 1. 模型转换（若已转换可跳过）

```bash
python3 -m mlx_ming_tts.convert   --hf-dir models/Ming-3.0-tts-0.5b   --out-dir models/Ming-3.0-tts-0.5b-mlx   --dtype bfloat16
```

### 2. 语音合成

合成中文（使用预置音色 `灵小萌`）：
```bash
python3 -m mlx_ming_tts.pipeline   --model-dir models/Ming-3.0-tts-0.5b-mlx   --text "你好，我是纯MLX架构驱动的Ming语音合成系统。现在声音更加清晰自然了。"   --ip "灵小萌"   --out "output/chinese_sample.wav"
```

合成英文（使用预置音色 `US_Emma`）：
```bash
python3 -m mlx_ming_tts.pipeline   --model-dir models/Ming-3.0-tts-0.5b-mlx   --text "Hello! This is Ming-3.0-tts running completely on Apple Silicon with Apple MLX."   --ip "US_Emma"   --out "output/english_sample.wav"
```

### 3. 零样本声音克隆 (Zero-Shot Voice Cloning)

提供一段 3~10 秒的参考音频及对应转写文本，即可一键克隆音色：
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-0.5b-mlx \
  --ref-audio "path/to/reference.wav" \
  --prompt-text "参考音频的准确朗读文本" \
  --text "需要使用该音色合成的新文本" \
  --out "output/cloned_voice.wav"
```
