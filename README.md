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

### 4. 模型量化与轻量化部署 (8-bit / 4-bit)

为了在轻薄 Mac 设备上实现超轻显存占用与极致推理速度，项目原生支持 MLX 8-bit 和 4-bit 权重量化（保持声学敏感头及对齐层高精度，其余线性层采用 `group_size=64` 量化）：

| 规格版本 | 权重体积 | 生成耗时 (6.5s 音频) | 实时率 (RTF) | ASR 字准率 (CER) |
| :--- | :--- | :--- | :--- | :--- |
| **16-bit (bfloat16)** | 3.0 GB | 5.65 秒 | 0.844 | 100% |
| **8-bit (int8)** | **1.7 GB** | **2.53 秒** | **0.395** | **100%** |
| **4-bit (int4)** | **1.0 GB** | **2.55 秒** | **0.374** | **98%** |

直接指定对应量化目录即可无缝加载：
```bash
# 使用 8-bit 量化模型（仅 1.7GB，速度提升 2.1 倍）
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-0.5b-mlx-8bit \
  --text "八比特量化模型极速语音合成测试。" \
  --ip "灵小萌" \
  --out "output/sample_8bit.wav"

# 使用 4-bit 量化模型（仅 1.0GB，超低内存占用）
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-0.5b-mlx-4bit \
  --text "四比特量化模型极速语音合成测试。" \
  --ip "灵小萌" \
  --out "output/sample_4bit.wav"
```

### 5. 试听对比页面 (Interactive Demo Page)

项目中已集成基于现代化暗黑主题的交互式试听对比页面，支持在浏览器中同时横向试听 **官方原始 Demo**、**MLX bf16 (全精度)**、**MLX 8-bit (整数量化)** 与 **MLX 4-bit (超轻量化)** 四路音频，并直观对比耗时、RTF 和模型体积：

```bash
# 启动本地试听服务器
cd /Users/vanch/mlx-Ming-3.0-tts/demo_page && python3 -m http.server 8765 --bind 0.0.0.0 --directory demo_page
```
打开浏览器访问 [http://127.0.0.1:8765/index.html](http://127.0.0.1:8765/index.html) 即可播放试听与评测。


## 📦 Hugging Face 开源模型发布 (Hugging Face Hub)

三个精度的 MLX 模型已全部发布至 Hugging Face，并在模型页面附带了实测生成的测试音频，支持直接通过 `huggingface-cli` 一键拉取：

| 模型仓库 | 精度与量化 | 权重体积 | 推荐场景 | 在线地址 |
| :--- | :--- | :--- | :--- | :--- |
| **[vanch007/Ming-3.0-tts-0.5b-mlx-bf16](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-bf16)** | BFloat16 (全精度) | 3.08 GB | 原生高保真基准对照 | [HF 仓库页面](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-bf16) |
| **[vanch007/Ming-3.0-tts-0.5b-mlx-8bit](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-8bit)** | **8-bit (整数量化)** | **1.76 GB** | **主力推荐：速度提升 2.1 倍，品质无损** | [HF 仓库页面](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-8bit) |
| **[vanch007/Ming-3.0-tts-0.5b-mlx-4bit](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-4bit)** | 4-bit (超轻量) | 1.06 GB | 超轻显存（仅 1GB）设备部署 | [HF 仓库页面](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-4bit) |

直接通过 CLI 一键下载任意版本：
```bash
# 下载 8-bit 主力推荐版
huggingface-cli download vanch007/Ming-3.0-tts-0.5b-mlx-8bit --local-dir models/Ming-3.0-tts-0.5b-mlx-8bit
```


## 🌟 Ming-3.0-tts-tiny (7.9B MoE + KDA 线性注意力)

项目现已完整支持官方 **Ming-3.0-tts-tiny** 架构：
- **混合注意力模型**：MLA（多头潜在注意力）+ 纯原生 MLX 实现的 **Kimi-Delta Attention (KDA)** 线性注意力，彻底摆脱 Linux Triton/CUDA 依赖。
- **128 专家 MoE 架构**：7.9B 总参数量，激活仅 1.3B。
- **独占能力**：原生支持多人/双人对话剧本（`podcast`）多角色交替发音。

```bash
# 1. 运行 Tiny 8-bit 量化模型 (9.78 GB)
python3 -m mlx_ming_tts.pipeline   --model-dir models/Ming-3.0-tts-tiny-mlx-8bit   --text "傍晚的风从山口吹过来，远处的云慢慢染成了金色。"   --out "output/tiny_out.wav"

# 2. 运行 Tiny 4-bit 量化模型 (5.42 GB)
python3 -m mlx_ming_tts.pipeline   --model-dir models/Ming-3.0-tts-tiny-mlx-4bit   --text "超轻量 4-bit Tiny 模型测试。"   --out "output/tiny_4bit.wav"
```
