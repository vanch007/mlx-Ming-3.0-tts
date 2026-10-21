# mlx-Ming-3.0-tts

<p align="center">
  <a href="#english"><b>English</b></a> | <a href="#中文"><b>中文说明</b></a>
</p>

Pure Apple Silicon **MLX** native port of [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) — supporting both **0.5B Dense** and **Tiny 7.9B MoE (128 Experts + KDA)** architectures across multiple precisions (bf16 / 8-bit / 4-bit).

Completely free of Linux-only `triton` kernels, PyTorch, and NumPy vocoder runtime dependencies, running natively on Apple Silicon Metal GPUs (M1/M2/M3/M4).

---

<span id="english"></span>
## 🇬🇧 English Documentation

### Key Features

- **Dual Architecture Support**:
  - **0.5B Dense Architecture**: Ultra-lightweight and fast, high-fidelity zero-shot voice cloning, and minimal memory footprint.
  - **Tiny 7.9B MoE Architecture**: 128 routed experts + 1 shared expert + KDA recurrent linear attention, featuring alternating multi-speaker podcast dialogue, nuanced emotion, and dialect controls.
- **Pure MLX Native Acoustic Engine**:
  - Audio synthesis pipeline and Inverse Short-Time Fourier Transform (ISTFT) are implemented natively with `mx.fft.irfft` and MLX Overlap-Add (OLA), eliminating runtime dependencies on PyTorch and NumPy.
  - Audio export uses standard Python library `wave`, saving directly to 44.1kHz 16-bit PCM WAV.
- **Zero Triton Dependency**:
  - Native vectorized recurrence replaces Linux-only Triton kernels for KDA (Kimi Delta Attention), delivering full Metal hardware acceleration on macOS.
- **Persistent Step Caching**:
  - Implemented `KDACache` (recurrent state & 1D causal convolution) and `MLACache` (KV cache) to maintain context stability across autoregressive generation steps.
- **Comprehensive Scenario Coverage**:
  - Supports zero-shot voice cloning, podcast multi-speaker scripts, paralinguistic controls (`[Laughter]`, `[Sigh]`), built-in voice personas (Lingxiaomeng, Lingxiaofeng, US_James, etc.), complex text normalization, multilingual synthesis (Japanese, Korean, French, English), and dialect control.

### Model Hub (Hugging Face Matrix)

All 6 pre-quantized MLX models are published on Hugging Face with complete model cards and benchmark audio samples:

#### 1. Ming-3.0-tts-tiny (7.9B MoE · 128 Experts + KDA)

| Model Repository | Precision | Size | Description & Performance | Link |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-tiny-mlx-8bit`** | 8-bit Quantized | **9.78 GB** | **Recommended Mainline**: Balanced quality & speed, RTF ~1.68 | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-4bit`** | 4-bit Quantized | **5.42 GB** | **Ultra-Lightweight**: 70% memory reduction, ideal for 8GB/16GB unified memory | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-bf16`** | bfloat16 Full | **17.96 GB** | **Lossless Baseline**: 100% full-precision floating-point weights | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-bf16) |

#### 2. Ming-3.0-tts-0.5b (0.5B Dense Architecture)

| Model Repository | Precision | Size | Description & Performance | Link |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-8bit`** | 8-bit Quantized | **1.76 GB** | **Recommended Mainline**: Real-time speed on Apple Silicon, RTF ~1.06 | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-4bit`** | 4-bit Quantized | **1.06 GB** | **Ultra-Compact**: ~1 GB file size, minimal memory overhead | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-bf16`** | bfloat16 Full | **3.08 GB** | **Dense Full Precision**: Standard reference float model | [🤗 Model](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-bf16) |

### Offline & Online Audio Evaluation Page

- **Online Demo URL**: **[https://vanch007.github.io/mlx-Ming-3.0-tts/](https://vanch007.github.io/mlx-Ming-3.0-tts/)**
- **Self-Contained File**: `standalone_compare.html` (40.19 MB)
- **Features**: Built-in Base64 waveforms for 15 test scenarios comparing all 6 model versions side-by-side with instantaneous English / Chinese language switching. No server or network required.

### Quick Start

#### 1. Setup

```bash
git clone https://github.com/vanch007/mlx-Ming-3.0-tts.git
cd mlx-Ming-3.0-tts
pip install mlx transformers soundfile onnxruntime
```

#### 2. Download Recommended Weights

```bash
# Download Tiny 7.9B MoE 8-bit model
huggingface-cli download vanch007/Ming-3.0-tts-tiny-mlx-8bit --local-dir models/Ming-3.0-tts-tiny-mlx-8bit

# Download 0.5B Dense 8-bit model
huggingface-cli download vanch007/Ming-3.0-tts-0.5b-mlx-8bit --local-dir models/Ming-3.0-tts-0.5b-mlx-8bit
```

#### 3. CLI Synthesis

**Built-in Persona (`灵小萌`)**:
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-tiny-mlx-8bit \
  --text "夜晚的微风吹散了白天的喧嚣，愿好梦常伴你左右。" \
  --ip "灵小萌" \
  --out "output_lingxiaomeng.wav"
```

**Podcast Dialogue (Multi-Speaker Script)**:
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-tiny-mlx-8bit \
  --text "speaker_1: 你最近是不是又去那家面馆了？\nspeaker_2: 对啊，排了半个小时的队，不过真的值得。" \
  --out "output_podcast.wav"
```

**Zero-Shot Voice Cloning**:
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-0.5b-mlx-8bit \
  --ref-audio "path/to/reference.wav" \
  --prompt-text "Exact transcript of the reference audio" \
  --text "New text to be spoken with cloned voice" \
  --out "output_cloned.wav"
```

#### 4. Python API

```python
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

# Load native MLX engine
runner = MingMLX("models/Ming-3.0-tts-tiny-mlx-8bit")

# Synthesize audio
wav = runner.generate(
    DEFAULT_TTS_PROMPT,
    text="Welcome to Ming-3.0-tts running purely on Apple Silicon with MLX.",
    ip="灵小萌",
    sigma=0.15,
    cfg=2.0,
    noise_seed=42
)

# Export 44.1kHz 16-bit PCM WAV
save_wav(wav, "output.wav", runner.sample_rate)
```

---

<span id="中文"></span>
## 🇨🇳 中文说明文档

### 核心特性

- **双模型双架构支持**：
  - **0.5B 稠密模型 (Dense)**：极速轻量，零样本声音克隆 (Zero-Shot Clone) 还原度高，内存占用极低。
  - **Tiny 7.9B MoE 架构**：128 路由专家 + 1 共享专家 + KDA 循环线性注意力，独家支持双人交替对话剧本 (Podcast Dialogue)、精细情感与方言控制。
- **纯 MLX 原生声学引擎**：
  - 音频合成核心与逆短时傅里叶变换 (ISTFT) 基于 `mx.fft.irfft` 与 MLX 重叠相加 (OLA) 原生实现，消除 NumPy/PyTorch 运行时依赖。
  - 音频保存基于 Python 标准库 `wave`，直接输出 44.1kHz 16-bit PCM WAV。
- **脱离 Triton 束缚**：
  - 针对原模型中依赖 Linux Triton 的 KDA (Kimi Delta Attention) 算子，在 MLX 中实现原生向量化递归更新，保持数值精度对齐的同时享受 Metal 原生硬件加速。
- **自回归状态持久缓存**：
  - 原生实现 `KDACache`（循环状态与 1D 卷积缓存）与 `MLACache`（键值缓存），杜绝 step 间失忆导致的声学崩溃与噪音。
- **全场景能力覆盖**：
  - 支持零样本声音克隆、双人对话剧本、中文/英文副语言控制（`[Laughter]` 笑声、`[Sigh]` 叹气）、官方内置音色（灵小萌、灵小锋等）、复杂电商混合文本归一化、多语种（日/韩/法/英）及四川方言调腔控制。

### 开源模型矩阵 (Hugging Face)

全部 6 个版本的纯 MLX safetensors 权重已发布至 Hugging Face，内置 Model Card 与测试音频样本：

#### 1. Ming-3.0-tts-tiny (7.9B MoE · 128 专家架构)

| 模型仓库 ID | 精度类型 | 权重体积 | 核心特性与适用场景 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-tiny-mlx-8bit`** | 8-bit 整数量化 | **9.78 GB** | **推荐主力版**：平衡极致音质与极速推理，实测 RTF ~1.68 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-4bit`** | 4-bit 极限压缩 | **5.42 GB** | **超轻量版**：显存节省 70%，适合 8GB / 16GB 统一内存设备 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-bf16`** | bfloat16 全精度 | **17.96 GB** | **无损全精度版**：100% 原始权重浮点声学表现 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-bf16) |

#### 2. Ming-3.0-tts-0.5b (0.5B 稠密模型)

| 模型仓库 ID | 精度类型 | 权重体积 | 核心特性与适用场景 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-8bit`** | 8-bit 整数量化 | **1.76 GB** | **推荐主力版**：低显存极速运行，实测 RTF ~1.06 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-4bit`** | 4-bit 极限压缩 | **1.06 GB** | **极速轻量版**：体积仅 1GB，超低内存开销 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-bf16`** | bfloat16 全精度 | **3.08 GB** | **全精度版**：高保真稠密声学基准 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-bf16) |

### 试听对比报告页面

- **在线试听网址**：**[https://vanch007.github.io/mlx-Ming-3.0-tts/](https://vanch007.github.io/mlx-Ming-3.0-tts/)**
- **单文件离线报告**：`standalone_compare.html`（40.19 MB）
- **特点**：音频全部内嵌 Base64，**无需启动本地服务器或端口**，支持中文 / 英文右上角即时一键切换，汇集 15 组全场景 6 版本同台对比。

---

## 🛠️ 项目目录结构 (Repository Structure)

```text
mlx-Ming-3.0-tts/
├── mlx_ming_tts/              # 核心纯 MLX 实现包
│   ├── models/                # 模型架构
│   │   ├── lingv3_tts.py      # Tiny 7.9B MoE 顶层调用管线与前缀注入
│   │   ├── tiny_moe.py        # 128 专家 MoE + KDA 循环注意力与状态缓存
│   │   ├── bailingmm.py       # 0.5B 稠密 Transformer 架构与调度
│   │   ├── audio_vae.py       # 纯 MLX AudioVAE 编解码器与 ISTFT 重构头
│   │   ├── flow.py            # FlowLoss、DiT、Aggregator 与 ODE 求解器
│   │   └── qwen2.py           # 带有滑动窗口注意力的 Qwen2 语言骨干
│   ├── ops/                   # 底层算子实现
│   │   ├── attention.py       # SDPA 与因果滑动窗口掩码
│   │   ├── norm.py            # RMSNorm 算子
│   │   └── rope.py            # 旋转位置编码 RoPE
│   ├── bridge/                # 跨模块桥接
│   │   ├── istft.py           # 原生快速傅里叶逆变换与重叠相加
│   │   └── speaker.py         # CampPlus 说话人向量提取桥接
│   ├── convert.py             # 0.5B 模型转换工具
│   ├── config.py              # 配置解析与模型规格
│   └── pipeline.py            # 端到端 CLI 入口与推理调度
├── demo_page/                 # 试听对比页面前端素材与 JSON 元数据
│   ├── index.html             # 试听页面
│   └── unified_same_cases_comparison.json
├── convert_tiny.py            # Tiny 7.9B 权重合并与 safetensors 转换工具
├── quantize_tiny.py           # Tiny 8-bit / 4-bit 量化工具
├── generate_bilingual_html.py # 双语试听报告编译脚本
├── standalone_compare.html    # 自包含全场景双语 6 版本离线对比报告 (40MB)
└── README.md                  # 中英文主说明文档
```

---

## 📄 开源协议与鸣谢 (License & Acknowledgement)

- 本项目遵循 [Apache-2.0 License](LICENSE) 开源协议。
- 感谢 [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) 官方团队开源的高质量 TTS 模型权重与算法设计。
- 感谢 [Apple MLX](https://github.com/ml-explore/mlx) 团队带来的高性能端侧深度学习框架。