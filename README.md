# mlx-Ming-3.0-tts

Pure Apple Silicon **MLX** native port of [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) — supporting both the **0.5B Dense** and **Tiny 7.9B MoE (128 Experts + KDA)** models across multiple precisions (bf16 / 8-bit / 4-bit).

Completely free of Linux-only `triton` kernels and PyTorch/NumPy vocoder runtime dependencies, running natively on Apple Silicon Metal GPUs (M1/M2/M3/M4).

---

## 🌟 核心特性 (Key Features)

- **双模型双架构支持 (Dual Architecture Support)**：
  - **0.5B Dense Architecture**：轻量高速，零样本声音克隆 (Zero-Shot Clone) 还原度高，极低显存占用。
  - **Tiny 7.9B MoE Architecture**：128 路由专家 + 1 共享专家 + KDA 循环线性注意力，独家支持双人交替对话剧本 (Podcast Dialogue)、精细情感与方言控制。
- **纯 MLX 原生声学引擎 (Pure MLX Native Engine)**：
  - 音频合成核心与逆短时傅里叶变换 (ISTFT) 基于 `mx.fft.irfft` 与 MLX 重叠相加 (OLA) 原生实现，消除 NumPy/PyTorch 运行时依赖。
  - 音频保存基于 Python 标准库 `wave`，直接输出 44.1kHz 16-bit PCM WAV。
- **脱离 Triton 束缚 (Zero Triton Dependency)**：
  - 针对原模型中依赖 Linux Triton 的 KDA (Kimi Delta Attention) 算子，在 MLX 中实现原生向量化递归更新，保持数值精度对齐的同时享受 Metal 原生硬件加速。
- **自回归状态持久缓存 (Persistent Step Caching)**：
  - 原生实现 `KDACache`（循环状态与 1D 卷积缓存）与 `MLACache`（键值缓存），杜绝 step 间失忆导致的声学崩溃与噪音。
- **全场景能力覆盖 (Full Capabilities)**：
  - 支持零样本声音克隆、双人对话剧本、中文/英文副语言控制（`[Laughter]` 笑声、`[Sigh]` 叹气）、官方内置音色（灵小萌、灵小锋等）、复杂电商混合文本归一化、多语种（日/韩/法/英）及四川方言调腔控制。

---

## 📦 Hugging Face 开源模型矩阵 (Model Hub)

已将全部 6 个版本的纯 MLX safetensors 权重发布至 Hugging Face，每个仓库均内置 Model Card 与测试音频样本：

### 1. Ming-3.0-tts-tiny (7.9B MoE · 128 Experts + KDA)

| 模型仓库 ID | 精度类型 | 权重体积 | 核心特性与适用场景 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-tiny-mlx-8bit`** | 8-bit 整数量化 | **9.78 GB** | **推荐主力版**：平衡极致音质与极速推理，实测 RTF ~1.68 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-4bit`** | 4-bit 极限压缩 | **5.42 GB** | **超轻量版**：显存节省 70%，适合 8GB / 16GB 统一内存设备 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-tiny-mlx-bf16`** | bfloat16 全精度 | **17.96 GB** | **无损全精度版**：100% 原始权重浮点声学表现 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-tiny-mlx-bf16) |

### 2. Ming-3.0-tts-0.5b (0.5B Dense Architecture)

| 模型仓库 ID | 精度类型 | 权重体积 | 核心特性与适用场景 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-8bit`** | 8-bit 整数量化 | **1.76 GB** | **推荐主力版**：低显存极速运行，实测 RTF ~1.06 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-8bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-4bit`** | 4-bit 极限压缩 | **1.06 GB** | **极速轻量版**：体积仅 1GB，超低内存开销 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-4bit) |
| **`vanch007/Ming-3.0-tts-0.5b-mlx-bf16`** | bfloat16 全精度 | **3.08 GB** | **全精度版**：高保真稠密声学基准 | [🤗 访问仓库](https://huggingface.co/vanch007/Ming-3.0-tts-0.5b-mlx-bf16) |

---

## 🎧 离线单文件试听对比报告 (Offline Demo Page)

本项目交付了完全自包含的离线单文件对比页面：

- **文件路径**：`standalone_compare.html`（或 `demo_page/index.html`）
- **特点**：所有测试音频均通过 Base64 内嵌在单个 HTML 文件内，**无需启动任何本地 HTTP 服务或端口**，在 Mac 上直接双击或拖入任意浏览器（Chrome / Safari / Edge）即可播放。
- **对比维度**：汇集了 15 组全场景同题同台对比卡片（基准原声 vs 0.5B 三精度 vs Tiny 7.9B 三精度，共 6 个版本同屏试听对比）。

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备

```bash
git clone https://github.com/vanch007/mlx-Ming-3.0-tts.git
cd mlx-Ming-3.0-tts
pip install mlx transformers soundfile onnxruntime
```

### 2. 下载推荐主力模型

```bash
# 下载 Tiny 7.9B MoE 8-bit 主力模型
huggingface-cli download vanch007/Ming-3.0-tts-tiny-mlx-8bit --local-dir models/Ming-3.0-tts-tiny-mlx-8bit

# 下载 0.5B 8-bit 主力模型
huggingface-cli download vanch007/Ming-3.0-tts-0.5b-mlx-8bit --local-dir models/Ming-3.0-tts-0.5b-mlx-8bit
```

### 3. CLI 快速推理

**① 官方推荐内置音色 (灵小萌)**：
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-tiny-mlx-8bit \
  --text "夜晚的微风吹散了白天的喧嚣，愿好梦常伴你左右。" \
  --ip "灵小萌" \
  --out "output_lingxiaomeng.wav"
```

**② 独家双人角色交替对话剧本 (Podcast Dialogue)**：
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-tiny-mlx-8bit \
  --text "speaker_1: 你最近是不是又去那家面馆了？\nspeaker_2: 对啊，排了半个小时的队，不过真的值得。" \
  --out "output_podcast.wav"
```

**③ 零样本声音克隆 (Zero-Shot Voice Cloning)**：
```bash
python3 -m mlx_ming_tts.pipeline \
  --model-dir models/Ming-3.0-tts-0.5b-mlx-8bit \
  --ref-audio "path/to/reference.wav" \
  --prompt-text "参考音频中说话人念出的准确文本内容" \
  --text "需要克隆该说话人声线来朗读的新文本" \
  --out "output_cloned.wav"
```

### 4. Python API 调用

```python
from mlx_ming_tts.pipeline import MingMLX, save_wav, DEFAULT_TTS_PROMPT

# 加载 MLX 原生模型引擎
runner = MingMLX("models/Ming-3.0-tts-tiny-mlx-8bit")

# 合成语音
wav = runner.generate(
    DEFAULT_TTS_PROMPT,
    text="欢迎使用纯 MLX 原生移植的 Ming 语音大模型。",
    ip="灵小萌",
    sigma=0.15,
    cfg=2.0,
    noise_seed=42
)

# 导出标准 44.1kHz 16-bit PCM WAV
save_wav(wav, "output.wav", runner.sample_rate)
```

---

## 🛠️ 项目结构 (Repository Structure)

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
├── build_unified_html.py      # 离线单文件试听报告编译脚本
├── standalone_compare.html    # 自包含全场景 6 版本同台离线对比报告 (40MB)
└── README.md                  # 项目主文档
```

---

## 📄 开源许可与鸣谢 (License & Acknowledgement)

- 本项目遵循 [Apache-2.0 License](LICENSE) 开源协议。
- 感谢 [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) 官方团队开源的高质量 TTS 模型权重与算法设计。
- 感谢 [Apple MLX](https://github.com/ml-explore/mlx) 团队带来的高性能端侧深度学习框架。