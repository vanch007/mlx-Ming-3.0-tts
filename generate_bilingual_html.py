import json, os, base64

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/unified_same_cases_comparison.json'
OUT_HTML = '/Users/vanch/mlx-Ming-3.0-tts/standalone_compare.html'
OUT_HTML_DEMO = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/index.html'

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

def to_b64(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, 'rb') as f:
        return f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('ascii')}"

# 双语对照字典
I18N_CASES = {
    "case01_clone_zh": {
        "title_zh": "CASE01_CLONE_ZH · 零样本声音克隆 (Zero-Shot Clone: 灵光)",
        "title_en": "CASE01_CLONE_ZH · Zero-Shot Voice Cloning (Prompt: Lingguang)",
        "desc_zh": "基于灵光参考音频克隆全新文本，声线特征自适应对齐",
        "desc_en": "Adaptively clone acoustic features from Lingguang reference audio onto unseen text"
    },
    "case02_paralinguistic_zh": {
        "title_zh": "CASE02_PARALINGUISTIC_ZH · 中文副语言控制 ([Laughter] 行内笑声)",
        "title_en": "CASE02_PARALINGUISTIC_ZH · Chinese Paralinguistic Control ([Laughter])",
        "desc_zh": "行内笑声标签控制，原样透传不被文本归一化破坏",
        "desc_en": "Inline laughter tag synthesis without disruption from frontend text normalization"
    },
    "case03_paralinguistic_en": {
        "title_zh": "CASE03_PARALINGUISTIC_EN · 英文副语言控制 ([Sigh] 叹气 + [Uhm] 迟疑)",
        "title_en": "CASE03_PARALINGUISTIC_EN · English Paralinguistic Control ([Sigh] + [Uhm])",
        "desc_zh": "英文行内复合副语言标签控制，叹气与语气助词平滑衔接",
        "desc_en": "Composite inline tags smoothly connecting sighed breathing with hesitation cues"
    },
    "case04_multilingual_ja": {
        "title_zh": "CASE04_MULTILINGUAL_JA · 官方多语言：日语 (Japanese)",
        "title_en": "CASE04_MULTILINGUAL_JA · Multilingual Synthesis: Japanese",
        "desc_zh": "指定日语目标语种，绕过汉字归一化保持平假名/片假名纯正发音",
        "desc_en": "Target Japanese language acoustic guidance preserving natural Hiragana/Katakana prosody"
    },
    "case05_multilingual_ko": {
        "title_zh": "CASE05_MULTILINGUAL_KO · 官方多语言：韩语 (Korean)",
        "title_en": "CASE05_MULTILINGUAL_KO · Multilingual Synthesis: Korean",
        "desc_zh": "官方支持多语种测试：韩语音韵结构与声学合成",
        "desc_en": "Korean syllable block phonetics and cross-lingual acoustic transfer"
    },
    "case06_multilingual_fr": {
        "title_zh": "CASE06_MULTILINGUAL_FR · 官方多语言：法语 (French)",
        "title_en": "CASE06_MULTILINGUAL_FR · Multilingual Synthesis: French",
        "desc_zh": "官方支持多语种测试：法语连音与欧系语言声学合成",
        "desc_en": "French liaison and European acoustic prosody synthesis"
    },
    "case07_freeform_en": {
        "title_zh": "CASE07_FREEFORM_EN · 英文自由描述指令 (Free-Form Instruction)",
        "title_en": "CASE07_FREEFORM_EN · English Free-Form Instruction",
        "desc_zh": "纯自然语言指令定制：温暖、怀旧、舒缓亲切的说书人叙事质感",
        "desc_en": "Natural language direction: warm, intimate storytelling voice with a gentle nostalgic pace"
    },
    "case08_voice_design": {
        "title_zh": "CASE08_VOICE_DESIGN · 自然语言音色设计 (Voice Design)",
        "title_en": "CASE08_VOICE_DESIGN · Natural Language Voice Design",
        "desc_zh": "纯自然语言定制：成熟沉稳的纪录片男声旁白，吐字舒缓清晰",
        "desc_en": "Voice prompt direction: mature, resonant male documentary narration with clear pacing"
    },
    "case09_basic_fast": {
        "title_zh": "CASE09_BASIC_FAST · 基础属性控制 (语速: 快速)",
        "title_en": "CASE09_BASIC_FAST · Basic Attribute Control (Speed: Fast)",
        "desc_zh": "显式基础声学控制：设置快速语速、中音高与中音量",
        "desc_en": "Explicit acoustic control: speed fast, pitch medium, volume medium"
    },
    "case10_emotion_sad": {
        "title_zh": "CASE10_EMOTION_SAD · 情感控制：悲伤 (Emotion: Sad)",
        "title_en": "CASE10_EMOTION_SAD · Emotion Control: Sad",
        "desc_zh": "跨情感控制，使用中性提示生成深沉悲伤、低落的语音色彩",
        "desc_en": "Restrained, subdued, and poignant sad emotional vocal delivery"
    },
    "case11_dialect_sichuan": {
        "title_zh": "CASE11_DIALECT_SICHUAN · 方言控制：四川话 (Sichuanese Dialect)",
        "title_en": "CASE11_DIALECT_SICHUAN · Dialect Control: Sichuanese",
        "desc_zh": "地道川渝方言调腔控制，发音纯正地道",
        "desc_en": "Authentic Sichuan-Chongqing dialect intonation and vernacular expressions"
    },
    "case12_builtin_ip_us_james": {
        "title_zh": "CASE12_BUILTIN_IP_US_JAMES · 官方推荐英文内置音色 (US_James)",
        "title_en": "CASE12_BUILTIN_IP_US_JAMES · Official Built-in Voice: US_James",
        "desc_zh": "官方推荐 5 大英文高品质原生音色之一，纯正美式成熟男声",
        "desc_en": "Official top English voice preset: mature American male broadcaster delivery"
    },
    "case13_builtin_ip_lingxiaomeng": {
        "title_zh": "CASE13_BUILTIN_IP_LINGXIAOMENG · 官方推荐中文内置音色 (灵小萌)",
        "title_en": "CASE13_BUILTIN_IP_LINGXIAOMENG · Official Built-in Voice: Lingxiaomeng",
        "desc_zh": "官方推荐音色，治愈系电台主播女声，吐字干练清脆",
        "desc_en": "Official Chinese voice preset: soothing female radio host with crisp delivery"
    },
    "case14_text_norm": {
        "title_zh": "CASE14_TEXT_NORM · 复杂电商混合文本归一化 (Text Normalization)",
        "title_en": "CASE14_TEXT_NORM · Complex Ecommerce Text Normalization",
        "desc_zh": "订单号、日期、货币符号混合自动扩展与精准朗读",
        "desc_en": "Automatic spoken expansion of order IDs, dates, currencies, and numerals"
    },
    "case15_podcast": {
        "title_zh": "CASE15_PODCAST · 双人对话剧本 (Podcast Dialogue - Tiny独家)",
        "title_en": "CASE15_PODCAST · Multi-Speaker Podcast Dialogue (Tiny Exclusive)",
        "desc_zh": "7.9B MoE 独家双人角色交替对话剧本（0.5b 不支持此任务）",
        "desc_en": "7.9B MoE exclusive alternating multi-speaker role-play script (0.5b does not support)"
    }
}

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ming-3.0-tts (0.5B Dense vs Tiny 7.9B MoE) 6-Version Full Comparison Platform</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --success: #34d399;
      --warning: #fbbf24;
      --purple: #a855f7;
      --cyan: #06b6d4;
      --indigo: #6366f1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 40px 20px; line-height: 1.6; }
    .container { max-width: 1440px; margin: 0 auto; }
    
    header { text-align: center; margin-bottom: 36px; position: relative; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 12px; background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    /* 语言切换器按钮 */
    .lang-switcher {
      position: absolute;
      top: 0;
      right: 0;
      display: flex;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      border-radius: 30px;
      padding: 4px;
      backdrop-filter: blur(10px);
    }
    .lang-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 20px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.2s ease;
    }
    .lang-btn.active {
      background: var(--primary);
      color: #0b0f19;
      box-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
    }
    
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 36px; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; text-align: center; }
    .stat-title { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 1.4rem; font-weight: 700; }
    .stat-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
    
    .case-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; margin-bottom: 32px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .case-header { padding: 16px 24px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    .case-title { font-size: 1.15rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge.tiny { background: rgba(168, 85, 247, 0.15); color: var(--purple); border-color: rgba(168, 85, 247, 0.3); }
    .case-body { padding: 20px 24px; }
    .case-desc { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 12px; }
    .case-text { background: rgba(0,0,0,0.25); border-left: 3px solid var(--primary); padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 0.95rem; color: #cbd5e1; margin-bottom: 24px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    
    /* 两大模型组标题 */
    .model-group-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #e2e8f0;
      margin: 18px 0 12px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .player-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
    @media (max-width: 1024px) {
      .player-grid { grid-template-columns: 1fr; }
    }
    .player-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; gap: 12px; transition: border-color 0.2s, transform 0.2s; }
    .player-card:hover { border-color: rgba(56, 189, 248, 0.4); transform: translateY(-2px); }
    
    .player-card.dense-bf16 { border-top: 3px solid var(--primary); }
    .player-card.dense-int8 { border-top: 3px solid var(--cyan); }
    .player-card.dense-int4 { border-top: 3px solid var(--warning); }
    .player-card.tiny-bf16 { border-top: 3px solid var(--indigo); }
    .player-card.tiny-int8 { border-top: 3px solid var(--success); }
    .player-card.tiny-int4 { border-top: 3px solid var(--warning); }
    .player-card.ref-card { border-top: 3px solid var(--purple); background: rgba(168, 85, 247, 0.05); }
    
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .player-name { font-size: 0.95rem; font-weight: 600; color: #f1f5f9; }
    .meta-tag { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.08); color: var(--text-muted); }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; }
    .metric-val { color: var(--text); font-weight: 600; }
    audio { width: 100%; height: 36px; border-radius: 6px; outline: none; }
    
    footer { text-align: center; margin-top: 60px; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--card-border); padding-top: 24px; }
    
    /* 语言切换控制显示 */
    body.lang-zh .i18n-en { display: none !important; }
    body.lang-en .i18n-zh { display: none !important; }
  </style>
</head>
<body class="lang-zh">
  <div class="container">
    <header>
      <div class="lang-switcher">
        <button id="btn-zh" class="lang-btn active" onclick="setLang('zh')">🇨🇳 中文</button>
        <button id="btn-en" class="lang-btn" onclick="setLang('en')">🇬🇧 English</button>
      </div>
      <h1>
        <span class="i18n-zh">Ming-3.0-tts (0.5B 稠密 vs Tiny 7.9B MoE) 6个版本全精度同台对比</span>
        <span class="i18n-en">Ming-3.0-tts (0.5B Dense vs Tiny 7.9B MoE) 6-Version Full Comparison</span>
      </h1>
      <p class="subtitle">
        <span class="i18n-zh">纯 Apple Silicon MLX 原生算子重构 · 0.5B (bf16/8bit/4bit) vs Tiny 7.9B (bf16/8bit/4bit) 同题同台离线试听</span>
        <span class="i18n-en">Pure Apple Silicon MLX Native Port · 0.5B (bf16/8bit/4bit) vs Tiny 7.9B (bf16/8bit/4bit) Side-by-Side Audio Evaluation</span>
      </p>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">
          <span class="i18n-zh">评测用例总数</span>
          <span class="i18n-en">Total Test Cases</span>
        </div>
        <div class="stat-value" style="color: var(--primary)">15</div>
        <div class="stat-sub">
          <span class="i18n-zh">全场景能力覆盖对比</span>
          <span class="i18n-en">All-Scenario Coverage</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">
          <span class="i18n-zh">模型版本同台</span>
          <span class="i18n-en">Model Precision Variants</span>
        </div>
        <div class="stat-value" style="color: var(--purple)">6</div>
        <div class="stat-sub">
          <span class="i18n-zh">0.5B + Tiny 7.9B 全精度</span>
          <span class="i18n-en">0.5B + Tiny 7.9B Full Precisions</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">
          <span class="i18n-zh">端侧硬件加速</span>
          <span class="i18n-en">Hardware Acceleration</span>
        </div>
        <div class="stat-value" style="color: var(--success)">100% MLX</div>
        <div class="stat-sub">
          <span class="i18n-zh">Apple Silicon Metal 全程加速</span>
          <span class="i18n-en">Apple Silicon Metal Native GPU</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">
          <span class="i18n-zh">量化与推理比率</span>
          <span class="i18n-en">Quantization & Speed</span>
        </div>
        <div class="stat-value" style="color: var(--warning)">8-bit & 4-bit</div>
        <div class="stat-sub">
          <span class="i18n-zh">最高提速 2.1x · 显存节省 70%</span>
          <span class="i18n-en">Up to 2.1x Faster · 70% RAM Saved</span>
        </div>
      </div>
    </div>
'''

for idx, c in enumerate(data):
    cid = c['case_id']
    info = I18N_CASES.get(cid, {
        "title_zh": c.get('category', cid),
        "title_en": cid.upper(),
        "desc_zh": c.get('desc', ''),
        "desc_en": c.get('desc', '')
    })
    
    text = c['text']
    ref_b64 = to_b64(c.get('ref_audio'))
    
    d_dict = c.get('dense_0_5b', {})
    d_int8 = d_dict.get('int8')
    d_bf16 = d_dict.get('bf16')
    d_int4 = d_dict.get('int4')
    
    t_dict = c.get('tiny_moe', {})
    t_int8 = t_dict.get('tiny_8bit')
    t_bf16 = t_dict.get('tiny_bf16')
    t_int4 = t_dict.get('tiny_4bit')
    
    html += f'''
    <div class="case-card">
      <div class="case-header">
        <div class="case-title">
          <span class="i18n-zh">{info["title_zh"]}</span>
          <span class="i18n-en">{info["title_en"]}</span>
        </div>
        <span class="badge">
          <span class="i18n-zh">同题同台 6 个模型版本对比</span>
          <span class="i18n-en">6 Models Side-by-Side</span>
        </span>
      </div>
      <div class="case-body">
        <div class="case-desc">
          <span class="i18n-zh">{info["desc_zh"]}</span>
          <span class="i18n-en">{info["desc_en"]}</span>
        </div>
        <div class="case-text">
          <strong><span class="i18n-zh">同一测试输入文本:</span><span class="i18n-en">Identical Input Text:</span></strong> {text}
        </div>
        
        <!-- 参考基准 (如有) -->
        {f"""
        <div style="max-width: 460px; margin-bottom: 20px;">
          <div class="player-card ref-card">
            <div>
              <div class="player-head">
                <span class="player-name"><span class="i18n-zh">官方参考原声 (Prompt Audio)</span><span class="i18n-en">Official Prompt Audio</span></span>
                <span class="meta-tag"><span class="i18n-zh">基准原声</span><span class="i18n-en">Reference</span></span>
              </div>
            </div>
            <audio controls preload="none" src="{ref_b64}"></audio>
          </div>
        </div>
        """ if ref_b64 else ""}

        <!-- 第一组：0.5B 稠密模型 (3个版本) -->
        <div class="model-group-title">
          <span>🔹 <span class="i18n-zh">0.5B 稠密模型 (Dense Architecture) · 3个版本</span><span class="i18n-en">0.5B Dense Architecture · 3 Precisions</span></span>
        </div>
        <div class="player-grid">
          
          <!-- 0.5B bf16 -->
          <div class="player-card dense-bf16">
            <div>
              <div class="player-head">
                <span class="player-name">0.5B bf16 (<span class="i18n-zh">全精度</span><span class="i18n-en">Full Precision</span>)</span>
                <span class="meta-tag">{f"{d_bf16['weight_size_mb']/1024:.2f} GB" if d_bf16 else "3.08 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val">{f"{d_bf16['elapsed_s']}s" if d_bf16 else "-"}</span></div>
                <div>RTF: <span class="metric-val">{d_bf16['rtf'] if d_bf16 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{d_bf16['duration_s']}s" if d_bf16 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{d_bf16['b64']}'></audio>" if d_bf16 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>0.5b 不支持多人对话</span><span class='i18n-en'>0.5b does not support multi-speaker</span></div>"}
          </div>

          <!-- 0.5B 8-bit -->
          <div class="player-card dense-int8">
            <div>
              <div class="player-head">
                <span class="player-name">0.5B 8-bit (<span class="i18n-zh">推荐主力</span><span class="i18n-en">Recommended</span>)</span>
                <span class="meta-tag">{f"{d_int8['weight_size_mb']/1024:.2f} GB" if d_int8 else "1.76 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val">{f"{d_int8['elapsed_s']}s" if d_int8 else "-"}</span></div>
                <div>RTF: <span class="metric-val">{d_int8['rtf'] if d_int8 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{d_int8['duration_s']}s" if d_int8 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{d_int8['b64']}'></audio>" if d_int8 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>0.5b 不支持多人对话</span><span class='i18n-en'>0.5b does not support multi-speaker</span></div>"}
          </div>

          <!-- 0.5B 4-bit -->
          <div class="player-card dense-int4">
            <div>
              <div class="player-head">
                <span class="player-name">0.5B 4-bit (<span class="i18n-zh">超轻量</span><span class="i18n-en">Ultra-Light</span>)</span>
                <span class="meta-tag">{f"{d_int4['weight_size_mb']/1024:.2f} GB" if d_int4 else "1.06 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val">{f"{d_int4['elapsed_s']}s" if d_int4 else "-"}</span></div>
                <div>RTF: <span class="metric-val">{d_int4['rtf'] if d_int4 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{d_int4['duration_s']}s" if d_int4 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{d_int4['b64']}'></audio>" if d_int4 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>0.5b 不支持多人对话</span><span class='i18n-en'>0.5b does not support multi-speaker</span></div>"}
          </div>

        </div>

        <!-- 第二组：Tiny 7.9B MoE 模型 (3个版本) -->
        <div class="model-group-title">
          <span>🔶 <span class="i18n-zh">Tiny 7.9B MoE 模型 (128 Experts + KDA) · 3个版本</span><span class="i18n-en">Tiny 7.9B MoE Architecture (128 Experts + KDA) · 3 Precisions</span></span>
        </div>
        <div class="player-grid">
          
          <!-- Tiny bf16 -->
          <div class="player-card tiny-bf16">
            <div>
              <div class="player-head">
                <span class="player-name">Tiny 7.9B bf16 (<span class="i18n-zh">全精度</span><span class="i18n-en">Full Precision</span>)</span>
                <span class="meta-tag">{f"{t_bf16['weight_size_mb']/1024:.2f} GB" if t_bf16 else "17.96 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val">{f"{t_bf16['elapsed_s']}s" if t_bf16 else "-"}</span></div>
                <div>RTF: <span class="metric-val">{t_bf16['rtf'] if t_bf16 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{t_bf16['duration_s']}s" if t_bf16 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{t_bf16['b64']}'></audio>" if t_bf16 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>生成失败</span><span class='i18n-en'>Failed</span></div>"}
          </div>

          <!-- Tiny 8-bit -->
          <div class="player-card tiny-int8">
            <div>
              <div class="player-head">
                <span class="player-name" style="color:var(--success)">Tiny 7.9B 8-bit (<span class="i18n-zh">推荐主力</span><span class="i18n-en">Recommended</span>)</span>
                <span class="meta-tag" style="background: rgba(52, 211, 153, 0.2); color: var(--success)">{f"{t_int8['weight_size_mb']/1024:.2f} GB" if t_int8 else "9.78 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val" style="color:var(--success)">{f"{t_int8['elapsed_s']}s" if t_int8 else "-"}</span></div>
                <div>RTF: <span class="metric-val" style="color:var(--success)">{t_int8['rtf'] if t_int8 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{t_int8['duration_s']}s" if t_int8 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{t_int8['b64']}'></audio>" if t_int8 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>生成失败</span><span class='i18n-en'>Failed</span></div>"}
          </div>

          <!-- Tiny 4-bit -->
          <div class="player-card tiny-int4">
            <div>
              <div class="player-head">
                <span class="player-name">Tiny 7.9B 4-bit (<span class="i18n-zh">超轻量</span><span class="i18n-en">Ultra-Light</span>)</span>
                <span class="meta-tag">{f"{t_int4['weight_size_mb']/1024:.2f} GB" if t_int4 else "5.42 GB"}</span>
              </div>
              <div class="metrics">
                <div><span class="i18n-zh">耗时</span><span class="i18n-en">Elapsed</span>: <span class="metric-val">{f"{t_int4['elapsed_s']}s" if t_int4 else "-"}</span></div>
                <div>RTF: <span class="metric-val">{t_int4['rtf'] if t_int4 else "-"}</span></div>
                <div><span class="i18n-zh">时长</span><span class="i18n-en">Duration</span>: <span class="metric-val">{f"{t_int4['duration_s']}s" if t_int4 else "-"}</span></div>
                <div><span class="i18n-zh">采样率</span><span class="i18n-en">Rate</span>: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {f"<audio controls preload='none' src='{t_int4['b64']}'></audio>" if t_int4 else "<div style='color:var(--text-muted);font-size:0.85rem;padding:8px 0;'><span class='i18n-zh'>生成失败</span><span class='i18n-en'>Failed</span></div>"}
          </div>

        </div>

      </div>
    </div>
'''

html += '''
    <footer>
      <p>
        <span class="i18n-zh">mlx-Ming-3.0-tts · 纯 Apple Silicon MLX 架构移植 · 蚂蚁百灵语音大模型开源社区对齐</span>
        <span class="i18n-en">mlx-Ming-3.0-tts · Pure Apple Silicon MLX Native Port · Aligned with Ant Group inclusionAI Ming-3.0-tts</span>
      </p>
      <p style="margin-top: 6px; opacity: 0.7;">
        <span class="i18n-zh">所有音频均在本地 Apple Silicon (M3 Max) 统一硬件环境实测生成并直接内嵌 Base64 交付</span>
        <span class="i18n-en">All audio samples benchmarked locally on Apple Silicon (M3 Max) with embedded Base64 waveforms</span>
      </p>
    </footer>
  </div>

  <script>
    function setLang(lang) {
      if (lang === 'en') {
        document.body.className = 'lang-en';
        document.getElementById('btn-en').classList.add('active');
        document.getElementById('btn-zh').classList.remove('active');
        localStorage.setItem('mlx_tts_lang', 'en');
      } else {
        document.body.className = 'lang-zh';
        document.getElementById('btn-zh').classList.add('active');
        document.getElementById('btn-en').classList.remove('active');
        localStorage.setItem('mlx_tts_lang', 'zh');
      }
    }

    // 初始化语言偏好
    (function() {
      var saved = localStorage.getItem('mlx_tts_lang');
      if (saved === 'en') {
        setLang('en');
      } else {
        setLang('zh');
      }
    })();
  </script>
</body>
</html>
'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

with open(OUT_HTML_DEMO, 'w', encoding='utf-8') as f:
    f.write(html)

print("Generated complete bilingual comparison page:")
print(f"  -> {OUT_HTML} ({len(html)/1024/1024:.2f} MB)")
print(f"  -> {OUT_HTML_DEMO} ({len(html)/1024/1024:.2f} MB)")

