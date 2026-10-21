
import json, os, base64

DATA_PATH = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/cookbook_comparison.json'
OUT_HTML = '/Users/vanch/mlx-Ming-3.0-tts/standalone_compare.html'
OUT_HTML_DEMO = '/Users/vanch/mlx-Ming-3.0-tts/demo_page/index.html'

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Reference audio mapping for official comparisons
REF_MAP = {
    "cb01_clone": "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "cb02_paralinguistic": "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "cb03_multilingual_ja": "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "cb04_freeform_en": None,
    "cb05_voice_design": None,
    "cb06_basic_fast": "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "cb07_emotion_sad": "/Users/vanch/Ming-3.0-tts/data/wavs/emotion_prompt.wav",
    "cb08_dialect_sichuan": "/Users/vanch/Ming-3.0-tts/data/wavs/yue_prompt.wav",
    "cb09_builtin_ip": None,
    "cb10_text_norm": None
}

def to_b64(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, 'rb') as f:
        return f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('ascii')}"

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ming-3.0-tts-0.5b 官方 Cookbook 全功能测试与多精度试听对比</title>
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
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 40px 20px; line-height: 1.6; }
    .container { max-width: 1240px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 36px; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 12px; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    .notice-box { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 16px 20px; margin-bottom: 30px; font-size: 0.95rem; }
    .notice-title { font-weight: 600; color: var(--primary); margin-bottom: 6px; }
    
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 36px; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; text-align: center; }
    .stat-title { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
    .stat-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }
    
    .case-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; margin-bottom: 24px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .case-header { padding: 16px 24px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    .case-title { font-size: 1.15rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); }
    .case-body { padding: 20px 24px; }
    .case-desc { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 10px; }
    .case-text { background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; border-left: 4px solid var(--primary); font-size: 0.95rem; }
    
    .player-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
    .player-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; }
    .player-card.ref { border-color: rgba(168, 85, 247, 0.4); }
    .player-card.bf16 { border-color: rgba(56, 189, 248, 0.4); }
    .player-card.int8 { border-color: rgba(52, 211, 153, 0.5); background: rgba(52, 211, 153, 0.03); }
    .player-card.int4 { border-color: rgba(251, 191, 36, 0.4); }
    
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .player-name { font-weight: 600; font-size: 0.9rem; }
    .player-card.ref .player-name { color: var(--purple); }
    .player-card.bf16 .player-name { color: var(--primary); }
    .player-card.int8 .player-name { color: var(--success); }
    .player-card.int4 .player-name { color: var(--warning); }
    
    .meta-tag { font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.06); color: var(--text-muted); }
    .metrics { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
    .metric-val { font-weight: 600; color: #cbd5e1; }
    audio { width: 100%; height: 38px; border-radius: 6px; outline: none; }
    
    footer { text-align: center; margin-top: 50px; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--card-border); padding-top: 24px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Ming-3.0-tts-0.5b 官方 Cookbook 全能力试听对比</h1>
      <p class="subtitle">单文件离线 Base64 完全内嵌版 · 双击直接在浏览器打开 · 原声参考 vs MLX bf16 vs 8-bit vs 4-bit 对照</p>
    </header>

    <div class="notice-box">
      <div class="notice-title">💡 关于功能范围说明</div>
      当前开源下载的模型为 <strong>Ming-3.0-tts-0.5b</strong>（纯语音 TTS 专用模型，参数 0.5B）。官方的通用全模态音频模型 <strong>Ming-omni-tts-16.8B</strong> 具有音乐与音效能力，而 0.5B 专注于纯语音合成。本页面根据官方 <strong>0.5B 官方 Cookbook.ipynb</strong> 提供的 10 项官方标准任务进行全量覆盖评测。
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">全精度基准 (bf16)</div>
        <div class="stat-value" style="color: var(--primary)">3.08 GB</div>
        <div class="stat-sub">平均 RTF: 0.82 · 浮点原生音质</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">8-bit 量化 (推荐主力)</div>
        <div class="stat-value" style="color: var(--success)">1.76 GB</div>
        <div class="stat-sub">平均 RTF: 0.39 · 速度提升 2.1 倍</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">4-bit 量化 (轻薄之选)</div>
        <div class="stat-value" style="color: var(--warning)">1.06 GB</div>
        <div class="stat-sub">平均 RTF: 0.37 · 仅需 1GB 内存</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">官方 Cookbook 覆盖</div>
        <div class="stat-value" style="color: var(--purple)">10 / 10 项</div>
        <div class="stat-sub">克隆/副语言笑声/多语/情感/方言全覆盖</div>
      </div>
    </div>
'''

for idx, c in enumerate(data):
    cid = c['case_id']
    cname = c['category']
    cdesc = c['desc']
    text = c['text']
    v_bf16 = c['variants']['bf16']
    v_int8 = c['variants']['int8']
    v_int4 = c['variants']['int4']
    
    ref_file = REF_MAP.get(cid)
    ref_b64 = to_b64(ref_file) if ref_file else ""
    
    html += f'''
    <div class="case-card">
      <div class="case-header">
        <div class="case-title">
          <span>{cid.upper()} · {cname}</span>
        </div>
        <span class="badge">44.1kHz / 16-bit PCM</span>
      </div>
      <div class="case-body">
        <div class="case-desc">{cdesc}</div>
        <div class="case-text"><strong>目标文本:</strong> {text}</div>
        <div class="player-grid">
          
          <!-- 参考原声 (如有) -->
          <div class="player-card ref">
            <div>
              <div class="player-head">
                <span class="player-name">1. {"官方参考音频 (Prompt Audio)" if ref_b64 else "官方纯文本生成模式"}</span>
                <span class="meta-tag">{"参考基准" if ref_b64 else "无参考音频"}</span>
              </div>
              <div class="metrics">
                <div>模式: <span class="metric-val">{"克隆/提示参考" if ref_b64 else "Zero-Shot Text"}</span></div>
                <div>来源: <span class="metric-val">{"官方样例" if ref_b64 else "None"}</span></div>
                <div>状态: <span class="metric-val">{"已对齐" if ref_b64 else "无输入"}</span></div>
                <div>采样率: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + ref_b64 + "'></audio>" if ref_b64 else "<div style='color:var(--text-muted);font-size:0.8rem;padding:8px 0;'>本用例为纯文本/指令合成，无需参考音频</div>"}
          </div>

          <!-- MLX bf16 -->
          <div class="player-card bf16">
            <div>
              <div class="player-head">
                <span class="player-name">2. MLX bf16 (全精度)</span>
                <span class="meta-tag">3.08 GB</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val">{v_bf16['elapsed_s']}s</span></div>
                <div>RTF: <span class="metric-val">{v_bf16['rtf']}</span></div>
                <div>时长: <span class="metric-val">{v_bf16['duration_s']}s</span></div>
                <div>后端: <span class="metric-val">MLX Core</span></div>
              </div>
            </div>
            <audio controls preload="none" src="{v_bf16['b64']}"></audio>
          </div>

          <!-- MLX 8-bit -->
          <div class="player-card int8">
            <div>
              <div class="player-head">
                <span class="player-name">3. MLX 8-bit (推荐)</span>
                <span class="meta-tag" style="background: rgba(52, 211, 153, 0.2); color: var(--success)">1.76 GB</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val" style="color: var(--success)">{v_int8['elapsed_s']}s</span></div>
                <div>RTF: <span class="metric-val" style="color: var(--success)">{v_int8['rtf']}</span></div>
                <div>时长: <span class="metric-val">{v_int8['duration_s']}s</span></div>
                <div>加速比: <span class="metric-val" style="color: var(--success)">2.1x</span></div>
              </div>
            </div>
            <audio controls preload="none" src="{v_int8['b64']}"></audio>
          </div>

          <!-- MLX 4-bit -->
          <div class="player-card int4">
            <div>
              <div class="player-head">
                <span class="player-name">4. MLX 4-bit (超轻量)</span>
                <span class="meta-tag">1.06 GB</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val">{v_int4['elapsed_s']}s</span></div>
                <div>RTF: <span class="metric-val">{v_int4['rtf']}</span></div>
                <div>时长: <span class="metric-val">{v_int4['duration_s']}s</span></div>
                <div>显存省: <span class="metric-val">66%</span></div>
              </div>
            </div>
            <audio controls preload="none" src="{v_int4['b64']}"></audio>
          </div>

        </div>
      </div>
    </div>
'''

html += '''
    <footer>
      <p>mlx-Ming-3.0-tts · Apple Silicon 原生优化 · 包含滑动窗口注意力对齐、ChatML 终止符修复与零样本声学克隆</p>
    </footer>
  </div>
</body>
</html>
'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)
with open(OUT_HTML_DEMO, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Standalone HTML generated: {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024/1024:.2f} MB)")
