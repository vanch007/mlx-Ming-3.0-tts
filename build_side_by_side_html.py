
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

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ming-3.0-tts (0.5B 稠密 vs Tiny 7.9B MoE) 同台同用例全场景试听对比平台</title>
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
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 40px 20px; line-height: 1.6; }
    .container { max-width: 1400px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 36px; }
    h1 { font-size: 2.3rem; font-weight: 700; margin-bottom: 12px; background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 36px; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; text-align: center; }
    .stat-title { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 1.6rem; font-weight: 700; }
    .stat-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }
    
    .case-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; margin-bottom: 28px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .case-header { padding: 16px 24px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    .case-title { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); }
    .case-body { padding: 20px 24px; }
    .case-desc { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 10px; }
    .case-text { background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 8px; margin-bottom: 18px; border-left: 4px solid var(--primary); font-size: 0.95rem; }
    
    .player-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }
    .player-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; }
    
    .player-card.ref { border-color: rgba(168, 85, 247, 0.4); }
    .player-card.dense { border-color: rgba(6, 182, 212, 0.5); background: rgba(6, 182, 212, 0.03); }
    .player-card.tiny-bf16 { border-color: rgba(56, 189, 248, 0.4); }
    .player-card.tiny-int8 { border-color: rgba(52, 211, 153, 0.5); background: rgba(52, 211, 153, 0.03); }
    .player-card.tiny-int4 { border-color: rgba(251, 191, 36, 0.4); }
    
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .player-name { font-weight: 600; font-size: 0.9rem; }
    
    .player-card.ref .player-name { color: var(--purple); }
    .player-card.dense .player-name { color: var(--cyan); }
    .player-card.tiny-bf16 .player-name { color: var(--primary); }
    .player-card.tiny-int8 .player-name { color: var(--success); }
    .player-card.tiny-int4 .player-name { color: var(--warning); }
    
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
      <h1>Ming-3.0-tts (0.5B vs Tiny 7.9B MoE) 同台同用例试听对比</h1>
      <p class="subtitle">单文件完全内嵌离线版 · 同一输入句子横向对比 0.5B 稠密模型与 Tiny 7.9B MoE 各精度表现</p>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">0.5B 稠密模型 (8-bit)</div>
        <div class="stat-value" style="color: var(--cyan)">1.76 GB</div>
        <div class="stat-sub">平均 RTF: ~0.39 · 极速实时推理</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tiny 7.9B (8-bit 推荐)</div>
        <div class="stat-value" style="color: var(--success)">9.78 GB</div>
        <div class="stat-sub">128 专家 MoE + KDA 线性注意力</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tiny 7.9B (4-bit 轻量)</div>
        <div class="stat-value" style="color: var(--warning)">5.42 GB</div>
        <div class="stat-sub">5GB 显存畅跑 7.9B 大模型</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">同台对比用例</div>
        <div class="stat-value" style="color: var(--primary)">15 组全场景</div>
        <div class="stat-sub">同一文本/参考原声 · 一对一直接对照</div>
      </div>
    </div>
'''

for idx, c in enumerate(data):
    cid = c['case_id']
    cname = c['category']
    cdesc = c['desc']
    text = c['text']
    
    # 0.5b 8-bit variant
    v_dense = c.get('dense_0_5b', {}).get('int8')
    # Tiny variants
    v_t_8bit = c.get('tiny_moe', {}).get('tiny_8bit')
    v_t_4bit = c.get('tiny_moe', {}).get('tiny_4bit')
    v_t_bf16 = c.get('tiny_moe', {}).get('tiny_bf16')
    
    ref_file = c.get('ref_audio')
    ref_b64 = to_b64(ref_file) if ref_file else ""
    
    html += f'''
    <div class="case-card">
      <div class="case-header">
        <div class="case-title">
          <span>{cid.upper()} · {cname}</span>
        </div>
        <span class="badge">44.1kHz / 16-bit PCM 同题对比</span>
      </div>
      <div class="case-body">
        <div class="case-desc">{cdesc}</div>
        <div class="case-text"><strong>同一测试输入文本:</strong> {text}</div>
        <div class="player-grid">
          
          <!-- 参考原声 -->
          <div class="player-card ref">
            <div>
              <div class="player-head">
                <span class="player-name">1. {"参考原声 (Prompt Audio)" if ref_b64 else "纯文本/指令生成模式"}</span>
                <span class="meta-tag">{"参考基准" if ref_b64 else "无参考音频"}</span>
              </div>
              <div class="metrics">
                <div>模式: <span class="metric-val">{"声学克隆提示" if ref_b64 else "Zero-Shot Text"}</span></div>
                <div>来源: <span class="metric-val">{"官方样例" if ref_b64 else "None"}</span></div>
                <div>状态: <span class="metric-val">{"已对齐" if ref_b64 else "无输入"}</span></div>
                <div>采样率: <span class="metric-val">44.1kHz</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + ref_b64 + "'></audio>" if ref_b64 else "<div style='color:var(--text-muted);font-size:0.8rem;padding:8px 0;'>本用例为纯文本/指令合成，无需参考音频</div>"}
          </div>

          <!-- 0.5B 8-bit (稠密模型) -->
          <div class="player-card dense">
            <div>
              <div class="player-head">
                <span class="player-name">2. 0.5B (8-bit 推荐)</span>
                <span class="meta-tag" style="background: rgba(6, 182, 212, 0.2); color: var(--cyan)">1.76 GB 稠密</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val" style="color: var(--cyan)">{v_dense['elapsed_s'] if v_dense else '—'}s</span></div>
                <div>RTF: <span class="metric-val" style="color: var(--cyan)">{v_dense['rtf'] if v_dense else '—'}</span></div>
                <div>时长: <span class="metric-val">{v_dense['duration_s'] if v_dense else '—'}s</span></div>
                <div>架构: <span class="metric-val">Qwen2 稠密</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + v_dense['b64'] + "'></audio>" if v_dense else "<div style='color:var(--text-muted);font-size:0.8rem;padding:8px 0;'>0.5B 官方不支持多人对话任务</div>"}
          </div>

          <!-- Tiny 7.9B MoE (8-bit) -->
          <div class="player-card tiny-int8">
            <div>
              <div class="player-head">
                <span class="player-name">3. Tiny 7.9B (8-bit 推荐)</span>
                <span class="meta-tag" style="background: rgba(52, 211, 153, 0.2); color: var(--success)">9.78 GB 128MoE</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val" style="color: var(--success)">{v_t_8bit['elapsed_s'] if v_t_8bit else '—'}s</span></div>
                <div>RTF: <span class="metric-val" style="color: var(--success)">{v_t_8bit['rtf'] if v_t_8bit else '—'}</span></div>
                <div>时长: <span class="metric-val">{v_t_8bit['duration_s'] if v_t_8bit else '—'}s</span></div>
                <div>架构: <span class="metric-val">KDA + MoE</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + v_t_8bit['b64'] + "'></audio>" if v_t_8bit else ""}
          </div>

          <!-- Tiny 7.9B MoE (4-bit) -->
          <div class="player-card tiny-int4">
            <div>
              <div class="player-head">
                <span class="player-name">4. Tiny 7.9B (4-bit 轻量)</span>
                <span class="meta-tag" style="background: rgba(251, 191, 36, 0.2); color: var(--warning)">5.42 GB 超轻</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val">{v_t_4bit['elapsed_s'] if v_t_4bit else '—'}s</span></div>
                <div>RTF: <span class="metric-val">{v_t_4bit['rtf'] if v_t_4bit else '—'}</span></div>
                <div>时长: <span class="metric-val">{v_t_4bit['duration_s'] if v_t_4bit else '—'}s</span></div>
                <div>显存省: <span class="metric-val">70%</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + v_t_4bit['b64'] + "'></audio>" if v_t_4bit else ""}
          </div>

          <!-- Tiny 7.9B MoE (bf16) -->
          <div class="player-card tiny-bf16">
            <div>
              <div class="player-head">
                <span class="player-name">5. Tiny 7.9B (bf16 全精度)</span>
                <span class="meta-tag" style="background: rgba(56, 189, 248, 0.2); color: var(--primary)">17.96 GB</span>
              </div>
              <div class="metrics">
                <div>耗时: <span class="metric-val">{v_t_bf16['elapsed_s'] if v_t_bf16 else '—'}s</span></div>
                <div>RTF: <span class="metric-val">{v_t_bf16['rtf'] if v_t_bf16 else '—'}</span></div>
                <div>时长: <span class="metric-val">{v_t_bf16['duration_s'] if v_t_bf16 else '—'}s</span></div>
                <div>后端: <span class="metric-val">原生浮点</span></div>
              </div>
            </div>
            {"<audio controls preload='none' src='" + v_t_bf16['b64'] + "'></audio>" if v_t_bf16 else ""}
          </div>

        </div>
      </div>
    </div>
'''

html += '''
    <footer>
      <p>mlx-Ming-3.0-tts · Apple Silicon 原生优化 · 0.5B 稠密模型与 Tiny 7.9B MoE 同台同题对比页面</p>
    </footer>
  </div>
</body>
</html>
'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)
with open(OUT_HTML_DEMO, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Generated unified side-by-side HTML: {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024/1024:.2f} MB)")
