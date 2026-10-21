
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
  <title>Ming-3.0-tts (0.5B 稠密 vs Tiny 7.9B MoE) 6个版本全精度同台对比</title>
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
    header { text-align: center; margin-bottom: 36px; }
    h1 { font-size: 2.3rem; font-weight: 700; margin-bottom: 12px; background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 36px; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; text-align: center; }
    .stat-title { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 1.4rem; font-weight: 700; }
    .stat-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
    
    .case-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; margin-bottom: 32px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .case-header { padding: 16px 24px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
    .case-title { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; background: rgba(56, 189, 248, 0.15); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.3); }
    .case-body { padding: 22px 24px; }
    .case-desc { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 10px; }
    .case-text { background: rgba(0,0,0,0.25); padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid var(--primary); font-size: 0.95rem; }
    
    .model-group-title { font-size: 0.95rem; font-weight: 700; color: #cbd5e1; margin: 16px 0 10px 0; display: flex; align-items: center; gap: 8px; }
    .player-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-bottom: 14px; }
    .player-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between; }
    
    .player-card.ref { border-color: rgba(168, 85, 247, 0.4); }
    .player-card.dense-bf16 { border-color: rgba(56, 189, 248, 0.3); }
    .player-card.dense-int8 { border-color: rgba(6, 182, 212, 0.5); background: rgba(6, 182, 212, 0.03); }
    .player-card.dense-int4 { border-color: rgba(251, 191, 36, 0.3); }
    
    .player-card.tiny-bf16 { border-color: rgba(129, 140, 248, 0.4); }
    .player-card.tiny-int8 { border-color: rgba(52, 211, 153, 0.5); background: rgba(52, 211, 153, 0.03); }
    .player-card.tiny-int4 { border-color: rgba(245, 158, 11, 0.4); }
    
    .player-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .player-name { font-weight: 600; font-size: 0.88rem; }
    
    .player-card.ref .player-name { color: var(--purple); }
    .player-card.dense-bf16 .player-name { color: var(--primary); }
    .player-card.dense-int8 .player-name { color: var(--cyan); }
    .player-card.dense-int4 .player-name { color: var(--warning); }
    
    .player-card.tiny-bf16 .player-name { color: var(--indigo); }
    .player-card.tiny-int8 .player-name { color: var(--success); }
    .player-card.tiny-int4 .player-name { color: #f59e0b; }
    
    .meta-tag { font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.06); color: var(--text-muted); }
    .metrics { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px; display: grid; grid-template-columns: 1fr 1fr; gap: 3px; }
    .metric-val { font-weight: 600; color: #cbd5e1; }
    audio { width: 100%; height: 36px; border-radius: 6px; outline: none; }
    
    footer { text-align: center; margin-top: 50px; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--card-border); padding-top: 24px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Ming-3.0-tts (0.5B vs Tiny 7.9B MoE) 6个版本全精度同台对比</h1>
      <p class="subtitle">单文件完全内嵌离线版 · 同一测试句子横向对比【0.5B 3精度】与【Tiny 7.9B 3精度】共 6 个 MLX 原生模型版本</p>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">0.5B bf16 (全精度)</div>
        <div class="stat-value" style="color: var(--primary)">3.08 GB</div>
        <div class="stat-sub">原生浮点全精度</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">0.5B 8-bit (主力推荐)</div>
        <div class="stat-value" style="color: var(--cyan)">1.76 GB</div>
        <div class="stat-sub">RTF ~0.39 · 极速实时</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">0.5B 4-bit (超轻量)</div>
        <div class="stat-value" style="color: var(--warning)">1.06 GB</div>
        <div class="stat-sub">仅 1GB 内存消耗</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tiny 7.9B (bf16)</div>
        <div class="stat-value" style="color: var(--indigo)">17.96 GB</div>
        <div class="stat-sub">128 专家 MoE 原生全精度</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tiny 7.9B (8-bit 推荐)</div>
        <div class="stat-value" style="color: var(--success)">9.78 GB</div>
        <div class="stat-sub">大模型无损音质</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tiny 7.9B (4-bit 轻量)</div>
        <div class="stat-value" style="color: #f59e0b">5.42 GB</div>
        <div class="stat-sub">显存省 70% 畅跑 MoE</div>
      </div>
    </div>
'''

def render_player(card_class, name, badge, meta_dict, audio_b64, unavailable_text=None):
    if unavailable_text:
        return f'''
        <div class="player-card {card_class}">
          <div>
            <div class="player-head">
              <span class="player-name">{name}</span>
              <span class="meta-tag">{badge}</span>
            </div>
            <div style="color:var(--text-muted);font-size:0.8rem;padding:12px 0;">{unavailable_text}</div>
          </div>
        </div>
        '''
    dur = meta_dict.get('duration_s', '—')
    elapsed = meta_dict.get('elapsed_s', '—')
    rtf = meta_dict.get('rtf', '—')
    return f'''
    <div class="player-card {card_class}">
      <div>
        <div class="player-head">
          <span class="player-name">{name}</span>
          <span class="meta-tag">{badge}</span>
        </div>
        <div class="metrics">
          <div>耗时: <span class="metric-val">{elapsed}s</span></div>
          <div>RTF: <span class="metric-val">{rtf}</span></div>
          <div>时长: <span class="metric-val">{dur}s</span></div>
          <div>采样率: <span class="metric-val">44.1kHz</span></div>
        </div>
      </div>
      <audio controls preload="none" src="{audio_b64}"></audio>
    </div>
    '''

for idx, c in enumerate(data):
    cid = c['case_id']
    cname = c['category']
    cdesc = c['desc']
    text = c['text']
    
    dense_bf16 = c.get('dense_0_5b', {}).get('bf16')
    dense_int8 = c.get('dense_0_5b', {}).get('int8')
    dense_int4 = c.get('dense_0_5b', {}).get('int4')
    
    tiny_bf16 = c.get('tiny_moe', {}).get('tiny_bf16')
    tiny_int8 = c.get('tiny_moe', {}).get('tiny_8bit')
    tiny_int4 = c.get('tiny_moe', {}).get('tiny_4bit')
    
    ref_file = c.get('ref_audio')
    ref_b64 = to_b64(ref_file) if ref_file else ""
    
    html += f'''
    <div class="case-card">
      <div class="case-header">
        <div class="case-title">
          <span>{cid.upper()} · {cname}</span>
        </div>
        <span class="badge">同题同台 6 个模型版本对比</span>
      </div>
      <div class="case-body">
        <div class="case-desc">{cdesc}</div>
        <div class="case-text"><strong>同一测试输入文本:</strong> {text}</div>
        
        <!-- 参考基准 (如有) -->
        {f"""
        <div style="margin-bottom: 12px; max-width: 380px;">
          <div class="player-card ref">
            <div>
              <div class="player-head">
                <span class="player-name">官方参考原声 (Prompt Audio)</span>
                <span class="meta-tag">基准原声</span>
              </div>
            </div>
            <audio controls preload="none" src="{ref_b64}"></audio>
          </div>
        </div>
        """ if ref_b64 else ""}

        <!-- 第一组：0.5B 稠密模型 (3个版本) -->
        <div class="model-group-title">
          <span>🔹 0.5B 稠密模型 (Dense Architecture) · 3个版本</span>
        </div>
        <div class="player-grid">
          {render_player("dense-bf16", "0.5B bf16 (全精度)", "3.08 GB", dense_bf16 or {}, dense_bf16.get('b64', '') if dense_bf16 else None, "0.5b 不支持多人对话" if not dense_bf16 else None)}
          {render_player("dense-int8", "0.5B 8-bit (推荐主力)", "1.76 GB", dense_int8 or {}, dense_int8.get('b64', '') if dense_int8 else None, "0.5b 不支持多人对话" if not dense_int8 else None)}
          {render_player("dense-int4", "0.5B 4-bit (超轻量)", "1.06 GB", dense_int4 or {}, dense_int4.get('b64', '') if dense_int4 else None, "0.5b 不支持多人对话" if not dense_int4 else None)}
        </div>

        <!-- 第二组：Tiny 7.9B MoE 模型 (3个版本) -->
        <div class="model-group-title">
          <span>🔸 Tiny 7.9B MoE 模型 (128 Experts + KDA) · 3个版本</span>
        </div>
        <div class="player-grid">
          {render_player("tiny-bf16", "Tiny 7.9B bf16 (全精度)", "17.96 GB", tiny_bf16 or {}, tiny_bf16.get('b64', '') if tiny_bf16 else None)}
          {render_player("tiny-int8", "Tiny 7.9B 8-bit (推荐主力)", "9.78 GB", tiny_int8 or {}, tiny_int8.get('b64', '') if tiny_int8 else None)}
          {render_player("tiny-int4", "Tiny 7.9B 4-bit (超轻量)", "5.42 GB", tiny_int4 or {}, tiny_int4.get('b64', '') if tiny_int4 else None)}
        </div>

      </div>
    </div>
'''

html += '''
    <footer>
      <p>mlx-Ming-3.0-tts · Apple Silicon 原生优化 · 0.5B 与 Tiny 7.9B MoE 全部 6 个版本全场景横向对照</p>
    </footer>
  </div>
</body>
</html>
'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)
with open(OUT_HTML_DEMO, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Generated complete 6-version side-by-side HTML: {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024/1024:.2f} MB)")
