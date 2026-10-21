
import time, os, sys
import mlx.core as mx

sys.path.insert(0, '/Users/vanch/mlx-Ming-3.0-tts')
from mlx_ming_tts.pipeline import MingMLX, save_wav

ming = MingMLX('models/Ming-3.0-tts-0.5b-mlx-8bit')

PROMPT_WAV = "/Users/vanch/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

vi_prompt = "Please generate speech based on the following description.\nTarget language: Vietnamese.\n"
vi_text = "Xin chào, đây là hệ thống tổng hợp giọng nói Ming chạy hoàn toàn trên Apple Silicon."

ms_prompt = "Please generate speech based on the following description.\nTarget language: Malay.\n"
ms_text = "Helo, ini adalah sistem sintesis pertuturan Ming yang berjalan sepenuhnya pada Apple Silicon."

print("Generating Vietnamese speech...")
wav_vi = ming.generate(vi_prompt, vi_text, ref_audio=PROMPT_WAV, prompt_text=PROMPT_TEXT, sigma=0.15, cfg=2.0)
save_wav(wav_vi, 'output/test_vietnamese.wav', ming.sample_rate)
print(f"Saved output/test_vietnamese.wav, duration: {len(wav_vi)/ming.sample_rate:.2f}s")

print("Generating Malay speech...")
wav_ms = ming.generate(ms_prompt, ms_text, ref_audio=PROMPT_WAV, prompt_text=PROMPT_TEXT, sigma=0.15, cfg=2.0)
save_wav(wav_ms, 'output/test_malay.wav', ming.sample_rate)
print(f"Saved output/test_malay.wav, duration: {len(wav_ms)/ming.sample_rate:.2f}s")
