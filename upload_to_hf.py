
import os, sys
from huggingface_hub import HfApi

api = HfApi()

REPOS = [
    ("vanch007/Ming-3.0-tts-0.5b-mlx-4bit", "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-0.5b-mlx-4bit"),
    ("vanch007/Ming-3.0-tts-0.5b-mlx-8bit", "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-0.5b-mlx-8bit"),
    ("vanch007/Ming-3.0-tts-0.5b-mlx-bf16", "/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-0.5b-mlx-bf16"),
]

for repo_id, folder_path in REPOS:
    print(f"\n==========================================")
    print(f"Creating repo if not exists: {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    print(f"Uploading {folder_path} -> {repo_id}...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload pure MLX weights ({repo_id.split('-')[-1]}) with test audios"
    )
    print(f"Successfully uploaded {repo_id}!")

print("\nAll 3 MLX models uploaded to Hugging Face successfully!")
