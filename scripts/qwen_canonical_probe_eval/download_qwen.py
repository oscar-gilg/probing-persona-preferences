"""One-shot: download Qwen-3.5-122B-A10B weights to HF cache on the pod.

Run with HF_HUB_ENABLE_HF_TRANSFER=1 for ~10x faster download.
"""
import os

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

load_dotenv()
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
path = snapshot_download(
    "Qwen/Qwen3.5-122B-A10B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
)
print(f"downloaded to: {path}")
