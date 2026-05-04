import os, sys, torch, transformers
from dotenv import load_dotenv

load_dotenv()
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"OPENROUTER_API_KEY_set={bool(os.getenv('OPENROUTER_API_KEY'))}")
print(f"HF_TOKEN_set={bool(os.getenv('HF_TOKEN'))}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)} mem_GB={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}")
