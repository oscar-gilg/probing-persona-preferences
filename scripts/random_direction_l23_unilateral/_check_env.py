import sys
import torch
print("python", sys.executable, sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
