import os
import torch

if torch.cuda.is_available() and 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"