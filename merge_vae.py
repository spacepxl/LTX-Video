import os
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


dtype = torch.bfloat16
original_path = "./ltx-video-v0.9-vae.safetensors"
new_path = "./outputs/2025-01-01_14-20-39/checkpoints/00011000/diffusion_pytorch_model.safetensors"
output_path = os.path.join(os.path.split(new_path)[0], "ltx_vae_merged.safetensors")

state_dict = {}
with safe_open(new_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if "decoder" in key or "encoder" in key:
            state_dict[key] = f.get_tensor(key).to(dtype)

with safe_open(original_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        if "statistics" in key:
            state_dict[key] = f.get_tensor(key).to(dtype)

save_file(state_dict, output_path)