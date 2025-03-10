import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2, InterpolationMode
from PIL import Image
import decord

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_decode, vae_encode

with torch.inference_mode():
    ckpt_path = Path("./outputs/2025-01-01_14-20-39/checkpoints/00011000/ltx_vae_merged.safetensors")
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path).to("cuda", dtype=torch.bfloat16)
    normalize = False
    timestep = None
    
    sd = vae.state_dict()
    for key in sd.keys():
        if "decoder" in key:
            if ".conv.conv." in key:
                alpha = 0.5
            else:
                alpha = 0.05
            std, mean = torch.std_mean(sd[key])
            sd[key] = sd[key] * (1 - alpha) + (torch.randn_like(sd[key]) * std + mean) * alpha
    vae.load_state_dict(sd)
    
    # print(vae)
    # quit()
    
    # input_pixels = torch.rand(1, 3, 65, 512, 512).to("cuda", dtype=torch.bfloat16) # * 2 - 1
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader("./inputs/1.mp4")
    frames = vr[:65]
    
    transforms = v2.Compose([
        v2.ToDtype(torch.bfloat16, scale=True),
        v2.Resize(
            size=512,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
            ),
        v2.CenterCrop(size=512),
    ])
    frames = frames.movedim(3, 1).contiguous() # FHWC -> FCHW
    frames = transforms(frames)
    frames = torch.clamp(torch.nan_to_num(frames), min=0, max=1)
    input_pixels = frames.movedim(1, 0).unsqueeze(0).to("cuda") # FCHW -> BCFHW
    
    latents = vae_encode(
        media_items = input_pixels,
        vae = vae,
        vae_per_channel_normalize = normalize,
    )
    
    print(f"latents shape = {latents.shape}, mean = {latents.mean().item()}, std = {latents.std().item()}")
    
    if timestep is not None:
        noise = torch.randn_like(latents)
        timestep = torch.tensor([timestep] * latents.shape[0]).to(latents.device)
        noise_scale = timestep[:, None, None, None, None] * 0.5
        latents = latents * (1 - noise_scale) + noise * noise_scale
    
    output_pixels = vae_decode(
        latents = latents,
        vae = vae,
        vae_per_channel_normalize = normalize,
        timestep = timestep,
    )
    
    loss = F.mse_loss(output_pixels, input_pixels, reduction="mean").item()
    print(f"output {loss = }, min = {output_pixels.min().item()}, max = {output_pixels.max().item()}")
    
    def fft_loss(target, pred):
        diff = target.float().mean(2) - pred.float().mean(2)
        print(diff.shape)
        diff = diff - diff.mean(dim=list(range(1, len(diff.shape))), keepdim=True)
        fft = torch.view_as_real(torch.fft.rfft2(diff))
        return torch.mean(fft ** 2) ** 0.5
    
    print(f"fft loss = {fft_loss(input_pixels, output_pixels).item()}")
    
    for i in range(output_pixels.shape[2]):
        frame = output_pixels[0, :, i].movedim(0, -1).float().clamp(0, 1).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = Image.fromarray(frame)
        frame.save(f"./outputs/noise_model/frame_{i:04}.png")