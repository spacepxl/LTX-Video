import os
import math
import random
import logging
import argparse
import datetime
from glob import glob
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2, InterpolationMode
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import save_image
import decord

from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.autoencoders.vae_encode import vae_decode, vae_encode
from transformers import AutoModel


def get_dino_model(model="dinov2-base", device="cuda"):
    if os.path.exists("models/dino/" + model):
        dino_model = AutoModel.from_pretrained("models/dino/" + model, local_files_only=True).to(device)
    else:
        dino_model = AutoModel.from_pretrained("facebook/" + model).to(device)
        dino_model.save_pretrained("models/dino/" + model)
    return dino_model


def calculate_dino_loss(target, pred, model, layers):
    target_activations = model(pixel_values=target, output_hidden_states=True).hidden_states
    pred_activations = model(pixel_values=pred, output_hidden_states=True).hidden_states
    layer_losses = []
    for layer in layers:
        layer_losses.append(F.l1_loss(target_activations[layer][:, 1:], pred_activations[layer][:, 1:]))
    return torch.stack(layer_losses).mean()


@contextmanager
def temp_rng(new_seed=None):
	"""
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""
    
	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	# old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()
    
	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		# np.random.seed(new_seed)
		random.seed(new_seed)
    
	yield
    
	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	# np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)


class RandomDownscale(v2.Transform):
    def __init__(
        self,
        min_size: int,
        p: float = 0.5,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` should be a floating point value in the interval [0.0, 1.0].")
        super().__init__()
        self.p = p
        self.min_size = min_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = v2._utils.query_size(flat_inputs)
        size = min(orig_height, orig_width)
        if size <= self.min_size:
            size = self.min_size
        else:
            size = int(torch.randint(self.min_size, size, ()))
        return dict(size=[size])

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if torch.rand(1) >= self.p:
            return inpt
        return self._call_kernel(
            v2.functional.resize, inpt, params["size"], interpolation=self.interpolation, antialias=self.antialias
            )


class TrainDataset(Dataset):
    def __init__(self, args):
        self.root_folder = args.train_data_dir
        self.resolution = args.resolution
        self.num_frames = args.num_frames
        self.videos_folders = os.listdir(self.root_folder)
        
        transform_list = [v2.ToDtype(torch.float32, scale=True),]
        if args.flip_x:
            transform_list.append(v2.RandomHorizontalFlip(p=0.5))
        transform_list.append(
            RandomDownscale(
                min_size=self.resolution,
                p=0.7,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
                )
            )
        transform_list.append(
            v2.RandomCrop(
                size=self.resolution,
                pad_if_needed=True,
                padding_mode="reflect",
                )
            )
        if args.color_jitter:
            transform_list.append(
                v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                    )
                )
        self.transforms = v2.Compose(transform_list)
    
    def __len__(self):
        return len(self.videos_folders)
    
    def __getitem__(self, idx):
        video_id = self.videos_folders[idx]
        video_folder = os.path.join(self.root_folder, video_id)
        video_file = os.path.join(video_folder, video_id) + ".mp4"
        
        vr = decord.VideoReader(video_file)
        stride = min(random.randint(1, 3), len(vr) // self.num_frames)
        if stride > 0:
            seg_len = stride * self.num_frames
            start_frame = random.randint(0, len(vr) - seg_len)
            frames = vr[start_frame : start_frame+seg_len : stride]
        else: frames = vr[:]
        while frames.shape[0] < self.num_frames:
            frames = torch.cat([frames, frames[-1].unsqueeze(0)], dim=0)
        
        frames = frames.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> BFCHW
        frames = self.transforms(frames).movedim(1, 2).squeeze(0) * 2 - 1 # CFHW
        frames = torch.clamp(torch.nan_to_num(frames), min=-1, max=1)
        return frames


class TestDataset(Dataset):
    def __init__(self, args):
        self.root_folder = args.test_data_dir
        self.resolution = args.resolution
        self.num_frames = args.num_frames
        self.num_samples = args.test_samples
        
        self.videos_folders = os.listdir(self.root_folder)
        stride = len(self.videos_folders) // self.num_samples
        self.videos_folders = self.videos_folders[0 : stride*self.num_samples : stride]
        
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=self.resolution,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
                ),
            v2.CenterCrop(size=self.resolution),
            ])
    
    def __len__(self):
        return len(self.videos_folders)
    
    def __getitem__(self, idx):
        video_id = self.videos_folders[idx]
        video_folder = os.path.join(self.root_folder, video_id)
        video_file = os.path.join(video_folder, video_id) + ".mp4"
        
        vr = decord.VideoReader(video_file)
        frames = vr[:self.num_frames]
        while frames.shape[0] < 16:
            frames = torch.cat([frames, frames[-1].unsqueeze(0)], dim=0)
        
        frames = frames.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> BFCHW
        frames = self.transforms(frames).movedim(1, 2).squeeze(0) * 2 - 1 # CFHW
        frames = torch.clamp(torch.nan_to_num(frames), min=-1, max=1)
        return frames


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Finetune LTXV decoder, leaving the latent space unchanged",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--pretrained_model_path",
        type = str,
        default = None,
        required = True,
        help = "Path to pretrained VAE model .safetensors",
        )
    parser.add_argument(
        "--train_data_dir",
        type = str,
        default = None,
        required = True,
        help = "Path to train folder where each subfolder contains one video",
        )
    parser.add_argument(
        "--test_data_dir",
        type = str,
        default = None,
        help = "Path to test folder where each subfolder contains one video",
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results"
        )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training"
        )
    parser.add_argument(
        "--resolution",
        type = int,
        default = 448,
        help = "Patch resolution for training/testing"
        )
    parser.add_argument(
        "--num_frames",
        type = int,
        default = 49,
        help = "Context length for videos"
        )
    parser.add_argument(
        "--color_jitter",
        action = "store_true",
        default = False,
        help = "Randomly adjust the color of inputs for augmentation"
        )
    parser.add_argument(
        "--flip_x",
        action = "store_true",
        default = False,
        help = "Randomly flip inputs horizontally for augmentation"
        )
    parser.add_argument(
        "--test_samples",
        type = int,
        default = 8,
        help = "Number of videos to sample for validation",
        )
    parser.add_argument(
        "--train_batch_size",
        type = int,
        default = 1,
        help = "Batch size for training",
        )
    parser.add_argument(
        "--train_steps",
        type = int,
        default = 100_000,
        help = "Total number of training steps",
        )
    parser.add_argument(
        "--test_steps",
        type = int,
        default = 100,
        help = "Validate on the test set after every n steps",
        )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-6,
        help = "Initial learning rate to use",
        )
    parser.add_argument(
        "--scale_lr",
        action = "store_true",
        default = True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size",
        )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 1000,
        help = "Save a checkpoint of the training state every X steps",
        )
    parser.add_argument(
        "--use_8bit_adam",
        action = "store_true",
        default = False,
        help = "Use 8-bit Adam from bitsandbytes"
        )
    parser.add_argument(
        "--dino_scale",
        type = float,
        default = 10.0,
        help = "Scaling factor for the DINO loss",
        )
    parser.add_argument(
        "--fft_scale",
        type = float,
        default = 0.0,
        help = "Scaling factor for the FFT loss",
        )
    parser.add_argument(
        "--temporal_scale",
        type = float,
        default = 0.0,
        help = "Scaling factor for RAFT optical flow warped temporal loss",
        )
    parser.add_argument(
        "--partial_randomize",
        action = "store_true",
        default = False,
        help = "Inject random noise into the decoder weights to disrupt frequency patterns"
        )
    parser.add_argument(
        "--train_module",
        type = str,
        default = "decoder",
        choices = ["encoder", "decoder"],
        help = "Which module of the VAE to train",
        )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    decord.bridge.set_bridge('torch')
    device = "cuda"
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, date_time)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "log")
    val_dir = os.path.join(output_dir, "val")
    for d in [output_dir, checkpoint_dir, log_dir, val_dir]:
        os.makedirs(d, exist_ok=True)
    t_writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    
    train_dataset = TrainDataset(args)
    test_dataset = TestDataset(args)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size = args.train_batch_size,
        num_workers = 0,
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        shuffle = False,
        batch_size = 1,
        num_workers = 0,
        )
    
    vae = CausalVideoAutoencoder.from_pretrained(args.pretrained_model_path).to(dtype=torch.float32)
    
    if args.partial_randomize:
        with temp_rng(args.seed):
            sd = vae.state_dict()
            for key in sd.keys():
                if "decoder" in key:
                    alpha = 0.25 if ".conv.conv." in key else 0.025 # target the pixel shuffle upscale blocks at higher strength
                    std, mean = torch.std_mean(sd[key])
                    sd[key] = sd[key] * (1 - alpha) + (torch.randn_like(sd[key]) * std + mean) * alpha
            vae.load_state_dict(sd)
    
    vae.to(device)
    vae.requires_grad_(False)
    if args.train_module == "encoder":
        vae.encoder.requires_grad_(True)
    else:
        vae.decoder.requires_grad_(True)
    
    if args.use_8bit_adam:
        try: import bitsandbytes as bnb
        except ImportError: raise ImportError("Please install bitsandbytes to use 8-bit Adam")
        optimizer_cls = bnb.optim.AdamW8bit
    else: optimizer_cls = torch.optim.AdamW
    
    train_parameters = list(filter(lambda p: p.requires_grad, vae.parameters()))
    scaled_lr = args.learning_rate * (args.train_batch_size ** 0.5) if args.scale_lr else args.learning_rate
    optimizer = optimizer_cls(
        params       = train_parameters,
        lr           = scaled_lr,
        betas        = (args.adam_beta1, args.adam_beta2),
        weight_decay = args.adam_weight_decay,
        eps          = args.adam_epsilon,
        )
    
    if args.dino_scale > 0:
        dino_model = get_dino_model(device=device)
        dino_model.requires_grad_(False)
    else:
        dino_model = None
    
    if args.temporal_scale > 0:
        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        raft_model.requires_grad_(False)
    else:
        raft_model = None
    
    dino_crop = v2.RandomCrop(size=224)
    def batch_dino_loss(target, pred, samples=8):
        losses = []
        for frame in random.sample(range(target.shape[2]), samples):
            minibatch = torch.cat([target[:, :, frame], pred[:, :, frame]], dim=0)
            f_target, p_target = torch.chunk(dino_crop(minibatch), chunks=2, dim=0)
            losses.append(calculate_dino_loss(f_target, p_target, dino_model, [2, 5]))
        return torch.stack(losses).mean()
    
    def batch_fft_loss(target, pred):
        diff = target.float().mean(2) - pred.float().mean(2)
        diff = diff - diff.mean(dim=list(range(1, len(diff.shape))), keepdim=True)
        fft = torch.view_as_real(torch.fft.rfft2(diff))
        return torch.mean(fft ** 2) ** 0.5
    
    def batch_temporal_loss(target, pred, samples=8):
        target1, target2, pred1, pred2 = [], [], [], []
        for frame in random.sample(range(target.shape[2] - 1), samples):
            target1.append(target[:, :, frame].float())
            target2.append(target[:, :, frame + 1].float())
            pred1.append(pred[:, :, frame].float())
            pred2.append(pred[:, :, frame + 1].float())
        target1, target2, pred1, pred2 = torch.cat(target1), torch.cat(target2), torch.cat(pred1), torch.cat(pred2)
        flow = raft_model(target1, target2)[-1].movedim(1, -1) # BCHW -> BHWC
        flow[..., 0] = flow[..., 0] * 2 / target1.shape[-1] + torch.linspace(-1, 1, steps=target1.shape[-1])[None, None, :].to("cuda") # x
        flow[..., 1] = flow[..., 1] * 2 / target1.shape[-2] + torch.linspace(-1, 1, steps=target1.shape[-2])[None, :, None].to("cuda") # y
        target_diff = target1 - F.grid_sample(target2, flow, align_corners=True)
        pred_diff = pred1 - F.grid_sample(pred2, flow, align_corners=True)
        return F.mse_loss(target_diff, pred_diff)
    
    print(f"===================== Training =====================")
    print(f"                 examples = {len(train_dataset)}")
    print(f"             test samples = {len(test_dataset)}")
    print(f"    batch size per device = {args.train_batch_size}")
    print(f"          steps per epoch = {len(train_dataloader)}")
    print(f"             total epochs = {args.train_steps / len(train_dataloader):0.2f}")
    print(f"    perceptual loss scale = {args.dino_scale}")
    print(f"           fft loss scale = {args.fft_scale}")
    print(f"      temporal loss scale = {args.temporal_scale}")
    print(f"                  base lr = {args.learning_rate}")
    print(f"                scaled lr = {scaled_lr}")
    print(f"                optimizer = {optimizer_cls.__name__}")
    print(f"====================================================")
    
    global_step = 0
    progress_bar = tqdm(range(global_step, args.train_steps))
    vae.train()
    while global_step < args.train_steps:
        for step, batch in enumerate(train_dataloader):
            target = batch.to(device)
            optimizer.zero_grad()
            
            latents = vae_encode(media_items=target, vae=vae, vae_per_channel_normalize=False)
            output_pixels = vae_decode(latents=latents, vae=vae, vae_per_channel_normalize=False, timestep=None)
            
            loss = F.l1_loss(target, output_pixels, reduction="mean")
            t_writer.add_scalar("loss/MAE", loss.detach().item(), global_step)
            
            if dino_model is not None:
                dino_loss = batch_dino_loss(target, output_pixels)
                loss += dino_loss * args.dino_scale
                t_writer.add_scalar("loss/dino", dino_loss.detach().item(), global_step)
            
            if args.fft_scale > 0:
                fft_loss = batch_fft_loss(target, output_pixels)
                loss += fft_loss * args.fft_scale
                t_writer.add_scalar("loss/fft", fft_loss.detach().item(), global_step)
            
            if raft_model is not None:
                temporal_loss = batch_temporal_loss(target, output_pixels)
                loss += temporal_loss * args.temporal_scale
                t_writer.add_scalar("loss/temporal", temporal_loss.detach().item(), global_step)
            
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            
            loss.backward()
            optimizer.step()
            global_step += 1
            progress_bar.update(1)
            
            # save
            if global_step == args.train_steps or global_step % args.checkpointing_steps == 0:
                step_checkpoint_dir = os.path.join(checkpoint_dir, f"{global_step:08}")
                os.makedirs(step_checkpoint_dir, exist_ok=True)
                vae.save_pretrained(step_checkpoint_dir, safe_serialization=True)
            
            if global_step == 1 or global_step % args.test_steps == 0:
                with torch.inference_mode(), temp_rng(args.seed):
                    pixel_val = 0
                    dino_val = 0
                    for test_step, test_batch in enumerate(tqdm(test_dataloader, desc="val", leave=False)):
                        target = test_batch.to(device)
                        
                        latents = vae_encode(media_items=target, vae=vae, vae_per_channel_normalize=False)
                        output_pixels = vae_decode(latents=latents, vae=vae, vae_per_channel_normalize=False, timestep=None)
                        
                        val_image = os.path.join(val_dir, f"{test_step:04}_{global_step:08}.png")
                        save_image(output_pixels[:, :, 8].detach() * 0.5 + 0.5, val_image)
                        
                        pixel_val += F.l1_loss(target, output_pixels, reduction="mean").detach().item() / len(test_dataloader)
                        if dino_model is not None:
                            dino_val += batch_dino_loss(target, output_pixels).detach().item() / len(test_dataloader)
                    
                    t_writer.add_scalar("val/MAE", pixel_val, global_step)
                    if dino_model is not None:
                        t_writer.add_scalar("val/dino", dino_val, global_step)
            
            if global_step >= args.train_steps:
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)