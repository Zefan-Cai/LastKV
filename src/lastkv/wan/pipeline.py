# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# LastKV causal rollout adapted from world_models b65c563ffac350811444cab1710c6903a6295215.
# Licensed under Apache-2.0.
"""Single-device LastKV Wan text-to-video generation with staged CPU offload."""

import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from .cache import allocate_caches
from .loading import load_model
from .scheduler import FlowUniPCMultistepScheduler

DEFAULT_NEGATIVE_PROMPT = "deformed, low quality, bad aesthetics, poor aesthetics"
VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def latent_frame_count(num_frames, block_frames=3):
    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    latent_frames = math.ceil((num_frames - 1) / 4) + 1
    return math.ceil(latent_frames / block_frames) * block_frames


def base_assets(base_model):
    path = Path(base_model).expanduser()
    if not path.is_dir():
        path = Path(snapshot_download(str(base_model), allow_patterns=[
            "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"]))
    for name in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing Wan base asset: {path / name}")
    return path


class LastKVWanPipeline:
    """EMA by default; select a published raw branch through ``revision``."""

    def __init__(self, model, text_encoder=None, vae=None, device="cuda", dtype=torch.bfloat16):
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_id="ZefanCai/LastKV-Wan-1.3B", *, revision="main",
                        base_model=None, device="cuda", dtype=torch.bfloat16):
        from .t5 import T5EncoderModel
        from .vae import _video_vae

        model = load_model(model_id, revision=revision, device="cpu", dtype=dtype)
        if base_model is None:
            size = {1536: "1.3B", 5120: "14B"}.get(model.dim)
            if size is None:
                raise ValueError("Provide base_model for an unrecognized Wan size.")
            base_model = f"Wan-AI/Wan2.1-T2V-{size}"
        assets = base_assets(base_model)
        text_encoder = T5EncoderModel(
            text_len=model.text_len, dtype=dtype, device="cpu",
            checkpoint_path=str(assets / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(assets / "google/umt5-xxl"))
        vae = _video_vae(pretrained_path=str(assets / "Wan2.1_VAE.pth"), z_dim=16,
                         device="cpu").eval().requires_grad_(False).to(dtype=dtype)
        return cls(model, text_encoder, vae, device, dtype)

    @torch.inference_mode()
    def generate_latents(self, noise, prompt_embeds, negative_prompt_embeds, *,
                         steps=50, guidance_scale=5.0, shift=5.0, progress=True,
                         shared_cache=True):
        """Roll out [B,F,C,H,W] noise against a clean prefix, then commit each block."""
        if steps < 1 or shift <= 0:
            raise ValueError("steps and shift must be positive.")
        batch, frames, channels, height, width = noise.shape
        block = self.model.num_frame_per_block
        if frames % block or height % 2 or width % 2:
            raise ValueError("Latent frames must form complete blocks; spatial dimensions must be even.")
        frame_tokens = height * width // 4
        cache_tokens = (frames - block) * frame_tokens
        positive = allocate_caches(self.model, batch, cache_tokens, block * frame_tokens,
                                   noise.dtype, noise.device, shared=shared_cache)
        negative = allocate_caches(self.model, batch, cache_tokens, block * frame_tokens,
                                   noise.dtype, noise.device, shared=shared_cache)
        cross_positive = [{"k": None, "v": None, "is_init": False} for _ in self.model.blocks]
        cross_negative = [{"k": None, "v": None, "is_init": False} for _ in self.model.blocks]
        output = torch.zeros_like(noise)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)

        def predict(latents, timestep, context, cache, cross_cache, start, update=False):
            return self.model(
                latents.permute(0, 2, 1, 3, 4), timestep, context,
                kv_cache=cache, crossattn_cache=cross_cache,
                current_start=start * frame_tokens, update_cache=update).permute(0, 2, 1, 3, 4)

        with torch.autocast(device_type=noise.device.type, dtype=self.dtype,
                            enabled=noise.device.type == "cuda"):
            for start in tqdm(range(0, frames, block), desc="Generating blocks", disable=not progress):
                latents = noise[:, start:start + block]
                scheduler.set_timesteps(steps, device=noise.device, shift=shift)
                for t in scheduler.timesteps:
                    timestep = t * torch.ones(batch, block, device=noise.device, dtype=torch.float32)
                    conditioned = predict(latents, timestep, prompt_embeds, positive, cross_positive, start)
                    unconditioned = predict(latents, timestep, negative_prompt_embeds, negative, cross_negative, start)
                    flow = unconditioned + guidance_scale * (conditioned - unconditioned)
                    latents = scheduler.step(flow, t, latents, return_dict=False)[0]
                output[:, start:start + block] = latents
                if start + block < frames:
                    clean_timestep = torch.zeros(batch, block, device=noise.device, dtype=torch.float32)
                    predict(latents, clean_timestep, prompt_embeds, positive, cross_positive, start, True)
                    predict(latents, clean_timestep, negative_prompt_embeds, negative, cross_negative, start, True)
        if not torch.isfinite(output).all():
            raise RuntimeError("Generation produced non-finite latent values.")
        return output

    @torch.inference_mode()
    def generate(self, prompt, *, negative_prompt=DEFAULT_NEGATIVE_PROMPT, num_frames=81,
                 height=480, width=832, steps=50, guidance_scale=5.0, shift=5.0,
                 seed=42, progress=True):
        """Return a CPU RGB float tensor [frames,3,height,width] in [0,1]."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Full Wan video generation requires CUDA.")
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        frames = latent_frame_count(num_frames, self.model.num_frame_per_block)
        if frames > 1024:
            raise ValueError("Requested duration exceeds the model's rotary-position table.")
        self.text_encoder.model.to(self.device)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            contexts = self.text_encoder([prompt, negative_prompt], self.device)
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        self.model.to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, frames, self.model.in_dim, height // 8, width // 8),
                            generator=generator, dtype=self.dtype, device=self.device)
        latents = self.generate_latents(noise, [contexts[0]], [contexts[1]], steps=steps,
                                        guidance_scale=guidance_scale, shift=shift, progress=progress)
        self.model.cpu()
        del noise, contexts
        torch.cuda.empty_cache()
        self.vae.to(self.device)
        mean = torch.tensor(VAE_MEAN, device=self.device, dtype=self.dtype)
        std = torch.tensor(VAE_STD, device=self.device, dtype=self.dtype)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            decoded = self.vae.decode(latents.permute(0, 2, 1, 3, 4), [mean, 1.0 / std])
        video = (decoded[0].float() * 0.5 + 0.5).clamp(0, 1).permute(1, 0, 2, 3)[:num_frames].cpu()
        self.vae.cpu()
        self.vae.clear_cache()
        if video.shape != (num_frames, 3, height, width) or not torch.isfinite(video).all():
            raise RuntimeError("VAE output failed the expected shape/finite check.")
        return video


def save_video(video, path, fps=16):
    import imageio.v2 as imageio

    if fps <= 0:
        raise ValueError("fps must be positive.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (video.clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
