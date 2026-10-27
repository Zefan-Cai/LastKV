"""LastKV novel-view rendering, preserving the released checkpoint parameter names."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NVSConfig:
    image_size: int = 512
    patch_size: int = 8
    dim: int = 768
    layers: int = 24
    head_dim: int = 64
    inter_multi: int = 3
    vi_rms_norm: bool = True
    to_kv_mode: str = "shared_last"

    @classmethod
    def for_domain(cls, domain):
        sizes = {"objaverse": 1024, "dl3dv": 512}
        if domain.lower() not in sizes:
            raise ValueError("domain must be 'objaverse' or 'dl3dv'")
        return cls(image_size=sizes[domain.lower()])


def compute_rays(intrinsics, c2w, height, width):
    """Pixel-unit [fx, fy, cx, cy], OpenCV camera-to-world; outputs [B,V,3,H,W]."""
    batch, views = intrinsics.shape[:2]
    intrinsics = intrinsics.reshape(batch * views, 4)
    c2w = c2w.reshape(batch * views, 4, 4)
    y, x = torch.meshgrid(
        torch.arange(height, device=c2w.device, dtype=c2w.dtype),
        torch.arange(width, device=c2w.device, dtype=c2w.dtype),
        indexing="ij",
    )
    x = (x.flatten()[None] + 0.5 - intrinsics[:, 2:3]) / intrinsics[:, 0:1]
    y = (y.flatten()[None] + 0.5 - intrinsics[:, 3:4]) / intrinsics[:, 1:2]
    directions = c2w[:, :3, :3] @ torch.stack((x, y, torch.ones_like(x)), dim=1)
    directions = directions / directions.norm(dim=1, keepdim=True)
    origins = c2w[:, :3, 3:4].expand(-1, -1, height * width)
    shape = (batch, views, 3, height, width)
    return origins.reshape(shape), directions.reshape(shape)


class _Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        b, v, c, h, w = x.shape
        p = self.patch_size
        return x.reshape(b, v, c, h // p, p, w // p, p).permute(
            0, 1, 3, 5, 4, 6, 2
        ).reshape(b, v * (h // p) * (w // p), p * p * c)


class _Unpatchify(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.grid = image_size // patch_size
        self.patch_size = patch_size

    def forward(self, x):
        b, _, _ = x.shape
        p, g = self.patch_size, self.grid
        return x.reshape(b, -1, g, g, p, p, 3).permute(
            0, 1, 6, 2, 4, 3, 5
        ).reshape(b, -1, 3, g * p, g * p)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.head_dim = config.head_dim
        self.heads = d // self.head_dim
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_kv = nn.Linear(d, 2 * d, bias=False)
        self.to_out = nn.Linear(d, d, bias=False)

    def keys_values(self, x):
        kv = self.to_kv(x).unflatten(-1, (2, self.heads, self.head_dim))
        k, v = kv.unbind(dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x, cache):
        q = self.to_q(x).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        k, v = self.keys_values(x)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        # A view is one noncausal chunk; only preceding context views are cached.
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.to_out(x.transpose(1, 2).flatten(-2))


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, hidden = config.dim, config.dim * config.inter_multi
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_attn = nn.RMSNorm(config.dim, eps=1e-5)
        self.self_attention = _Attention(config)
        self.ln_mlp = nn.RMSNorm(config.dim, eps=1e-5)
        self.mlp = _MLP(config)

    def forward(self, x, cache):
        memory_in = self.ln_attn(x)
        x = x + self.self_attention(memory_in, cache)
        return x + self.mlp(self.ln_mlp(x)), memory_in


class _MemoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.layers)])

    def forward(self, x, cache, update):
        for block in self.blocks:
            layer_cache = cache
            if self.config.to_kv_mode == "per_layer" and cache is not None:
                layer_cache = block.self_attention.keys_values(cache)
            x, memory_in = block(x, layer_cache)
        if update:
            if self.config.vi_rms_norm:
                memory_in = F.rms_norm(memory_in, (self.config.dim,), eps=1e-5)
            if self.config.to_kv_mode == "per_layer":
                # The reference projects the same final-layer features separately
                # for each layer. Store features once and project them on read.
                cache = memory_in if cache is None else torch.cat((cache, memory_in), dim=1)
            else:
                k, v = self.blocks[-1].self_attention.keys_values(memory_in)
                if cache is not None:
                    k = torch.cat((cache[0], k), dim=2)
                    v = torch.cat((cache[1], v), dim=2)
                cache = (k, v)
        return x, cache


class NVSModel(nn.Module):
    """24-layer released NVS architecture; no training data or target RGB is needed."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or NVSConfig()
        c = self.config
        if c.layers < 1 or c.dim % c.head_dim or c.image_size % c.patch_size:
            raise ValueError("Invalid layer count, attention width, or image/patch size")
        if c.to_kv_mode not in ("shared_last", "per_layer"):
            raise ValueError("to_kv_mode must be 'shared_last' or 'per_layer'")
        self.image_tokenizer = nn.Sequential(
            _Patchify(c.patch_size), nn.Linear(12 * c.patch_size**2, c.dim, bias=False)
        )
        self.transformer_input_layernorm = nn.LayerNorm(c.dim, bias=False)
        self.transformer_blocks = _MemoryModel(c)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(c.dim, bias=False),
            nn.Linear(c.dim, c.patch_size**2 * 3, bias=False),
            _Unpatchify(c.image_size, c.patch_size),
            nn.Sigmoid(),
        )

    @classmethod
    def from_pretrained(cls, source, *, config=None, device="cuda", dtype=torch.bfloat16):
        """Load original model.pt with strict key/shape checks, from disk or the HF Hub."""
        source = str(source)
        presets = {
            "ZefanCai/LastKV-NVS-Objaverse": "objaverse",
            "ZefanCai/LastKV-NVS-DL3DV": "dl3dv",
        }
        if source in presets:
            from huggingface_hub import hf_hub_download

            config = config or NVSConfig.for_domain(presets[source])
            source = hf_hub_download(source, "model.pt")
        else:
            if config is None:
                raise ValueError("A local checkpoint requires config=NVSConfig.for_domain(...)")
            path = Path(source)
            source = str(path / "model.pt" if path.is_dir() else path)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        with torch.device("meta"):
            model = cls(config)
        model.load_state_dict(checkpoint["model"], strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval()

    def _tokens(self, intrinsics, c2w, image=None):
        size = self.config.image_size
        origins, directions = compute_rays(intrinsics, c2w, size, size)
        rgb = torch.zeros_like(origins) if image is None else image * 2.0 - 1.0
        posed = torch.cat((origins, directions, torch.cross(origins, directions, dim=2), rgb), dim=2)
        dtype = self.image_tokenizer[1].weight.dtype
        return self.transformer_input_layernorm(self.image_tokenizer(posed.to(dtype)))

    @torch.inference_mode()
    def render(self, context_images, context_intrinsics, context_c2w, target_intrinsics, target_c2w):
        """Render [B,T,3,H,W] in [0,1] from [B,C,3,H,W] context and calibrated cameras."""
        if context_images.ndim != 5:
            raise ValueError("context_images must be [B,C,3,H,W]")
        batch, count, channels, height, width = context_images.shape
        size = self.config.image_size
        if count < 1 or channels != 3 or (height, width) != (size, size):
            raise ValueError(f"At least one RGB context view of size {size}x{size} is required")
        target_count = target_intrinsics.shape[1]
        if target_count < 1:
            raise ValueError("At least one target camera is required")
        pairs = ((context_intrinsics, (batch, count, 4)),
                 (context_c2w, (batch, count, 4, 4)),
                 (target_intrinsics, (batch, target_count, 4)),
                 (target_c2w, (batch, target_count, 4, 4)))
        if any(tuple(x.shape) != shape for x, shape in pairs):
            raise ValueError("Camera shapes must agree with image batch and view counts")
        device = next(self.parameters()).device
        context_images = context_images.to(device=device, dtype=torch.float32)
        context_intrinsics, context_c2w, target_intrinsics, target_c2w = (
            value.to(device=device, dtype=torch.float32) for value, _ in pairs
        )
        if (context_intrinsics[..., :2] <= 0).any() or (target_intrinsics[..., :2] <= 0).any():
            raise ValueError("Focal lengths must be positive")
        # Camera/ray math remains FP32, including when called inside autocast.
        autocast = torch.autocast(device.type, enabled=False) if device.type in ("cuda", "cpu") else nullcontext()
        with autocast:
            cache = None
            for i in range(count):
                tokens = self._tokens(context_intrinsics[:, i:i+1], context_c2w[:, i:i+1], context_images[:, i:i+1])
                _, cache = self.transformer_blocks(tokens, cache, update=True)
            outputs = []
            for i in range(target_count):
                tokens = self._tokens(target_intrinsics[:, i:i+1], target_c2w[:, i:i+1])
                tokens, _ = self.transformer_blocks(tokens, cache, update=False)
                outputs.append(self.image_token_decoder(tokens).float())
        return torch.cat(outputs, dim=1)

    forward = render
