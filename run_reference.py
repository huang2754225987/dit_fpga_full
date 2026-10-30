#!/usr/bin/env python3
"""Run the local Supra2-IMG PyTorch reference pipeline."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "model" / "model_final_ema.pt"
IMAGE_SIZE = 256
LATENT_SIZE = 32
LATENT_CHANNELS = 4
PATCH = 2
TOKEN_COUNT = (LATENT_SIZE // PATCH) ** 2
D_MODEL = 576
DEPTH = 14
HEADS = 9
MLP_RATIO = 4.0
TEXT_DIM = 768
MAX_CONTEXT = 128
TEXT_ENCODER = "google/flan-t5-base"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
VAE_SCALE = 0.18215


def patchify_latents(z: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    if z.ndim != 4:
        raise ValueError("latent input must have shape [batch, channels, height, width]")
    batch, channels, height, width = z.shape
    if patch < 1 or height % patch or width % patch:
        raise ValueError("latent height and width must be divisible by patch size")
    h, w = height // patch, width // patch
    return z.view(batch, channels, h, patch, w, patch).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(batch, h * w, channels * patch * patch)


def unpatchify_latents(tokens: torch.Tensor, latent_shape: tuple[int, ...],
                       patch: int = PATCH) -> torch.Tensor:
    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [batch, channels, height, width]")
    batch, channels, height, width = latent_shape
    if patch < 1 or height % patch or width % patch:
        raise ValueError("latent height and width must be divisible by patch size")
    h, w = height // patch, width // patch
    if tokens.shape != (batch, h * w, channels * patch * patch):
        raise ValueError("token tensor shape does not match requested latent shape")
    return tokens.view(batch, h, w, channels, patch, patch).permute(
        0, 3, 1, 4, 2, 5
    ).reshape(batch, channels, height, width)


def modulate(x: torch.Tensor, shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int,
                 ctx_dim: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if self.is_self:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None,
                ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(
                batch, tokens, 3, self.n_heads, self.head_dim
            )
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            if ctx is None:
                raise ValueError("cross-attention requires context")
            context_tokens = ctx.shape[1]
            q = self.q(x).view(
                batch, tokens, self.n_heads, self.head_dim
            ).transpose(1, 2)
            kv = self.kv(ctx).view(
                batch, context_tokens, 2, self.n_heads, self.head_dim
            )
            k = kv[:, :, 0].transpose(1, 2)
            v = kv[:, :, 1].transpose(1, 2)

        mask = ctx_mask.bool()[:, None, None, :] if ctx_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, ctx_dim=dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(c).chunk(6, dim=1)
        )
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa)
        )
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SupraDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = PATCH
        self.x_embed = nn.Linear(LATENT_CHANNELS * PATCH * PATCH, D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, TOKEN_COUNT, D_MODEL))
        self.t_embed = TimestepEmbedder(D_MODEL)
        self.ctx_proj = nn.Linear(TEXT_DIM, D_MODEL)
        self.blocks = nn.ModuleList(
            [DiTBlock(D_MODEL, HEADS, MLP_RATIO) for _ in range(DEPTH)]
        )
        self.final = FinalLayer(D_MODEL, LATENT_CHANNELS * PATCH * PATCH)

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = z.shape
        x = patchify_latents(z, self.patch)
        x = self.x_embed(x) + self.pos_embed
        c = self.t_embed(t)
        ctx = self.ctx_proj(ctx)
        for block in self.blocks:
            x = block(x, c, ctx, ctx_mask)
        x = self.final(x, c)
        return unpatchify_latents(x, (batch, LATENT_CHANNELS, height, width),
                                  self.patch)


def load_pipeline(device: torch.device):
    try:
        from diffusers import AutoencoderKL
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install diffusers and transformers."
        ) from exc

    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise SystemExit("Expected a checkpoint dictionary with a 'model' state")
    config = state.get("config", {})
    if config.get("patch", PATCH) != PATCH:
        raise SystemExit(f"Checkpoint patch size mismatch: {config.get('patch')}")
    model = SupraDiT().to(device).eval()
    model.load_state_dict(state["model"], strict=True)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    text_model = T5EncoderModel.from_pretrained(TEXT_ENCODER).to(device).eval()
    vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(device).eval()
    return state, model, tokenizer, text_model, vae


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    state, model, tokenizer, text_model, vae = load_pipeline(device)
    context_length = int(state.get("config", {}).get("ctx_len", MAX_CONTEXT))

    tokens = tokenizer(
        [args.prompt], padding="max_length", truncation=True,
        max_length=context_length, return_tensors="pt",
    ).to(device)
    ctx = text_model(**tokens).last_hidden_state.float()
    mask = tokens["attention_mask"].float()

    use_cfg = args.cfg > 1.0
    if use_cfg:
        cfg = state["config"]
        if "uncond_text" in cfg and "uncond_mask" in cfg:
            uncond_ctx = cfg["uncond_text"].to(device).float().unsqueeze(0)
            uncond_mask = cfg["uncond_mask"].to(device).float().unsqueeze(0)
        else:
            empty = tokenizer(
                [""], padding="max_length", truncation=True,
                max_length=context_length, return_tensors="pt",
            ).to(device)
            uncond_ctx = text_model(**empty).last_hidden_state.float()
            uncond_mask = empty["attention_mask"].float()
        ctx_all = torch.cat([ctx, uncond_ctx], dim=0)
        mask_all = torch.cat([mask, uncond_mask], dim=0)

    z = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device)
    dt = 1.0 / args.steps
    started = time.perf_counter()
    for step in range(args.steps):
        t = torch.full((1,), step * dt, device=device)
        if use_cfg:
            velocity = model(
                torch.cat([z, z], dim=0), torch.cat([t, t], dim=0),
                ctx_all, mask_all,
            ).float()
            cond, uncond = velocity.chunk(2, dim=0)
            velocity = uncond + args.cfg * (cond - uncond)
        else:
            velocity = model(z, t, ctx, mask).float()
        z = z + dt * velocity
        print(f"[sample] step {step + 1}/{args.steps}", flush=True)
    print(f"[sample] elapsed={time.perf_counter() - started:.1f}s", flush=True)

    image = vae.decode(z / VAE_SCALE).sample
    image = ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    image = image[0].permute(1, 2, 0).cpu().numpy()
    from PIL import Image
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    print(f"[done] {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a red fox in a snowy forest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "reference.png"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
