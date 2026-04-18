import os
import sys
import glob
import random
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taehv"))
from taehv import TAEHV

from data.dataset import (
    StereoDisocclusionDataset, collate_fn,
    TRAINING_DATA_ROOT, PATCH_SIZE,
    LATENT_T, LATENT_C, LATENT_H, LATENT_W,
    TOKEN_T, TOKEN_H, TOKEN_W, TOKEN_DIM,
)
from model.network import OneShotStereoNet, patchify
from model.discriminator import LatentDiscriminator, d_hinge, g_hinge

VAE_CHECKPOINT  = os.path.join("taehv", "taew2_1.pth")
CHECKPOINT_DIR  = "checkpoints"
SAVE_EVERY      = 500    # steps between auto-saves
KEEP_LAST       = 5      # number of recent checkpoints to keep on disk

LAMBDA_LATENT_IN  = 1.0   # latent L1 inside holes
LAMBDA_LATENT_OUT = 0.1   # latent L1 outside holes (cheap, every step)
LAMBDA_PIXEL_OUT  = 1.0   # pixel L1 outside holes (catches VAE non-linear shifts)
PIXEL_LOSS_PROB   = 0.1   # probability of computing pixel loss each step
LAMBDA_GAN        = 0.1   # generator GAN loss weight
GAN_START_STEP    = 10000 # step at which GAN loss is switched on


# ---------------------------------------------------------------------------
# VAE helpers
# ---------------------------------------------------------------------------

def load_vae(device):
    vae = TAEHV(checkpoint_path=VAE_CHECKPOINT).to(device, torch.float16)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, batch_ntchw, device):
    return vae.encode_video(
        batch_ntchw.to(device, torch.float16),
        parallel=True, show_progress_bar=False,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _latent_mask(token_mask):
    """Expand token mask to full latent resolution (B, T, C, H, W)."""
    return (token_mask
            .unsqueeze(2)
            .repeat(1, 1, LATENT_C, 1, 1)
            .repeat_interleave(PATCH_SIZE, dim=3)
            .repeat_interleave(PATCH_SIZE, dim=4))


def latent_l1_inside(pred, target, token_mask):
    """L1 on latents inside hole regions — drives fill quality."""
    mask = _latent_mask(token_mask)
    return (pred - target).abs()[mask].mean()


def latent_l1_outside(pred, target, token_mask):
    """L1 on latents outside hole regions — cheap every-step stability term."""
    mask = ~_latent_mask(token_mask)
    return (pred - target).abs()[mask].mean()


def pixel_l1_outside(vae, pred_latents, gt_pixels, hole_mask, device):
    """
    Pixel L1 outside holes — catches shifts that look small in latent space
    but are visible after the non-linear VAE decode.

    Gradient checkpointing means decode activations are recomputed during
    backward rather than stored alongside the network activations.

    pred_latents : (B, 7, 16, 60, 104) bfloat16
    gt_pixels    : (B, 25, 3, H, W)    float32 [0,1]
    hole_mask    : (B, 25, H, W)        bool
    """
    pred_pixels = vae.decode_video(
        pred_latents.to(torch.float16),
        parallel=True, show_progress_bar=False,
    )   # (B, 25, 3, H, W) float16

    outside = ~hole_mask.to(device)                         # (B, 25, H, W)
    outside = outside.unsqueeze(2).expand_as(pred_pixels)   # (B, 25, 3, H, W)
    return (pred_pixels.float() - gt_pixels.to(device).float()).abs()[outside].mean()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(step):
    return os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")


def save_checkpoint(net, optimizer, disc, disc_optimizer, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(step)
    torch.save({
        "step":           step,
        "model":          net.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "disc":           disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
    }, path)
    print(f"  saved {path}")
    _prune_checkpoints()


def load_checkpoint(path, net, optimizer, disc, disc_optimizer, device):
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    disc.load_state_dict(state["disc"])
    disc_optimizer.load_state_dict(state["disc_optimizer"])
    return state["step"]


def latest_checkpoint():
    """Return the path of the most recent checkpoint, or None."""
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    return paths[-1] if paths else None


def _prune_checkpoints():
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    for old in paths[:-KEEP_LAST]:
        os.remove(old)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_root=TRAINING_DATA_ROOT,
    batch_size=4,
    num_workers=2,
    lr=1e-4,
    device_str="cuda",
    resume=None,          # path to checkpoint; None = auto-resume from latest
    max_steps=None,
):
    device = torch.device(device_str)

    dataset = StereoDisocclusionDataset(root=data_root, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"Loading VAE from {VAE_CHECKPOINT} ...")
    vae = load_vae(device)
    print(f"Latent shape: ({LATENT_T}, {LATENT_C}, {LATENT_H}, {LATENT_W})")
    print(f"Token grid:   {TOKEN_T}×{TOKEN_H}×{TOKEN_W} = "
          f"{TOKEN_T * TOKEN_H * TOKEN_W} tokens of dim {TOKEN_DIM}+1")

    net  = OneShotStereoNet().to(device, torch.bfloat16)
    disc = LatentDiscriminator().to(device, torch.bfloat16)
    optimizer      = torch.optim.AdamW(net.parameters(),  lr=lr)
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=lr)
    print(f"Network parameters:       {sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M")

    # Resume: explicit path > auto-latest > fresh start
    step = 0
    ckpt = resume if resume is not None else latest_checkpoint()
    if ckpt is not None:
        step = load_checkpoint(ckpt, net, optimizer, disc, disc_optimizer, device)
        print(f"Resumed from {ckpt}  (step {step})")
    else:
        print("Starting from scratch")
    
    net.train()
    for batch in loader:
        if max_steps is not None and step >= max_steps:
            break

        with torch.no_grad():
            input_latents = encode(vae, batch["input"], device)
            gt_latents    = encode(vae, batch["gt"],    device)

        token_mask   = batch["token_mask"].to(device)
        pred_latents = net(input_latents.to(torch.bfloat16), token_mask)
        gt_bf16      = gt_latents.to(torch.bfloat16)

        mask_flat    = token_mask.reshape(token_mask.shape[0], -1)   # (B, N)
        pred_tokens  = patchify(pred_latents).detach()                # (B, N, 256) detached for D update
        gt_tokens    = patchify(gt_bf16)

        # --- Discriminator update (only after GAN_START_STEP) ---
        d_loss_val = None
        if step >= GAN_START_STEP:
            real_scores = disc(gt_tokens,   mask_flat)
            fake_scores = disc(pred_tokens, mask_flat)
            d_loss = d_hinge(real_scores, fake_scores)
            disc_optimizer.zero_grad()
            d_loss.backward()
            disc_optimizer.step()
            d_loss_val = d_loss.item()

        # --- Generator update ---
        pred_tokens_g = patchify(pred_latents)   # not detached — gradients flow to net

        loss_in  = LAMBDA_LATENT_IN  * latent_l1_inside(pred_latents,  gt_bf16, token_mask)
        loss_out = LAMBDA_LATENT_OUT * latent_l1_outside(pred_latents, gt_bf16, token_mask)
        loss     = loss_in + loss_out

        pix_loss_val = None
        if random.random() < PIXEL_LOSS_PROB:
            pix = LAMBDA_PIXEL_OUT * pixel_l1_outside(
                vae, pred_latents, batch["gt"], batch["hole_mask"], device)
            loss = loss + pix
            pix_loss_val = pix.item()

        g_loss_val = None
        if step >= GAN_START_STEP:
            g_loss = LAMBDA_GAN * g_hinge(disc(pred_tokens_g, mask_flat))
            loss = loss + g_loss
            g_loss_val = g_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        pix_str = f" pix {pix_loss_val:.4f}" if pix_loss_val is not None else ""
        gan_str = (f" | D {d_loss_val:.4f} G {g_loss_val:.4f}"
                   if g_loss_val is not None else "")
        print(f"step {step:6d} | loss {loss.item():.4f} "
              f"(in {loss_in.item():.4f} out {loss_out.item():.4f}{pix_str}){gan_str} | "
              f"masked tokens {token_mask.sum().item()}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(net, optimizer, disc, disc_optimizer, step)

    # Always save at end of run
    save_checkpoint(net, optimizer, disc, disc_optimizer, step)
    print("Done.")


if __name__ == "__main__":
    train(
        batch_size=2,
        num_workers=2,
        lr=1e-4,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=5,
    )
