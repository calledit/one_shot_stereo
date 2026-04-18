import math
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
LOG_FILE        = "training_log.csv"
SAVE_EVERY      = 500     # steps between auto-saves
KEEP_LAST       = 5       # number of recent checkpoints to keep on disk

# LR schedule
WARMUP_STEPS    = 1_000
LR_DECAY_STEPS  = 200_000
LR_MIN_RATIO    = 0.1     # LR decays to lr * LR_MIN_RATIO

GRAD_CLIP_NORM  = 1.0

# Loss weights — static
LAMBDA_LATENT_OUT = 0.1   # latent L1 outside holes (cheap, every step)
LAMBDA_PIXEL_OUT  = 1.0   # pixel L1 outside holes
PIXEL_LOSS_PROB   = 0.1   # probability of computing pixel loss each step

# Loss weights — dynamic (shift from L1 toward GAN as training progresses)
GAN_START_STEP        = 10_000
GAN_RAMP_STEPS        = 20_000
LAMBDA_LATENT_IN_INIT = 1.0   # λ_latent_in at start and before GAN kicks in
LAMBDA_LATENT_IN_MIN  = 0.3   # λ_latent_in after full GAN ramp
LAMBDA_GAN_MAX        = 0.5   # λ_gan after full ramp


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
# Loss scheduling
# ---------------------------------------------------------------------------

def get_loss_weights(step):
    """Return (lambda_latent_in, lambda_gan) for the current step."""
    if step < GAN_START_STEP:
        return LAMBDA_LATENT_IN_INIT, 0.0
    t = min(1.0, (step - GAN_START_STEP) / GAN_RAMP_STEPS)
    lam_in  = LAMBDA_LATENT_IN_INIT + t * (LAMBDA_LATENT_IN_MIN - LAMBDA_LATENT_IN_INIT)
    lam_gan = t * LAMBDA_GAN_MAX
    return lam_in, lam_gan


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

def _lr_lambda(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    t = min(1.0, (step - WARMUP_STEPS) / LR_DECAY_STEPS)
    return LR_MIN_RATIO + (1 - LR_MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * t))


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _latent_mask(token_mask):
    """Expand token mask to full latent resolution (B, T, C, H, W)."""
    return (token_mask
            .unsqueeze(2)
            .repeat(1, 1, LATENT_C, 1, 1)
            .repeat_interleave(PATCH_SIZE, dim=3)
            .repeat_interleave(PATCH_SIZE, dim=4))


def latent_l1_inside(pred, target, token_mask):
    mask = _latent_mask(token_mask)
    return (pred - target).abs()[mask].mean()


def latent_l1_outside(pred, target, token_mask):
    mask = ~_latent_mask(token_mask)
    return (pred - target).abs()[mask].mean()


def pixel_l1_outside(vae, pred_latents, gt_pixels, hole_mask, device):
    """
    Pixel L1 outside holes — catches shifts that look small in latent space
    but are visible after the non-linear VAE decode.

    pred_latents : (B, 7, 16, 60, 104) bfloat16
    gt_pixels    : (B, 25, 3, H, W)    float32 [0,1]
    hole_mask    : (B, 25, H, W)        bool
    """
    pred_pixels = vae.decode_video(
        pred_latents.to(torch.float16),
        parallel=True, show_progress_bar=False,
    )   # (B, 25, 3, H, W) float16

    outside = ~hole_mask.to(device)
    outside = outside.unsqueeze(2).expand_as(pred_pixels)
    return (pred_pixels.float() - gt_pixels.to(device).float()).abs()[outside].mean()


# ---------------------------------------------------------------------------
# Loss logging
# ---------------------------------------------------------------------------

def log_step(step, total, loss_in, loss_out, pix, d_loss, g_loss, lr, lam_in, lam_gan):
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if write_header:
            f.write("step,total,loss_in,loss_out,pix,d_loss,g_loss,lr,lam_in,lam_gan\n")
        def fmt(v):
            return f"{v:.6f}" if v is not None else ""
        f.write(
            f"{step},{fmt(total)},{fmt(loss_in)},{fmt(loss_out)},{fmt(pix)},"
            f"{fmt(d_loss)},{fmt(g_loss)},{lr:.8f},{fmt(lam_in)},{fmt(lam_gan)}\n"
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(step):
    return os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")


def save_checkpoint(net, optimizer, scheduler, disc, disc_optimizer, disc_scheduler, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(step)
    torch.save({
        "step":           step,
        "model":          net.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "disc":           disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict(),
    }, path)
    print(f"  saved {path}")
    _prune_checkpoints()


def load_checkpoint(path, net, optimizer, scheduler, disc, disc_optimizer, disc_scheduler, device):
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    disc.load_state_dict(state["disc"])
    disc_optimizer.load_state_dict(state["disc_optimizer"])
    disc_scheduler.load_state_dict(state["disc_scheduler"])
    return state["step"]


def latest_checkpoint():
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

    dataset     = StereoDisocclusionDataset(root=data_root, augment=True)
    val_dataset = StereoDisocclusionDataset(
        root=data_root, augment=False,
        subfolders=StereoDisocclusionDataset.VAL_SUBFOLDERS,
    )
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
    optimizer      = torch.optim.AdamW(
        net.parameters(),  lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    disc_optimizer = torch.optim.AdamW(
        disc.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    scheduler      = make_scheduler(optimizer)
    disc_scheduler = make_scheduler(disc_optimizer)
    print(f"Network parameters:       {sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M")

    # Resume: explicit path > auto-latest > fresh start
    step = 0
    ckpt = resume if resume is not None else latest_checkpoint()
    if ckpt is not None:
        step = load_checkpoint(
            ckpt, net, optimizer, scheduler, disc, disc_optimizer, disc_scheduler, device)
        print(f"Resumed from {ckpt}  (step {step})")
    else:
        print("Starting from scratch")

    net.train()
    disc.train()
    done = False
    while not done:
        for batch in loader:
            if max_steps is not None and step >= max_steps:
                done = True
                break

            with torch.no_grad():
                input_latents = encode(vae, batch["input"], device)
                gt_latents    = encode(vae, batch["gt"],    device)

            token_mask   = batch["token_mask"].to(device)
            pred_latents = net(input_latents.to(torch.bfloat16), token_mask)
            gt_bf16      = gt_latents.to(torch.bfloat16)

            mask_flat   = token_mask.reshape(token_mask.shape[0], -1)
            pred_tokens = patchify(pred_latents).detach()   # detached for D update
            gt_tokens   = patchify(gt_bf16)

            lam_in, lam_gan = get_loss_weights(step)

            # --- Discriminator update (only after GAN_START_STEP) ---
            d_loss_val = None
            if step >= GAN_START_STEP:
                real_scores = disc(gt_tokens,   mask_flat)
                fake_scores = disc(pred_tokens, mask_flat)
                d_loss = d_hinge(real_scores, fake_scores)
                disc_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP_NORM)
                disc_optimizer.step()
                disc_scheduler.step()
                d_loss_val = d_loss.item()

            # --- Generator update ---
            pred_tokens_g = patchify(pred_latents)   # not detached — gradients flow to net

            loss_in  = lam_in            * latent_l1_inside(pred_latents,  gt_bf16, token_mask)
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
                g_loss = lam_gan * g_hinge(disc(pred_tokens_g, mask_flat))
                loss = loss + g_loss
                g_loss_val = g_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            step += 1
            current_lr = scheduler.get_last_lr()[0]

            log_step(step, loss.item(), loss_in.item(), loss_out.item(),
                     pix_loss_val, d_loss_val, g_loss_val, current_lr, lam_in, lam_gan)

            pix_str = f" pix {pix_loss_val:.4f}" if pix_loss_val is not None else ""
            gan_str = (f" | D {d_loss_val:.4f} G {g_loss_val:.4f}"
                       if g_loss_val is not None else "")
            print(f"step {step:6d} | loss {loss.item():.4f} "
                  f"(in {loss_in.item():.4f} out {loss_out.item():.4f}{pix_str}){gan_str} | "
                  f"lr {current_lr:.2e} lam_in {lam_in:.2f} lam_gan {lam_gan:.2f} | "
                  f"masked {token_mask.sum().item()}")

            if step % SAVE_EVERY == 0:
                save_checkpoint(net, optimizer, scheduler, disc, disc_optimizer, disc_scheduler, step)

    # Always save at end of run
    save_checkpoint(net, optimizer, scheduler, disc, disc_optimizer, disc_scheduler, step)
    print("Done.")


if __name__ == "__main__":
    train(
        batch_size=2,
        num_workers=1,
        lr=1e-4,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
    )
