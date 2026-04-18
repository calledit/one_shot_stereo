import os
import sys
import glob
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
from model.network import OneShotStereoNet

VAE_CHECKPOINT  = os.path.join("taehv", "taew2_1.pth")
CHECKPOINT_DIR  = "checkpoints"
SAVE_EVERY      = 500    # steps between auto-saves
KEEP_LAST       = 5      # number of recent checkpoints to keep on disk


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

def latent_l1_loss(pred, target, token_mask):
    """L1 loss on latents inside hole regions only."""
    latent_mask = (token_mask
                   .unsqueeze(2)
                   .repeat(1, 1, LATENT_C, 1, 1)
                   .repeat_interleave(PATCH_SIZE, dim=3)
                   .repeat_interleave(PATCH_SIZE, dim=4))   # (B, T, C, H, W)
    return (pred - target).abs()[latent_mask].mean()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(step):
    return os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")


def save_checkpoint(net, optimizer, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(step)
    torch.save({
        "step":      step,
        "model":     net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  saved {path}")
    _prune_checkpoints()


def load_checkpoint(path, net, optimizer, device):
    state = torch.load(path, map_location=device)
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
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

    net = OneShotStereoNet().to(device, torch.bfloat16)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    print(f"Network parameters: {sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    # Resume: explicit path > auto-latest > fresh start
    step = 0
    ckpt = resume if resume is not None else latest_checkpoint()
    if ckpt is not None:
        step = load_checkpoint(ckpt, net, optimizer, device)
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
        loss         = latent_l1_loss(pred_latents, gt_latents.to(torch.bfloat16), token_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        print(f"step {step:6d} | loss {loss.item():.4f} | "
              f"masked tokens {token_mask.sum().item()}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(net, optimizer, step)

    # Always save at end of run
    save_checkpoint(net, optimizer, step)
    print("Done.")


if __name__ == "__main__":
    train(
        batch_size=2,
        num_workers=2,
        lr=1e-4,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=5,
    )
