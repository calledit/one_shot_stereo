import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taehv"))
from taehv import TAEHV

from data.dataset import (
    StereoDisocclusionDataset, collate_fn,
    TRAINING_DATA_ROOT,
    LATENT_T, LATENT_C, LATENT_H, LATENT_W,
    TOKEN_T, TOKEN_H, TOKEN_W, TOKEN_DIM,
)

VAE_CHECKPOINT = os.path.join("taehv", "taew2_1.pth")


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


@torch.no_grad()
def decode(vae, latents, device):
    return vae.decode_video(
        latents.to(device),
        parallel=True, show_progress_bar=False,
    )


def train(
    data_root=TRAINING_DATA_ROOT,
    batch_size=4,
    num_workers=2,
    device_str="cuda",
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

    # TODO: model = OneShotStereoNet(...).to(device, torch.bfloat16)
    # TODO: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step = 0
    for batch in loader:
        if max_steps is not None and step >= max_steps:
            break

        input_latents = encode(vae, batch["input"], device)   # (B, 7, 16, 60, 104)
        gt_latents    = encode(vae, batch["gt"],    device)
        token_mask    = batch["token_mask"].to(device)        # (B, 7, 15, 26)

        # TODO: pred = model(input_latents, token_mask)
        # TODO: loss = latent_l1(pred, gt_latents, token_mask)
        # TODO: loss.backward(); optimizer.step(); optimizer.zero_grad()

        step += 1
        print(f"step {step:5d} | masked tokens {token_mask.sum().item()}")

    print("Done.")


if __name__ == "__main__":
    train(
        batch_size=2,
        num_workers=2,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=5,
    )
