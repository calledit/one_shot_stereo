"""
train.py — One Shot Stereo Disocclusion Infill Network
Training script: data loading, preprocessing, VAE encoding, training loop scaffold.
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taehv"))
from taehv import TAEHV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_W = 832
OUTPUT_H = 480
FRAMES_PER_CLIP = 25

# TAEW2_1 compression ratios (measured empirically)
VAE_SPATIAL = 8          # 832→104, 480→60
VAE_TEMPORAL = 4         # 25 frames → 7 latents (1 + 24/4)
PATCH_SIZE = 4           # 4×4 spatial patching on latents → matches design doc token count

LATENT_C = 16
LATENT_T = 7             # (FRAMES_PER_CLIP - 1) // VAE_TEMPORAL + 1
LATENT_H = OUTPUT_H // VAE_SPATIAL   # 60
LATENT_W = OUTPUT_W // VAE_SPATIAL   # 104

TOKEN_T = LATENT_T                    # 7
TOKEN_H = LATENT_H // PATCH_SIZE      # 15
TOKEN_W = LATENT_W // PATCH_SIZE      # 26
TOKEN_DIM = PATCH_SIZE * PATCH_SIZE * LATENT_C  # 256
PIXELS_PER_TOKEN = VAE_SPATIAL * PATCH_SIZE      # 32

TRAINING_DATA_ROOT = "training_data"
VAE_CHECKPOINT = os.path.join("taehv", "taew2_1.pth")

# ---------------------------------------------------------------------------
# Token mask computation
# ---------------------------------------------------------------------------

def compute_token_mask(hole_mask):
    """
    Convert per-frame pixel-space hole masks to per-token binary flags.

    hole_mask : (25, OUTPUT_H, OUTPUT_W) bool
    Returns   : (TOKEN_T, TOKEN_H, TOKEN_W) bool

    Temporal grouping mirrors TAEW2_1 4× compression:
        t=0 → frame 0 only
        t=k → frames 4k-3 … 4k  (k ≥ 1)
    A token is True if ANY covered pixel is a hole in ANY covered frame.
    """
    assert hole_mask.shape == (FRAMES_PER_CLIP, OUTPUT_H, OUTPUT_W)
    temporal_groups = [
        hole_mask[0:1],
        hole_mask[1:5],
        hole_mask[5:9],
        hole_mask[9:13],
        hole_mask[13:17],
        hole_mask[17:21],
        hole_mask[21:25],
    ]
    token_mask = np.zeros((TOKEN_T, TOKEN_H, TOKEN_W), dtype=bool)
    for t, group in enumerate(temporal_groups):
        spatial_or = group.any(axis=0)                                    # (H, W)
        spatial_or = spatial_or.reshape(TOKEN_H, PIXELS_PER_TOKEN,
                                        TOKEN_W, PIXELS_PER_TOKEN)
        token_mask[t] = spatial_or.any(axis=(1, 3))
    return token_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StereoDisocclusionDataset(Dataset):
    """
    Loads training clips from training_data/<hex>/<name>_f<N>_gt.mp4 + .npz.
    Returns pixel-space tensors ready for VAE encoding.
    """

    def __init__(self, root=TRAINING_DATA_ROOT, augment=True):
        self.augment = augment
        self.clips = self._scan(root)
        print(f"Found {len(self.clips)} valid clips under {root}")

    def _scan(self, root):
        """
        File layout:
          <name>.txt           — status (OK or rejection reason)
          <name>_f<N>.npz      — hole mask
          <name>_f<N>_gt.mp4   — ground truth video
        Scan by finding npz files, derive txt from the name-only prefix.
        """
        import re
        clips = []
        for subfolder in "0123456789abcdef":
            folder = os.path.join(root, subfolder)
            if not os.path.isdir(folder):
                continue
            for npz_path in glob.glob(os.path.join(folder, "*_f[0-9]*.npz")):
                gt_path = npz_path[:-4] + "_gt.mp4"
                if not os.path.exists(gt_path):
                    continue
                # <name>_f<NNNNNN>.npz → <name>.txt
                txt_path = re.sub(r"_f\d+\.npz$", ".txt", npz_path)
                if not os.path.exists(txt_path):
                    continue
                with open(txt_path) as f:
                    status = f.read().strip()
                if status.startswith("OK"):
                    clips.append((gt_path, npz_path))
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        gt_path, npz_path = self.clips[idx]

        hole_mask_full = np.load(npz_path)["hole_mask"]   # (N, H, W) bool
        N = hole_mask_full.shape[0]
        gt_full = self._load_video(gt_path, N)             # (N, H, W, 3) uint8

        # Select 25-frame window (50-frame clips offer 3 windows)
        start = 0
        if N >= 50 and self.augment:
            choice = random.randint(0, 2)
            if choice == 1:
                start = (N - FRAMES_PER_CLIP) // 2
            elif choice == 2:
                start = N - FRAMES_PER_CLIP

        gt = gt_full[start:start + FRAMES_PER_CLIP].copy()          # (25, H, W, 3)
        hole_mask = hole_mask_full[start:start + FRAMES_PER_CLIP]    # (25, H, W)

        # Augmentation: horizontal flip and/or time reversal
        if self.augment:
            if random.random() < 0.5:
                gt = gt[:, :, ::-1, :].copy()
                hole_mask = hole_mask[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[::-1].copy()
                hole_mask = hole_mask[::-1].copy()

        # Build network input: GT with holes painted green
        inp = gt.copy()

        # 10% chance: first frame gets blurred GT fill (temporal consistency hint)
        if random.random() < 0.1:
            frame0 = inp[0].astype(np.float32)
            blurred = cv2.GaussianBlur(frame0, (21, 21), 0)
            frame0[hole_mask[0]] = blurred[hole_mask[0]]
            inp[0] = frame0.astype(np.uint8)
        else:
            inp[0][hole_mask[0]] = [0, 255, 0]

        for i in range(1, FRAMES_PER_CLIP):
            inp[i][hole_mask[i]] = [0, 255, 0]

        token_mask = compute_token_mask(hole_mask)   # (7, 30, 52) bool

        # Convert to float [0, 1] NTCHW for TAEW2_1
        def to_ntchw(frames_hwc):
            t = torch.from_numpy(frames_hwc).float().div_(255.0)   # (T, H, W, 3)
            return t.permute(0, 3, 1, 2).unsqueeze(0)             # (1, T, C, H, W)

        return {
            "input_ntchw": to_ntchw(inp),                           # (1, 25, 3, H, W)
            "gt_ntchw":    to_ntchw(gt),                            # (1, 25, 3, H, W)
            "hole_mask":   torch.from_numpy(hole_mask),             # (25, H, W) bool
            "token_mask":  torch.from_numpy(token_mask),            # (7, 30, 52) bool
        }

    @staticmethod
    def _load_video(path, expected_frames):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < expected_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) != expected_frames:
            raise RuntimeError(
                f"Expected {expected_frames} frames from {path}, got {len(frames)}")
        return np.stack(frames)   # (N, H, W, 3) uint8


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
def encode_batch(vae, batch_ntchw, device):
    """
    Encode a batch of pixel videos to latents.

    batch_ntchw : (B, 25, 3, H, W) float32 [0,1]
    Returns     : (B, 7, 16, 60, 104) float16 latents
    """
    x = batch_ntchw.to(device, torch.float16)
    return vae.encode_video(x, parallel=True, show_progress_bar=False)


@torch.no_grad()
def decode_batch(vae, latents, device):
    """
    Decode latents back to pixel videos.

    latents : (B, 7, 16, 60, 104) float16
    Returns : (B, 25, 3, H, W) float16 [0,1]
    """
    return vae.decode_video(latents.to(device), parallel=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Collate: squeeze the batch-1 dim added per-sample and stack
# ---------------------------------------------------------------------------

def collate_fn(samples):
    return {
        "input_latents": torch.cat([s["input_ntchw"] for s in samples], dim=0),  # (B,25,3,H,W)
        "gt_latents":    torch.cat([s["gt_ntchw"]    for s in samples], dim=0),
        "hole_mask":     torch.stack([s["hole_mask"]   for s in samples]),        # (B,25,H,W)
        "token_mask":    torch.stack([s["token_mask"]  for s in samples]),        # (B,7,30,52)
    }


# ---------------------------------------------------------------------------
# Training loop scaffold
# ---------------------------------------------------------------------------

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
    print(f"VAE ready. Latent shape for one clip: "
          f"({LATENT_T}, {LATENT_C}, {LATENT_H}, {LATENT_W})")
    print(f"Token grid: {TOKEN_T}×{TOKEN_H}×{TOKEN_W} = "
          f"{TOKEN_T * TOKEN_H * TOKEN_W} tokens of dim {TOKEN_DIM}+1")

    # TODO: instantiate the network here
    # model = OneShotStereoNet(...).to(device, torch.bfloat16)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step = 0
    for batch in loader:
        if max_steps is not None and step >= max_steps:
            break

        # Encode inputs and GT to latent space
        input_latents = encode_batch(vae, batch["input_latents"], device)  # (B,7,16,60,104)
        gt_latents    = encode_batch(vae, batch["gt_latents"],    device)

        token_mask = batch["token_mask"].to(device)   # (B, 7, 30, 52) bool

        # TODO: forward pass
        # pred_latents = model(input_latents, token_mask)

        # TODO: compute losses
        # latent_l1_loss = (pred_latents - gt_latents).abs()[hole_latent_mask].mean()
        # loss = latent_l1_loss
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        step += 1
        print(f"step {step:5d} | "
              f"input_latents {tuple(input_latents.shape)} | "
              f"gt_latents {tuple(gt_latents.shape)} | "
              f"masked_tokens {token_mask.sum().item()}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(
        data_root=TRAINING_DATA_ROOT,
        batch_size=2,
        num_workers=2,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=5,
    )
