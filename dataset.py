import os
import re
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

OUTPUT_W = 832
OUTPUT_H = 480
FRAMES_PER_CLIP = 25

VAE_SPATIAL = 8
VAE_TEMPORAL = 4
PATCH_SIZE = 4

LATENT_C = 16
LATENT_T = 7
LATENT_H = OUTPUT_H // VAE_SPATIAL    # 60
LATENT_W = OUTPUT_W // VAE_SPATIAL    # 104

TOKEN_T = LATENT_T                    # 7
TOKEN_H = LATENT_H // PATCH_SIZE      # 15
TOKEN_W = LATENT_W // PATCH_SIZE      # 26
TOKEN_DIM = PATCH_SIZE * PATCH_SIZE * LATENT_C   # 256
PIXELS_PER_TOKEN = VAE_SPATIAL * PATCH_SIZE       # 32

TRAINING_DATA_ROOT = "training_data"


def compute_token_mask(hole_mask):
    """
    hole_mask : (25, H, W) bool
    Returns   : (TOKEN_T, TOKEN_H, TOKEN_W) bool

    A token is True if any pixel it covers is a hole in any frame it covers.
    Temporal grouping: t=0 → frame 0; t=k → frames 4k-3…4k (mirrors VAE 4× temporal compression).
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
        spatial_or = group.any(axis=0)
        spatial_or = spatial_or.reshape(TOKEN_H, PIXELS_PER_TOKEN, TOKEN_W, PIXELS_PER_TOKEN)
        token_mask[t] = spatial_or.any(axis=(1, 3))
    return token_mask


class StereoDisocclusionDataset(Dataset):
    """
    Loads clips from training_data/<hex>/<name>_f<N>_gt.mp4 + <name>_f<N>.npz.

    Returns per-clip dicts with pixel-space NTCHW tensors [0,1] and masks.
    File layout:
        <name>.txt         — status line (must start with "OK")
        <name>_f<N>.npz    — hole_mask array, shape (N, H, W) bool
        <name>_f<N>_gt.mp4 — ground-truth video, N=25 or 50 frames
    """

    def __init__(self, root=TRAINING_DATA_ROOT, augment=True):
        self.augment = augment
        self.clips = self._scan(root)
        print(f"Found {len(self.clips)} valid clips under {root}")

    def _scan(self, root):
        clips = []
        for subfolder in "0123456789abcdef":
            folder = os.path.join(root, subfolder)
            if not os.path.isdir(folder):
                continue
            for npz_path in glob.glob(os.path.join(folder, "*_f[0-9]*.npz")):
                gt_path = npz_path[:-4] + "_gt.mp4"
                if not os.path.exists(gt_path):
                    continue
                txt_path = re.sub(r"_f\d+\.npz$", ".txt", npz_path)
                if not os.path.exists(txt_path):
                    continue
                with open(txt_path) as f:
                    if f.read().strip().startswith("OK"):
                        clips.append((gt_path, npz_path))
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        gt_path, npz_path = self.clips[idx]

        hole_mask_full = np.load(npz_path)["hole_mask"]    # (N, H, W) bool
        N = hole_mask_full.shape[0]
        gt_full = _load_video(gt_path, N)                  # (N, H, W, 3) uint8

        # 50-frame clips: pick start, middle, or end window
        start = 0
        if N >= 50 and self.augment:
            choice = random.randint(0, 2)
            if choice == 1:
                start = (N - FRAMES_PER_CLIP) // 2
            elif choice == 2:
                start = N - FRAMES_PER_CLIP

        gt        = gt_full[start:start + FRAMES_PER_CLIP].copy()
        hole_mask = hole_mask_full[start:start + FRAMES_PER_CLIP].copy()

        if self.augment:
            if random.random() < 0.5:
                gt        = gt[:, :, ::-1, :].copy()
                hole_mask = hole_mask[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt        = gt[::-1].copy()
                hole_mask = hole_mask[::-1].copy()

        inp = gt.copy()

        # 10% chance: first frame gets blurred GT fill instead of green
        # (trains the network to accept a filled reference for temporal consistency)
        if random.random() < 0.1:
            f0 = inp[0].astype(np.float32)
            blurred = cv2.GaussianBlur(f0, (21, 21), 0)
            f0[hole_mask[0]] = blurred[hole_mask[0]]
            inp[0] = f0.astype(np.uint8)
        else:
            inp[0][hole_mask[0]] = [0, 255, 0]

        for i in range(1, FRAMES_PER_CLIP):
            inp[i][hole_mask[i]] = [0, 255, 0]

        token_mask = compute_token_mask(hole_mask)

        return {
            "input":      _to_ntchw(inp),                         # (1, 25, 3, H, W) float32
            "gt":         _to_ntchw(gt),                          # (1, 25, 3, H, W) float32
            "hole_mask":  torch.from_numpy(hole_mask),            # (25, H, W) bool
            "token_mask": torch.from_numpy(token_mask),           # (7, 15, 26) bool
        }


def collate_fn(samples):
    return {
        "input":      torch.cat([s["input"]      for s in samples]),   # (B, 25, 3, H, W)
        "gt":         torch.cat([s["gt"]         for s in samples]),
        "hole_mask":  torch.stack([s["hole_mask"]  for s in samples]), # (B, 25, H, W)
        "token_mask": torch.stack([s["token_mask"] for s in samples]), # (B, 7, 15, 26)
    }


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
        raise RuntimeError(f"Expected {expected_frames} frames from {path}, got {len(frames)}")
    return np.stack(frames)


def _to_ntchw(frames_hwc):
    t = torch.from_numpy(frames_hwc).float().div_(255.0)  # (T, H, W, 3)
    return t.permute(0, 3, 1, 2).unsqueeze(0)             # (1, T, 3, H, W)
