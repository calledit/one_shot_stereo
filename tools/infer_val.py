"""
infer_val.py — run inference on a sample of validation clips and save results to test/.

Usage:
    python infer_val.py [--n 10] [--checkpoint PATH] [--window start|middle|end]
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "taehv"))
sys.path.insert(0, ROOT)
from taehv import TAEHV

from data.dataset import (
    StereoDisocclusionDataset, compute_token_mask,
    FRAMES_PER_CLIP, OUTPUT_W, OUTPUT_H,
)
from model.network import OneShotStereoNet

VAE_CHECKPOINT = os.path.join(ROOT, "taehv", "taew2_1.pth")
VAL_ROOT       = os.path.join(ROOT, "training_data", StereoDisocclusionDataset.VAL_SUBFOLDERS)
TEST_DIR       = os.path.join(ROOT, "test")


def latest_checkpoint():
    paths = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "step_*.pt")))
    return paths[-1] if paths else None


def load_clip(npz_path, window):
    gt_path = npz_path[:-4] + "_gt.mp4"
    hole_mask_full = np.load(npz_path)["hole_mask"]   # (N, H, W)
    N = hole_mask_full.shape[0]

    cap = cv2.VideoCapture(gt_path)
    frames = []
    while len(frames) < N:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    if window == "start":
        start = 0
    elif window == "middle":
        start = (N - FRAMES_PER_CLIP) // 2
    else:
        start = N - FRAMES_PER_CLIP
    start = max(0, min(start, N - FRAMES_PER_CLIP))

    gt        = np.stack(frames)[start:start + FRAMES_PER_CLIP]
    hole_mask = hole_mask_full[start:start + FRAMES_PER_CLIP]
    return gt, hole_mask


def write_video(frames_rgb, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (OUTPUT_W, OUTPUT_H))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def to_ntchw(frames_hwc, device):
    t = torch.from_numpy(frames_hwc).float().div_(255.0)
    return t.permute(0, 3, 1, 2).unsqueeze(0).to(device, torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=10,    help="Number of clips to process")
    parser.add_argument("--checkpoint", default=None,            help="Checkpoint path (default: latest)")
    parser.add_argument("--window",     default="start",         choices=["start", "middle", "end"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find validation clips
    npz_files = glob.glob(os.path.join(VAL_ROOT, "*_f[0-9]*.npz"))
    npz_files = [p for p in npz_files
                 if os.path.exists(p[:-4] + "_gt.mp4")]
    if not npz_files:
        print(f"No validation clips found under {VAL_ROOT}")
        sys.exit(1)

    random.shuffle(npz_files)
    npz_files = npz_files[:args.n]
    print(f"Processing {len(npz_files)} validation clips ...")

    print("Loading VAE ...")
    vae = TAEHV(checkpoint_path=VAE_CHECKPOINT).to(device, torch.float16)
    vae.eval()

    print("Loading network ...")
    net = OneShotStereoNet().to(device, torch.bfloat16)
    resolved = args.checkpoint or latest_checkpoint()
    if resolved:
        ckpt = torch.load(resolved, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model"])
        step = ckpt.get("step", 0)
        print(f"  weights from {resolved} (step {step})")
    else:
        step = 0
        print("  no checkpoint — random weights")
    net.eval()

    os.makedirs(TEST_DIR, exist_ok=True)

    for i, npz_path in enumerate(npz_files):
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"[{i+1}/{len(npz_files)}] {stem}")

        gt, hole_mask = load_clip(npz_path, args.window)
        coverage = hole_mask.mean() * 100

        masked = gt.copy()
        for f in range(FRAMES_PER_CLIP):
            masked[f][hole_mask[f]] = [0, 255, 0]

        token_mask = torch.from_numpy(compute_token_mask(hole_mask)).unsqueeze(0).to(device)

        with torch.no_grad():
            input_latents = vae.encode_video(
                to_ntchw(masked, device), parallel=True, show_progress_bar=False,
            )
            pred_latents = net(input_latents.to(torch.bfloat16), token_mask)
            pred_frames  = vae.decode_video(
                pred_latents.to(torch.float16), parallel=True, show_progress_bar=False,
            )

        pred_np = (pred_frames[0].permute(0, 2, 3, 1)
                   .clamp(0, 1).mul(255).round().byte().cpu().numpy())

        result = gt.copy()
        result[hole_mask] = pred_np[hole_mask]

        out_stem = f"step{step:07d}_{stem}"
        write_video(masked,  os.path.join(TEST_DIR, out_stem + "_masked.mp4"))
        write_video(gt,      os.path.join(TEST_DIR, out_stem + "_gt.mp4"))
        write_video(result,  os.path.join(TEST_DIR, out_stem + "_infilled.mp4"))
        print(f"  hole coverage {coverage:.1f}%  -> {out_stem}_infilled.mp4")

    print(f"\nDone. Results in {TEST_DIR}/")


if __name__ == "__main__":
    main()
