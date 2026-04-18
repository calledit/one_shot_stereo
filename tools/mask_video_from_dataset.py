"""
mask_video_from_dataset.py

Given a .npz file from the training dataset, produces two videos:
    <stem>_masked.mp4   — GT frames with holes painted green (input for infer.py)
    <stem>_original.mp4 — GT frames unmodified            (original for infer.py)

Usage:
    python mask_video_from_dataset.py <path/to/clip_fNNNNNN.npz> [--frame 0|middle|end]
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import FRAMES_PER_CLIP, OUTPUT_W, OUTPUT_H


def load_clip(npz_path, window):
    gt_path = npz_path[:-4] + "_gt.mp4"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT video not found: {gt_path}")

    hole_mask_full = np.load(npz_path)["hole_mask"]   # (N, H, W) bool
    N = hole_mask_full.shape[0]

    cap = cv2.VideoCapture(gt_path)
    frames = []
    while len(frames) < N:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) != N:
        raise RuntimeError(f"Expected {N} frames from {gt_path}, got {len(frames)}")

    if window == "start":
        start = 0
    elif window == "middle":
        start = (N - FRAMES_PER_CLIP) // 2
    elif window == "end":
        start = N - FRAMES_PER_CLIP
    else:
        raise ValueError(f"Unknown window: {window}")

    start = max(0, min(start, N - FRAMES_PER_CLIP))
    gt        = np.stack(frames)[start:start + FRAMES_PER_CLIP]          # (25, H, W, 3)
    hole_mask = hole_mask_full[start:start + FRAMES_PER_CLIP]            # (25, H, W)
    return gt, hole_mask


def write_video(frames_rgb, path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (OUTPUT_W, OUTPUT_H))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to a training .npz file")
    parser.add_argument("--window", choices=["start", "middle", "end"], default="start",
                        help="Which 25-frame window to use for 50-frame clips (default: start)")
    args = parser.parse_args()

    gt, hole_mask = load_clip(args.npz, args.window)

    masked = gt.copy()
    for i in range(FRAMES_PER_CLIP):
        masked[i][hole_mask[i]] = [0, 255, 0]

    stem = args.npz[:-4]
    masked_path   = stem + "_masked.mp4"
    original_path = stem + "_original.mp4"

    write_video(masked,  masked_path)
    write_video(gt,      original_path)

    print(f"Wrote {masked_path}")
    print(f"Wrote {original_path}")
    print(f"Hole coverage: {hole_mask.mean()*100:.1f}%")
    print()
    print("Run inference with:")
    print(f"  python infer.py {masked_path} {original_path} {stem}_infilled.mp4")


if __name__ == "__main__":
    main()
