"""
infer.py — One Shot Stereo inference

Usage:
    python infer.py <masked_video> <original_video> <output_video> [--checkpoint PATH]

    masked_video   — 25-frame video with disocclusion holes painted green (0,255,0)
    original_video — 25-frame clean stereo video (real pixels outside holes)
    output_video   — path to write the infilled result
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taehv"))
from taehv import TAEHV

from data.dataset import compute_token_mask, FRAMES_PER_CLIP, OUTPUT_H, OUTPUT_W
from model.network import OneShotStereoNet

VAE_CHECKPOINT = os.path.join("taehv", "taew2_1.pth")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != FRAMES_PER_CLIP:
        raise ValueError(f"{path}: expected {FRAMES_PER_CLIP} frames, got {len(frames)}")
    return np.stack(frames)   # (25, H, W, 3) uint8


def save_video(frames_rgb, path, fps=25.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (OUTPUT_W, OUTPUT_H))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def extract_green_mask(frames_rgb):
    """Detect pure green (0,255,0) pixels as the hole mask. Returns (25, H, W) bool."""
    g = frames_rgb.astype(np.int16)
    return (g[..., 0] == 0) & (g[..., 1] == 255) & (g[..., 2] == 0)


def to_ntchw(frames_hwc, device, dtype=torch.float16):
    t = torch.from_numpy(frames_hwc).float().div_(255.0)
    return t.permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)   # (1, T, 3, H, W)


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def run_inference(masked_video_path, original_video_path, output_path,
                  checkpoint_path=None, device_str="cuda"):
    device = torch.device(device_str)

    masked_frames   = load_video(masked_video_path)
    original_frames = load_video(original_video_path)

    hole_mask  = extract_green_mask(masked_frames)           # (25, H, W) bool
    token_mask = torch.from_numpy(compute_token_mask(hole_mask)).unsqueeze(0).to(device)
    # (1, 7, 15, 26)

    print("Loading VAE ...")
    vae = TAEHV(checkpoint_path=VAE_CHECKPOINT).to(device, torch.float16)
    vae.eval()

    print("Loading network ...")
    net = OneShotStereoNet().to(device, torch.bfloat16)
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  weights loaded from {checkpoint_path}")
    else:
        print("  no checkpoint given — running with random weights")
    net.eval()

    with torch.no_grad():
        input_latents = vae.encode_video(
            to_ntchw(masked_frames, device),
            parallel=True, show_progress_bar=False,
        )   # (1, 7, 16, 60, 104) float16

        pred_latents = net(input_latents.to(torch.bfloat16), token_mask)
        # (1, 7, 16, 60, 104) bfloat16

        pred_frames = vae.decode_video(
            pred_latents.to(torch.float16),
            parallel=True, show_progress_bar=False,
        )   # (1, 25, 3, H, W) float16

    pred_np = (pred_frames[0].permute(0, 2, 3, 1)
               .clamp(0, 1).mul(255).round()
               .byte().cpu().numpy())   # (25, H, W, 3) uint8

    # Composite: real pixels outside holes, network output inside holes
    result = original_frames.copy()
    result[hole_mask] = pred_np[hole_mask]

    save_video(result, output_path)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One Shot Stereo inference")
    parser.add_argument("masked_video",   help="Input video with green-painted holes")
    parser.add_argument("original_video", help="Clean stereo video for compositing")
    parser.add_argument("output_video",   help="Output path for infilled video")
    parser.add_argument("--checkpoint",   default=None, help="Path to trained model weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_inference(args.masked_video, args.original_video, args.output_video,
                  args.checkpoint, args.device)
