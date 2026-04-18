"""
infer.py — One Shot Stereo inference

Usage:
    python infer.py <masked_video> <original_video> <output_video>

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

from dataset import (
    compute_token_mask,
    FRAMES_PER_CLIP, OUTPUT_H, OUTPUT_W,
    LATENT_T, LATENT_C, LATENT_H, LATENT_W,
    TOKEN_T, TOKEN_H, TOKEN_W, TOKEN_DIM, PATCH_SIZE,
)

VAE_CHECKPOINT = os.path.join("taehv", "taew2_1.pth")
GREEN = np.array([0, 255, 0], dtype=np.uint8)


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
    """Detect pure green pixels as the hole mask. Returns (25, H, W) bool."""
    g = frames_rgb.astype(np.int16)
    mask = (g[..., 1] == 255) & (g[..., 0] == 0) & (g[..., 2] == 0)
    return mask


def to_ntchw(frames_hwc, device, dtype=torch.float16):
    t = torch.from_numpy(frames_hwc).float().div_(255.0)   # (T, H, W, 3)
    return t.permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)   # (1, T, 3, H, W)


# ---------------------------------------------------------------------------
# Tokenize / untokenize latents
# ---------------------------------------------------------------------------

def patchify(latents):
    """
    latents : (1, T, C, H, W)  e.g. (1, 7, 16, 60, 104)
    Returns : (1, T*TH*TW, PATCH²*C)  i.e. (1, 2730, 256)
    """
    B, T, C, H, W = latents.shape
    ph = H // PATCH_SIZE
    pw = W // PATCH_SIZE
    x = latents.reshape(B, T, C, ph, PATCH_SIZE, pw, PATCH_SIZE)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()   # (B, T, ph, pw, C, ps, ps)
    x = x.reshape(B, T * ph * pw, C * PATCH_SIZE * PATCH_SIZE)
    return x   # (B, 2730, 256)


def unpatchify(tokens, t=TOKEN_T, h=TOKEN_H, w=TOKEN_W):
    """
    tokens  : (1, T*H*W, PATCH²*C)
    Returns : (1, T, C, H*ps, W*ps)
    """
    B, N, D = tokens.shape
    C = D // (PATCH_SIZE * PATCH_SIZE)
    x = tokens.reshape(B, t, h, w, C, PATCH_SIZE, PATCH_SIZE)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()   # (B, T, C, h, ps, w, ps)
    x = x.reshape(B, t, C, h * PATCH_SIZE, w * PATCH_SIZE)
    return x   # (B, 7, 16, 60, 104)


# ---------------------------------------------------------------------------
# Mock model forward
# ---------------------------------------------------------------------------

def mock_forward(tokens, token_mask_flat):
    """
    Identity pass — returns input tokens unchanged.
    Replace this with the real model forward once trained.

    tokens          : (1, 2730, 256) float16
    token_mask_flat : (1, 2730) bool
    Returns         : (1, 2730, 256) float16
    """
    return tokens.clone()


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_inference(masked_video_path, original_video_path, output_path, device_str="cuda"):
    device = torch.device(device_str)

    # Load frames
    masked_frames   = load_video(masked_video_path)    # (25, H, W, 3) uint8 RGB, holes=green
    original_frames = load_video(original_video_path)  # (25, H, W, 3) uint8 RGB, clean

    # Derive hole mask from green pixels in the masked video
    hole_mask   = extract_green_mask(masked_frames)    # (25, H, W) bool
    token_mask  = compute_token_mask(hole_mask)        # (7, 15, 26) bool
    token_mask_t = torch.from_numpy(token_mask).to(device)

    # Flatten token mask to (1, 2730)
    token_mask_flat = token_mask_t.reshape(1, -1)

    # Load VAE
    print("Loading VAE ...")
    vae = TAEHV(checkpoint_path=VAE_CHECKPOINT).to(device, torch.float16)
    vae.eval()

    with torch.no_grad():
        # Encode masked input
        input_ntchw = to_ntchw(masked_frames, device)          # (1, 25, 3, H, W)
        input_latents = vae.encode_video(input_ntchw, parallel=True, show_progress_bar=False)
        # (1, 7, 16, 60, 104)

        # Tokenize → (1, 2730, 256)
        tokens = patchify(input_latents)

        # Append binary mask flag → (1, 2730, 257)
        flag = token_mask_flat.unsqueeze(-1).to(tokens.dtype)
        tokens_with_flag = torch.cat([tokens, flag], dim=-1)

        # Forward pass (mock: identity)
        pred_tokens = mock_forward(tokens_with_flag[..., :-1], token_mask_flat)

        # Untokenize → (1, 7, 16, 60, 104)
        pred_latents = unpatchify(pred_tokens)

        # Decode → (1, 25, 3, H, W) float16 [0,1]
        pred_frames = vae.decode_video(pred_latents, parallel=True, show_progress_bar=False)

    # Back to uint8 numpy (25, H, W, 3)
    pred_np = (pred_frames[0].permute(0, 2, 3, 1)
               .clamp(0, 1).mul(255).round()
               .byte().cpu().numpy())

    # Composite: use original pixels outside holes, network output inside holes
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_inference(args.masked_video, args.original_video, args.output_video, args.device)
