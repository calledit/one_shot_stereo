"""
create_release_weights.py — strip a training checkpoint down to inference-only weights.

Removes optimizer state, discriminator, and scheduler state.
The output file is compatible with infer.py as-is.

Usage:
    python create_release_weights.py <checkpoint> [--output PATH]

    checkpoint  — path to a training checkpoint (step_XXXXXXX.pt)
    --output    — output path (default: <checkpoint stem>_release.pt)
"""

import argparse
import os
import torch


def create_release_weights(checkpoint_path, output_path):
    print(f"Loading {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    release = {
        "model": ckpt["model"],
        "step":  ckpt.get("step"),
    }

    in_mb  = os.path.getsize(checkpoint_path) / (1024 ** 2)
    torch.save(release, output_path)
    out_mb = os.path.getsize(output_path) / (1024 ** 2)

    print(f"Saved {output_path}")
    print(f"  {in_mb:.0f} MB  →  {out_mb:.0f} MB  ({100*out_mb/in_mb:.0f}% of original)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip training checkpoint to inference weights")
    parser.add_argument("checkpoint", help="Training checkpoint (.pt)")
    parser.add_argument("--output", default=None, help="Output path (default: <stem>_release.pt)")
    args = parser.parse_args()

    if args.output is None:
        stem = args.checkpoint[:-3] if args.checkpoint.endswith(".pt") else args.checkpoint
        args.output = stem + "_release.pt"

    create_release_weights(args.checkpoint, args.output)
