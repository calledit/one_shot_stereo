"""
plot_loss.py — plot training loss curves from training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--out PATH] [--smooth 50]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default=os.path.join(ROOT, "training_log.csv"))
    parser.add_argument("--smooth", type=int, default=50, help="Smoothing window (steps)")
    parser.add_argument("--start",  type=int, default=0,  help="Ignore steps before this value")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    original_len = len(df)
    df = df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} stale rows from aborted runs")
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)
        print(f"Showing steps {args.start}+ ({len(df)} rows)")
    print(f"Loaded {len(df)} rows up to step {df['step'].max()}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Training Loss", fontsize=14)

    def plot(ax, col, label, color):
        data = df[col].dropna()
        steps = df.loc[data.index, "step"]
        ax.plot(steps, data, alpha=0.2, color=color, linewidth=0.5)
        if len(data) >= args.smooth:
            s = smooth(data.values, args.smooth)
            s_steps = steps.values[args.smooth - 1:]
            ax.plot(s_steps, s, color=color, linewidth=1.5, label=label)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # Panel 1: latent losses
    plot(axes[0], "loss_in",  "latent L1 inside holes",   "steelblue")
    plot(axes[0], "loss_out", "latent L1 outside holes",  "darkorange")
    axes[0].set_title("Latent L1 Loss")

    # Panel 2: pixel loss (sparse — only logged when computed)
    plot(axes[1], "pix", "pixel L1 outside holes", "seagreen")
    axes[1].set_title("Pixel Loss (5% of steps)")

    # Panel 3: GAN losses
    plot(axes[2], "d_loss", "discriminator", "crimson")
    plot(axes[2], "g_loss", "generator",     "purple")
    axes[2].set_title("GAN Loss")
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
