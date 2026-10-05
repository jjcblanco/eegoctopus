"""
Plot a recording saved by eegbridge.py as a 16-channel waterfall.

Usage:
    python inspect_recording.py recordings/20261005_142037_C3_rest.npz
    python inspect_recording.py recordings/xxx.npz --out plots/xxx.png
    python inspect_recording.py recordings/xxx.npz --detrend
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
    "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "Oz",
]

INK = "#3a86ff"      # single hue: every trace is the same entity (a channel)
AXIS_INK = "#5c6670"  # muted axis ink
GRID_INK = "#e3e6ea"  # recessive grid
BG = "#ffffff"


def plot_recording(npz_path, out_path, detrend):
    data = np.load(npz_path, allow_pickle=True)
    eeg = data["eeg"]
    metadata = json.loads(str(data["metadata"]))
    rate = metadata.get("effective_sample_rate_hz",
                        metadata.get("sample_rate_hz", 250))
    n_samples, n_channels = eeg.shape
    t = np.arange(n_samples) / rate

    fig, axes = plt.subplots(
        n_channels, 1, figsize=(9, 0.55 * n_channels),
        sharex=True, facecolor=BG,
    )
    for i, ax in enumerate(axes):
        y = eeg[:, i]
        if detrend:
            y = y - np.mean(y)
        ax.plot(t, y, color=INK, linewidth=0.7)
        ax.set_facecolor(BG)
        ax.set_ylim(y.min(), y.max())
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.spines["left"].set_color(AXIS_INK)
        ax.tick_params(axis="both", colors=AXIS_INK, labelsize=6, length=2)
        ax.grid(axis="x", color=GRID_INK, linewidth=0.5)
        ax.set_ylabel(CHANNEL_NAMES[i], rotation=0, labelpad=18,
                      color=AXIS_INK, fontsize=7, va="center", ha="right")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}")
        )

    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color(AXIS_INK)
    axes[-1].set_xlabel("Time (s)", color=AXIS_INK, fontsize=8)

    label = str(data["label"])
    region = metadata.get("brain_region", "?")
    fig.suptitle(
        f"{npz_path.name}  —  label '{label}'  region {region}  "
        f"{n_samples / rate:.1f}s @ {rate:.0f} Hz",
        fontsize=10, color=AXIS_INK, x=0.03, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, facecolor=BG, transparent=False)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot an eegbridge recording")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--detrend", action="store_true",
                        help="Remove per-channel mean (DC offset)")
    args = parser.parse_args()

    out = args.out or args.npz.with_suffix(".png")
    plot_recording(args.npz, out, args.detrend)
