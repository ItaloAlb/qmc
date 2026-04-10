#!/usr/bin/env python3
"""
Read DMC walker snapshots from a binary dump file and plot densities.

Usage:
    python plot_walkers.py <walkers.bin> [--bins 80] [--output density.png]

Binary format (per snapshot):
    int32  nWalkers
    int32  stride
    float64[nWalkers * stride]  positions
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_walkers(path):
    """Read all snapshots and return a single (N_total, stride) array."""
    snapshots = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            nw, st = np.frombuffer(header, dtype=np.int32)
            data = f.read(nw * st * 8)
            if len(data) < nw * st * 8:
                break
            pos = np.frombuffer(data, dtype=np.float64).reshape(nw, st)
            snapshots.append(pos)
    return np.concatenate(snapshots)


def plot_2d_exciton(positions, bins, output):
    """Plot densities for a 2-particle 2D system (stride=4)."""
    xe, ye = positions[:, 0], positions[:, 1]
    xh, yh = positions[:, 2], positions[:, 3]

    # relative coordinate
    dx = xe - xh
    dy = ye - yh

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # electron density
    ax = axes[0]
    h = ax.hist2d(xe, ye, bins=bins, cmap="inferno", density=True)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(r"x ($a_0$)")
    ax.set_ylabel(r"y ($a_0$)")
    ax.set_title("electron density")
    ax.set_aspect("equal")

    # hole density
    ax = axes[1]
    h = ax.hist2d(xh, yh, bins=bins, cmap="inferno", density=True)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(r"x ($a_0$)")
    ax.set_ylabel(r"y ($a_0$)")
    ax.set_title("hole density")
    ax.set_aspect("equal")

    # relative e-h density
    ax = axes[2]
    h = ax.hist2d(dx, dy, bins=bins, cmap="inferno", density=True)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel(r"$x_e - x_h$ ($a_0$)")
    ax.set_ylabel(r"$y_e - y_h$ ($a_0$)")
    ax.set_title("relative density")
    ax.set_aspect("equal")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize DMC walker snapshots")
    parser.add_argument("file", help="Path to the _walkers.bin file")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins (default: 80)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    positions = read_walkers(args.file)
    n_samples, stride = positions.shape
    print(f"Loaded {n_samples} walker configurations (stride={stride})")

    if stride == 4:
        plot_2d_exciton(positions, args.bins, args.output)
    else:
        print(f"No built-in plot for stride={stride}. Positions array shape: {positions.shape}")
        print("Access the data with: positions = read_walkers('file.bin')")


if __name__ == "__main__":
    main()
