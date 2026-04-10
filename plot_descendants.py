#!/usr/bin/env python3
"""
Read DMC descendant-weighting data and plot the pure |Phi_0|^2 density.

Usage:
    python3 plot_descendants.py <descendants.bin> [--bins 80] [--output density.png]

Binary format (per tagging event):
    int32  nTagged
    int32  stride
    float64[nTagged * stride]  taggedPositions
    int32  [nTagged]           descendantCounts
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_descendants(path):
    """Read all tagging events and return positions and weights."""
    all_positions = []
    all_weights = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            nt, st = np.frombuffer(header, dtype=np.int32)
            pos_data = f.read(nt * st * 8)
            if len(pos_data) < nt * st * 8:
                break
            positions = np.frombuffer(pos_data, dtype=np.float64).reshape(nt, st)
            wt_data = f.read(nt * 4)
            if len(wt_data) < nt * 4:
                break
            weights = np.frombuffer(wt_data, dtype=np.int32).astype(np.float64)
            all_positions.append(positions)
            all_weights.append(weights)

    positions = np.concatenate(all_positions)
    weights = np.concatenate(all_weights)
    return positions, weights


def plot_2d_exciton(positions, weights, bins, output):
    """Plot descendant-weighted densities for a 2-particle 2D system (stride=4)."""
    xe, ye = positions[:, 0], positions[:, 1]
    xh, yh = positions[:, 2], positions[:, 3]
    dx = xe - xh
    dy = ye - yh

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # electron density
    ax = axes[0]
    h, xedges, yedges = np.histogram2d(xe, ye, bins=bins, weights=weights, density=True)
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="inferno")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"x ($a_0$)")
    ax.set_ylabel(r"y ($a_0$)")
    ax.set_title(r"electron $|\Phi_0|^2$")
    ax.set_aspect("equal")

    # hole density
    ax = axes[1]
    h, xedges, yedges = np.histogram2d(xh, yh, bins=bins, weights=weights, density=True)
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="inferno")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"x ($a_0$)")
    ax.set_ylabel(r"y ($a_0$)")
    ax.set_title(r"hole $|\Phi_0|^2$")
    ax.set_aspect("equal")

    # relative e-h density
    ax = axes[2]
    h, xedges, yedges = np.histogram2d(dx, dy, bins=bins, weights=weights, density=True)
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="inferno")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$x_e - x_h$ ($a_0$)")
    ax.set_ylabel(r"$y_e - y_h$ ($a_0$)")
    ax.set_title(r"relative $|\Phi_0|^2$")
    ax.set_aspect("equal")

    # effective sample size diagnostic
    N_eff = weights.sum()**2 / (weights**2).sum()
    fig.suptitle(f"Descendant weighting  |  $N_{{eff}} = {N_eff:.0f}$ / {len(weights)}", y=1.02)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize DMC descendant-weighted density")
    parser.add_argument("file", help="Path to the _descendants.bin file")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins (default: 80)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    positions, weights = read_descendants(args.file)
    n_samples, stride = positions.shape
    n_alive = np.count_nonzero(weights)
    print(f"Loaded {n_samples} tagged walkers (stride={stride})")
    print(f"  Non-zero descendants: {n_alive} ({100*n_alive/n_samples:.1f}%)")
    print(f"  Total descendants: {weights.sum():.0f}")
    N_eff = weights.sum()**2 / (weights**2).sum()
    print(f"  Effective sample size: {N_eff:.0f}")

    if stride == 4:
        plot_2d_exciton(positions, weights, args.bins, args.output)
    else:
        print(f"No built-in plot for stride={stride}. Use the reader directly:")
        print("  from plot_descendants import read_descendants")
        print("  positions, weights = read_descendants('file.bin')")


if __name__ == "__main__":
    main()
