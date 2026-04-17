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


def iter_snapshots(path):
    """Yield (nWalkers, stride) positions arrays one snapshot at a time."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            nw, st = np.frombuffer(header, dtype=np.int32)
            nbytes = int(nw) * int(st) * 8
            data = f.read(nbytes)
            if len(data) < nbytes:
                break
            yield np.frombuffer(data, dtype=np.float64).reshape(int(nw), int(st))


def scan_range(path, stride_expected=None):
    """First pass: find per-column min/max and total walker count."""
    mins = None
    maxs = None
    total = 0
    stride = None
    for pos in iter_snapshots(path):
        if stride is None:
            stride = pos.shape[1]
            if stride_expected is not None and stride != stride_expected:
                return stride, None, None, 0
        cur_min = pos.min(axis=0)
        cur_max = pos.max(axis=0)
        mins = cur_min if mins is None else np.minimum(mins, cur_min)
        maxs = cur_max if maxs is None else np.maximum(maxs, cur_max)
        total += pos.shape[0]
    return stride, mins, maxs, total


def plot_2d_exciton_streaming(path, bins, output):
    """Two-pass streaming plot for stride=4 (2-particle 2D)."""
    print("Pass 1/2: scanning data range...")
    stride, mins, maxs, total = scan_range(path, stride_expected=4)
    if stride != 4:
        print(f"No built-in plot for stride={stride}.")
        return
    print(f"Found {total} walker configurations")

    xe_min, ye_min, xh_min, yh_min = mins
    xe_max, ye_max, xh_max, yh_max = maxs
    dx_min = xe_min - xh_max
    dx_max = xe_max - xh_min
    dy_min = ye_min - yh_max
    dy_max = ye_max - yh_min

    edges_e_x = np.linspace(xe_min, xe_max, bins + 1)
    edges_e_y = np.linspace(ye_min, ye_max, bins + 1)
    edges_h_x = np.linspace(xh_min, xh_max, bins + 1)
    edges_h_y = np.linspace(yh_min, yh_max, bins + 1)
    edges_r_x = np.linspace(dx_min, dx_max, bins + 1)
    edges_r_y = np.linspace(dy_min, dy_max, bins + 1)

    H_e = np.zeros((bins, bins), dtype=np.float64)
    H_h = np.zeros((bins, bins), dtype=np.float64)
    H_r = np.zeros((bins, bins), dtype=np.float64)

    print("Pass 2/2: accumulating histograms...")
    for pos in iter_snapshots(path):
        xe, ye = pos[:, 0], pos[:, 1]
        xh, yh = pos[:, 2], pos[:, 3]
        H_e += np.histogram2d(xe, ye, bins=[edges_e_x, edges_e_y])[0]
        H_h += np.histogram2d(xh, yh, bins=[edges_h_x, edges_h_y])[0]
        H_r += np.histogram2d(xe - xh, ye - yh, bins=[edges_r_x, edges_r_y])[0]

    # normalize to densities
    def normalize(H, ex, ey):
        area = (ex[1] - ex[0]) * (ey[1] - ey[0])
        s = H.sum()
        return H / (s * area) if s > 0 else H

    D_e = normalize(H_e, edges_e_x, edges_e_y)
    D_h = normalize(H_h, edges_h_x, edges_h_y)
    D_r = normalize(H_r, edges_r_x, edges_r_y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def show(ax, D, ex, ey, title, xlabel, ylabel):
        im = ax.pcolormesh(ex, ey, D.T, cmap="inferno", shading="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")

    show(axes[0], D_e, edges_e_x, edges_e_y, "electron density",
         r"x ($a_0$)", r"y ($a_0$)")
    show(axes[1], D_h, edges_h_x, edges_h_y, "hole density",
         r"x ($a_0$)", r"y ($a_0$)")
    show(axes[2], D_r, edges_r_x, edges_r_y, "relative density",
         r"$x_e - x_h$ ($a_0$)", r"$y_e - y_h$ ($a_0$)")

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

    plot_2d_exciton_streaming(args.file, args.bins, args.output)


if __name__ == "__main__":
    main()
