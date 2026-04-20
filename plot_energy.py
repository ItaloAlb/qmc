#!/usr/bin/env python3
"""
Plot DMC energy vs block step and run convergence diagnostics.

Usage:
    python plot_energy.py <output.dat> [--equil N] [--output energy.png]

Columns: block  block_energy  reference_energy  n_walkers  variance  acceptance_ratio
The .dat file contains accumulation-phase blocks only — equilibration is dropped
in DMC::run().
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def detect_equilibration(energies, max_frac=0.5, step_frac=0.02):
    """Find the smallest equilibration cutoff for which the Geweke test
    passes (|z| < 2) on the remaining production data.  Scans from 0 up to
    max_frac of the data in increments of step_frac.  Falls back to
    discarding 20 % if no cutoff is found."""
    n = len(energies)
    step = max(1, int(n * step_frac))
    for i in range(0, int(n * max_frac), step):
        prod = energies[i:]
        z, passed = geweke_test(prod)
        if passed:
            return i
    return int(n * 0.2)


def block_average(energies, max_block_exp=None):
    """Flyvbjerg-Petersen block averaging.
    Returns arrays of (block_size, error_estimate)."""
    n = len(energies)
    if max_block_exp is None:
        max_block_exp = int(np.floor(np.log2(n))) - 1
    sizes, errors = [], []
    for k in range(0, max_block_exp):
        bs = 2 ** k
        n_blocks = n // bs
        if n_blocks < 4:
            break
        blocked = energies[:n_blocks * bs].reshape(n_blocks, bs).mean(axis=1)
        err = np.std(blocked, ddof=1) / np.sqrt(n_blocks)
        sizes.append(bs)
        errors.append(err)
    return np.array(sizes), np.array(errors)


def geweke_test(energies, frac_a=0.1, frac_b=0.5):
    """Geweke convergence diagnostic. Compares the mean of the first frac_a
    of the chain to the last frac_b. Returns (z_score, converged)."""
    n = len(energies)
    na = int(n * frac_a)
    nb = int(n * frac_b)
    a = energies[:na]
    b = energies[-nb:]
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=1) / len(a)
    var_b = np.var(b, ddof=1) / len(b)
    z = (mean_a - mean_b) / np.sqrt(var_a + var_b)
    return z, abs(z) < 2.0


def main():
    parser = argparse.ArgumentParser(description="Plot DMC energy per block step")
    parser.add_argument("file", help="Path to the output .dat file")
    parser.add_argument("--equil", type=int, default=None,
                        help="Manual equilibration cutoff (block index). Auto-detected if omitted.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    block        = data[:, 0].astype(int)
    block_energy = data[:, 1]
    ref_energy   = data[:, 2]

    equil = args.equil if args.equil is not None else 0
    production = block_energy[equil:]
    n_prod = len(production)

    # --- Statistics ---
    E_mean = np.mean(production)

    # --- Print report ---
    print(f"Total blocks:        {len(block_energy)}")
    print(f"Equilibration cutoff: {equil}")
    print(f"Production blocks:   {n_prod}")
    print(f"Mean energy:         {E_mean:.8f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Energy traces
    ax.plot(block, block_energy, alpha=0.35, lw=0.8, label="Block energy")
    ax.plot(block, ref_energy, lw=1.2, label="Reference energy")
    ax.axvline(equil, color="k", ls="--", lw=1, label=f"Equilibration ({equil})")
    ax.axhline(E_mean, color="tab:red", ls=":", lw=1,
            label=f"E = {E_mean:.6f}")

    ax.set_xlabel("Block")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("DMC Energy vs Block Step")
    ax.legend(fontsize=9)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
