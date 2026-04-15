#!/usr/bin/env python3
"""
Find dt_max: the largest time step where the DMC energy bias is still linear.

Sweeps delta_tau values, runs DMC for each, and identifies the linear regime
boundary by fitting E(dt) progressively from small dt upward and detecting
where the residuals deviate from linearity.

Usage:
    python3 find_dt_max.py <config.json> [options]

Example:
    python3 find_dt_max.py configs/exciton_exciton.json --dt-values 0.001 0.002 0.005 0.01 0.02 0.05
    python3 find_dt_max.py configs/exciton_exciton.json --dt-min 0.001 --dt-max 0.1 --n-points 8
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt


QMC_BINARY = os.path.join(os.path.dirname(__file__), "build", "bin", "qmc")


def make_sweep_config(base_config, dt, block_time, output_dir, index):
    """Create a config for a single dt value, keeping block_time constant."""
    cfg = copy.deepcopy(base_config)

    cfg["dmc"]["enabled"] = True
    cfg["dmc"]["delta_tau"] = dt
    cfg["dmc"]["dump_walkers"] = False
    cfg["dmc"]["descendant_weighting"] = False

    # Keep block_time = n_steps_per_block * delta_tau constant
    n_steps_per_block = max(1, round(block_time / dt))
    cfg["dmc"]["n_steps_per_block"] = n_steps_per_block

    # Disable optimizer and VMC for the sweep
    cfg["optimizer"]["enabled"] = False
    cfg["vmc"]["enabled"] = False

    # Set unique output path
    tag = f"dt_{dt:.6f}".rstrip("0").rstrip(".")
    out_path = os.path.join(output_dir, tag)
    cfg["output"]["file"] = out_path

    return cfg, out_path


def run_single(base_config, dt, block_time, output_dir, index):
    """Run QMC for a single dt value. Returns (dt, energy, std_error) or None."""
    cfg, out_path = make_sweep_config(base_config, dt, block_time, output_dir, index)

    config_file = os.path.join(output_dir, f"config_{index:02d}.json")
    with open(config_file, "w") as f:
        json.dump(cfg, f, indent=4)

    results_file = out_path + "_results.json"

    actual_block_time = cfg["dmc"]["n_steps_per_block"] * dt
    print(f"  [{index+1}] dt = {dt:.6f}  "
          f"(n_steps_per_block = {cfg['dmc']['n_steps_per_block']}, "
          f"block_time = {actual_block_time:.4f})")

    try:
        result = subprocess.run(
            [QMC_BINARY, config_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"      stderr: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"      TIMEOUT")
        return None

    if not os.path.exists(results_file):
        print(f"      No results file: {results_file}")
        return None

    with open(results_file) as f:
        res = json.load(f)

    if "dmc" not in res:
        print(f"      No DMC results in output")
        return None

    energy = res["dmc"]["energy"]
    std_error = res["dmc"]["std_error"]
    print(f"      E = {energy:.8f} +/- {std_error:.8f}")
    return (dt, energy, std_error)


def find_linear_regime(dt_arr, e_arr, err_arr, chi2_threshold=2.5):
    """
    Find dt_max by progressively adding larger dt points to a linear fit
    and detecting where the fit breaks down using the reduced chi-squared.

    Returns (dt_max_index, slope, intercept, intercept_err) of the best fit.
    """
    n = len(dt_arr)
    if n < 3:
        print("Warning: fewer than 3 dt points, cannot determine linearity.")
        return n - 1, None, None, None

    best_idx = 2  # at least use 3 points
    best_slope = None
    best_intercept = None
    best_intercept_err = None

    for k in range(3, n + 1):
        dt_sub = dt_arr[:k]
        e_sub = e_arr[:k]
        err_sub = err_arr[:k]

        # Weighted linear fit: E(dt) = a + b * dt
        # np.polyfit expects weights as 1/sigma
        w = 1.0 / err_sub
        coeffs, cov = np.polyfit(dt_sub, e_sub, 1, w=w, cov=True)

        slope, intercept = coeffs
        intercept_err = np.sqrt(cov[1, 1])

        # Calculate reduced chi-squared
        predicted = slope * dt_sub + intercept
        chi2 = np.sum(((e_sub - predicted) / err_sub) ** 2)
        dof = k - 2  # degrees of freedom (N data points - 2 fit parameters)
        reduced_chi2 = chi2 / dof

        # If reduced chi-squared jumps too high, the linear model is failing
        if k > 3 and reduced_chi2 > chi2_threshold:
            print(f"\n  Linearity breaks at dt = {dt_sub[-1]:.6f} "
                  f"(Reduced chi^2 = {reduced_chi2:.2f} > {chi2_threshold})")
            break

        best_idx = k - 1
        best_slope = slope
        best_intercept = intercept
        best_intercept_err = intercept_err

    return best_idx, best_slope, best_intercept, best_intercept_err


def plot_results(dt_arr, e_arr, err_arr, dt_max_idx, slope, intercept, output):
    """Plot E(dt) vs dt with linear fit and dt_max marker."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: E(dt) vs dt with error bars and linear fit
    ax = axes[0]
    ax.errorbar(dt_arr, e_arr, yerr=err_arr, fmt="ko", capsize=3, label="DMC data")

    if slope is not None:
        dt_fit = np.linspace(0, dt_arr[dt_max_idx] * 1.3, 100)
        e_fit = slope * dt_fit + intercept
        ax.plot(dt_fit, e_fit, "r--", label=f"linear fit (E₀ = {intercept:.6f})")
        ax.axvline(dt_arr[dt_max_idx], color="blue", linestyle=":", alpha=0.7,
                   label=f"dt_max = {dt_arr[dt_max_idx]:.4f}")

    ax.set_xlabel(r"$\delta\tau$")
    ax.set_ylabel(r"$E_{\mathrm{DMC}}$")
    ax.set_title(r"$E(\delta\tau)$ vs $\delta\tau$")
    ax.legend()

    # Right panel: residuals from linear fit
    ax = axes[1]
    if slope is not None:
        residuals = e_arr - (slope * dt_arr + intercept)
        ax.errorbar(dt_arr, residuals, yerr=err_arr, fmt="ko", capsize=3)
        ax.axhline(0, color="r", linestyle="--", alpha=0.5)
        ax.axvline(dt_arr[dt_max_idx], color="blue", linestyle=":", alpha=0.7,
                   label=f"dt_max = {dt_arr[dt_max_idx]:.4f}")
        ax.set_xlabel(r"$\delta\tau$")
        ax.set_ylabel(r"Residual from linear fit")
        ax.set_title("Residuals")
        ax.legend()

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"\nPlot saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Find dt_max for DMC time step extrapolation")
    parser.add_argument("config", help="Path to base JSON config file")
    parser.add_argument("--dt-values", nargs="+", type=float, default=None,
                        help="Explicit list of dt values to sweep")
    parser.add_argument("--dt-min", type=float, default=0.001,
                        help="Minimum dt for auto-generated sweep (default: 0.001)")
    parser.add_argument("--dt-max", type=float, default=0.1,
                        help="Maximum dt for auto-generated sweep (default: 0.1)")
    parser.add_argument("--n-points", type=int, default=8,
                        help="Number of dt points for auto-generated sweep (default: 8)")
    parser.add_argument("--block-time", type=float, default=None,
                        help="Block time to keep constant (default: from config)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Sigma threshold for linearity test (default: 2.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for sweep outputs (default: results/<system>/dt_sweep)")
    parser.add_argument("--plot", "-p", type=str, default=None,
                        help="Save plot to file (default: show interactively)")
    parser.add_argument("--binary", type=str, default=None,
                        help="Path to QMC binary (default: build/bin/qmc)")
    args = parser.parse_args()

    if args.binary:
        global QMC_BINARY
        QMC_BINARY = args.binary

    if not os.path.exists(QMC_BINARY):
        print(f"Error: QMC binary not found at {QMC_BINARY}")
        print("Build the project first or specify --binary")
        sys.exit(1)

    with open(args.config) as f:
        base_config = json.load(f)

    # Determine dt values
    if args.dt_values:
        dt_values = sorted(args.dt_values)
    else:
        dt_values = np.geomspace(args.dt_min, args.dt_max, args.n_points).tolist()

    # Determine block_time
    if args.block_time:
        block_time = args.block_time
    else:
        base_dt = base_config["dmc"].get("delta_tau", 0.01)
        base_nsb = base_config["dmc"].get("n_steps_per_block", 100)
        block_time = base_dt * base_nsb

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        system_name = base_config.get("system", "unknown")
        output_dir = os.path.join("results", system_name, "dt_sweep")
    os.makedirs(output_dir, exist_ok=True)

    print(f"System: {base_config.get('system', '?')}")
    print(f"Block time: {block_time:.4f}")
    print(f"dt values: {[f'{d:.6f}' for d in dt_values]}")
    print(f"Output dir: {output_dir}")
    print(f"Sigma threshold: {args.sigma}")
    print()

    # Run sweep
    results = []
    for i, dt in enumerate(dt_values):
        r = run_single(base_config, dt, block_time, output_dir, i)
        if r is not None:
            results.append(r)

    if len(results) < 3:
        print(f"\nOnly {len(results)} successful runs. Need at least 3 to determine linearity.")
        sys.exit(1)

    dt_arr = np.array([r[0] for r in results])
    e_arr = np.array([r[1] for r in results])
    err_arr = np.array([r[2] for r in results])

    # Sort by dt
    order = np.argsort(dt_arr)
    dt_arr, e_arr, err_arr = dt_arr[order], e_arr[order], err_arr[order]

    # Find linear regime
    print("\nLinearity analysis:")
    # Passando o limite de chi2 (você pode linkar com args.sigma se quiser manter o mesmo CLI argument)
    dt_max_idx, slope, intercept, intercept_err = find_linear_regime(dt_arr, e_arr, err_arr, chi2_threshold=2.5)

    dt_max = dt_arr[dt_max_idx]
    print(f"\n{'='*50}")
    print(f"  dt_max = {dt_max:.6f}")
    if intercept is not None:
        # Agora exibimos o erro junto com o E0
        print(f"  E(dt->0) = {intercept:.8f} +/- {intercept_err:.8f} (from linear fit)")
        print(f"  slope    = {slope:.8f}")
    print(f"{'='*50}")
    print(f"\nFor extrapolation, run at dt_max = {dt_max:.6f} and dt_max/4 = {dt_max/4:.6f}")

    # Save summary
    summary = {
        "dt_max": dt_max,
        "dt_max_over_4": dt_max / 4,
        "E0_linear_fit": intercept,
        "E0_error": intercept_err,
        "slope": slope,
        "chi2_threshold": 2.5,
        "data": [{"dt": float(d), "energy": float(e), "std_error": float(s)}
                 for d, e, s in zip(dt_arr, e_arr, err_arr)]
    }
    summary_file = os.path.join(output_dir, "dt_sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_file}")

    plot_results(dt_arr, e_arr, err_arr, dt_max_idx, slope, intercept, args.plot)


if __name__ == "__main__":
    main()