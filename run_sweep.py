#!/usr/bin/env python3
"""
Parameter sweep for QMC.

Usage:
    python3 run_sweep.py <base_config.json> <param_path> <start> <stop> [step]
    python3 run_sweep.py <base_config.json> --dt-max <dt_max>

Examples:
    # Generic parameter sweep
    python3 run_sweep.py configs/exciton_exciton.json params.R 1.0 6.0 0.5

    # Time step extrapolation (Lee thesis: dt_max and dt_max/4, effort 1:8)
    python3 run_sweep.py configs/exciton_exciton.json --dt-max 0.02

The <param_path> uses dot notation to point at any value inside the config,
e.g. 'params.R', 'params.d', 'dmc.delta_tau', 'params.me'.

When sweeping 'dmc.delta_tau', n_steps_per_block is automatically adjusted
to keep block_time constant across runs.

The --dt-max flag triggers the optimal 2-point extrapolation protocol:
  - Runs at dt_max (1/9 of total effort) and dt_max/4 (8/9 of effort)
  - Keeps block_time constant by adjusting n_steps_per_block
  - Extrapolates E(dt) -> E(0) using linear fit
"""
import json
import subprocess
import os
import sys
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt

QMC_EXECUTABLE = "./build/bin/qmc"
RESULTS_DIR = "results/sweep"


def set_nested(d, path, value):
    """Set a nested value using dot notation, e.g. 'params.R'."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def get_nested(d, path):
    """Get a nested value using dot notation."""
    keys = path.split(".")
    for k in keys:
        d = d[k]
    return d


def adjust_block_time(config, base_block_time):
    """Adjust n_steps_per_block to keep block_time constant for the current delta_tau."""
    dt = config["dmc"]["delta_tau"]
    n_steps = max(1, round(base_block_time / dt))
    config["dmc"]["n_steps_per_block"] = n_steps
    return n_steps


def run_single(config, label, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    output_base = os.path.join(run_dir, "qmc")
    config["output"]["file"] = output_base

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"[{label}] Running QMC...")
    result = subprocess.run([QMC_EXECUTABLE, config_path])

    if result.returncode != 0:
        print(f"[{label}] ERROR (exit code {result.returncode})")
        return None

    results_path = output_base + "_results.json"
    if not os.path.exists(results_path):
        print(f"[{label}] Results file not found")
        return None

    with open(results_path) as f:
        return json.load(f)


def dt_extrapolation(results):
    """
    Linear extrapolation E(dt) -> E(0) from two points.

    E(dt) = E_0 + kappa * dt
    """
    dt1, e1, s1 = results[0]["dt"], results[0]["dmc_energy"], results[0]["dmc_std_error"]
    dt2, e2, s2 = results[1]["dt"], results[1]["dmc_energy"], results[1]["dmc_std_error"]

    # Weighted linear fit
    w1 = 1.0 / s1**2
    w2 = 1.0 / s2**2

    kappa = (e2 - e1) / (dt2 - dt1)
    E0 = e1 - kappa * dt1

    # Error propagation for E0
    # E0 = e1 - (e2 - e1) * dt1 / (dt2 - dt1)
    #    = e1 * (1 + dt1/(dt2-dt1)) - e2 * dt1/(dt2-dt1)
    r = dt1 / (dt2 - dt1)
    sigma_E0 = np.sqrt((1 + r)**2 * s1**2 + r**2 * s2**2)

    return E0, sigma_E0, kappa


def run_dt_extrapolation(base_config, dt_max, total_blocks):
    """Run the 2-point dt extrapolation protocol from Lee's thesis."""
    dt_small = dt_max / 4.0
    base_dt = base_config["dmc"].get("delta_tau", 0.01)
    base_nsb = base_config["dmc"].get("n_steps_per_block", 100)
    base_block_time = base_dt * base_nsb

    # Effort split: 1/9 on dt_max, 8/9 on dt_max/4
    # This comes from optimal variance: T1/T2 = (dt2/dt1)^(3/2) = 4^(3/2) = 8
    blocks_large = max(1, round(total_blocks / 9))
    blocks_small = total_blocks - blocks_large

    sweep_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown"
    sweep_dir = os.path.join(RESULTS_DIR, f"{sweep_name}_dt_extrapolation")

    runs = [
        {"dt": dt_max,   "n_block_steps": blocks_large, "label": f"dt_{dt_max:.6f}"},
        {"dt": dt_small,  "n_block_steps": blocks_small, "label": f"dt_{dt_small:.6f}"},
    ]

    print(f"Time step extrapolation (Lee thesis protocol)")
    print(f"  dt_max     = {dt_max}")
    print(f"  dt_max/4   = {dt_small}")
    print(f"  block_time = {base_block_time:.4f}")
    print(f"  effort split: {blocks_large} blocks (dt_max) + {blocks_small} blocks (dt_max/4)")
    print()

    sweep_results = []
    for run in runs:
        cfg = copy.deepcopy(base_config)
        cfg["dmc"]["enabled"] = True
        cfg["dmc"]["delta_tau"] = run["dt"]
        cfg["dmc"]["n_block_steps"] = run["n_block_steps"]
        adjust_block_time(cfg, base_block_time)

        actual_bt = cfg["dmc"]["n_steps_per_block"] * run["dt"]
        print(f"  [{run['label']}] n_steps_per_block = {cfg['dmc']['n_steps_per_block']}, "
              f"block_time = {actual_bt:.4f}, n_block_steps = {run['n_block_steps']}")

        run_dir = os.path.join(sweep_dir, run["label"])
        data = run_single(cfg, run["label"], run_dir)

        if data is None or "dmc" not in data:
            print(f"  [{run['label']}] FAILED - aborting extrapolation")
            return

        entry = {
            "dt": run["dt"],
            "n_block_steps": run["n_block_steps"],
            "dmc_energy": data["dmc"]["energy"],
            "dmc_std_error": data["dmc"]["std_error"],
            "dmc_variance": data["dmc"]["variance"],
        }
        sweep_results.append(entry)
        print(f"  [{run['label']}] E = {entry['dmc_energy']:.8f} +/- {entry['dmc_std_error']:.8f}")
        print()

    # Extrapolate
    E0, sigma_E0, kappa = dt_extrapolation(sweep_results)

    print("=" * 55)
    print(f"  E(dt_max)   = {sweep_results[0]['dmc_energy']:.8f} +/- {sweep_results[0]['dmc_std_error']:.8f}")
    print(f"  E(dt_max/4) = {sweep_results[1]['dmc_energy']:.8f} +/- {sweep_results[1]['dmc_std_error']:.8f}")
    print(f"  kappa       = {kappa:.8f}")
    print(f"  E(dt->0)    = {E0:.8f} +/- {sigma_E0:.8f}")
    print("=" * 55)

    # Save summary
    summary = {
        "dt_max": dt_max,
        "dt_small": dt_small,
        "E0": E0,
        "sigma_E0": sigma_E0,
        "kappa": kappa,
        "runs": sweep_results,
    }
    summary_path = os.path.join(sweep_dir, "extrapolation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to {summary_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    dts = np.array([r["dt"] for r in sweep_results])
    es = np.array([r["dmc_energy"] for r in sweep_results])
    errs = np.array([r["dmc_std_error"] for r in sweep_results])

    ax.errorbar(dts, es, yerr=errs, fmt="ko", capsize=4, markersize=8, label="DMC runs")

    # Extrapolation line
    dt_line = np.linspace(0, dt_max * 1.1, 100)
    e_line = E0 + kappa * dt_line
    ax.plot(dt_line, e_line, "r--", alpha=0.7, label="linear fit")

    # E(0) marker
    ax.errorbar([0], [E0], yerr=[sigma_E0], fmt="rs", capsize=4, markersize=10,
                label=f"$E_0$ = {E0:.6f} $\\pm$ {sigma_E0:.6f}")

    ax.set_xlabel(r"$\delta\tau$")
    ax.set_ylabel(r"$E_{\mathrm{DMC}}$")
    ax.set_title(r"Time step extrapolation $\delta\tau \to 0$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(sweep_dir, "extrapolation_plot.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


def run_generic_sweep(base_config, param_path, sweep_values):
    """Run a generic parameter sweep."""
    sweep_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown"
    sweep_dir = os.path.join(RESULTS_DIR,
                             f"{sweep_name}_{param_path.replace('.', '_')}")

    # Compute base block_time for delta_tau sweeps
    is_dt_sweep = param_path == "dmc.delta_tau"
    base_dt = base_config["dmc"].get("delta_tau", 0.01)
    base_nsb = base_config["dmc"].get("n_steps_per_block", 100)
    base_block_time = base_dt * base_nsb

    if is_dt_sweep:
        print(f"Detected delta_tau sweep: keeping block_time = {base_block_time:.4f} constant\n")

    sweep_results = []
    for val in sweep_values:
        cfg = copy.deepcopy(base_config)
        set_nested(cfg, param_path, float(val))

        if is_dt_sweep:
            n_steps = adjust_block_time(cfg, base_block_time)
            print(f"  dt = {val:.6f} -> n_steps_per_block = {n_steps}")

        label   = f"{param_path.split('.')[-1]}_{val:.4f}"
        run_dir = os.path.join(sweep_dir, label)
        data    = run_single(cfg, label, run_dir)

        if data is None:
            continue

        entry = {"param_value": float(val)}
        if "dmc" in data:
            entry["dmc_energy"]    = data["dmc"]["energy"]
            entry["dmc_std_error"] = data["dmc"]["std_error"]
        if "vmc" in data:
            entry["vmc_energy"]    = data["vmc"]["energy"]
            entry["vmc_std_error"] = data["vmc"]["std_error"]

        sweep_results.append(entry)
        print(f"[{label}] Done")

    if not sweep_results:
        print("No results collected.")
        return

    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"\nSummary saved to {summary_path}")

    # Plot
    params = [r["param_value"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(8, 5))

    if "dmc_energy" in sweep_results[0]:
        ax.errorbar(params,
                    [r["dmc_energy"] for r in sweep_results],
                    yerr=[r["dmc_std_error"] for r in sweep_results],
                    fmt="o-", label="DMC", capsize=3)

    if "vmc_energy" in sweep_results[0]:
        ax.errorbar(params,
                    [r["vmc_energy"] for r in sweep_results],
                    yerr=[r["vmc_std_error"] for r in sweep_results],
                    fmt="s--", label="VMC", capsize=3)

    ax.set_xlabel(param_path)
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy vs {param_path}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(sweep_dir, "sweep_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep / time step extrapolation for QMC",
        usage="%(prog)s <config> [param_path start stop [step]] | [--dt-max DT_MAX]")
    parser.add_argument("config", help="Path to base JSON config file")
    parser.add_argument("sweep_args", nargs="*",
                        help="param_path start stop [step] for generic sweep")
    parser.add_argument("--dt-max", type=float, default=None,
                        help="Run 2-point dt extrapolation (Lee thesis protocol)")
    parser.add_argument("--total-blocks", type=int, default=None,
                        help="Total block steps for dt extrapolation "
                             "(default: from config n_block_steps)")
    parser.add_argument("--binary", type=str, default=None,
                        help="Path to QMC binary")
    args = parser.parse_args()

    if args.binary:
        global QMC_EXECUTABLE
        QMC_EXECUTABLE = args.binary

    with open(args.config) as f:
        base_config = json.load(f)

    if args.dt_max is not None:
        # Default: use config's n_block_steps as the *per-run* baseline,
        # so total = 9 * n_block_steps (each run gets at least n_block_steps).
        if args.total_blocks is not None:
            total_blocks = args.total_blocks
        else:
            base_blocks = base_config["dmc"].get("n_block_steps", 1000)
            total_blocks = 9 * base_blocks
        run_dt_extrapolation(base_config, args.dt_max, total_blocks)
    elif len(args.sweep_args) >= 3:
        param_path = args.sweep_args[0]
        start = float(args.sweep_args[1])
        stop = float(args.sweep_args[2])
        step = float(args.sweep_args[3]) if len(args.sweep_args) > 3 else 0.5
        sweep_values = np.arange(start, stop, step)
        run_generic_sweep(base_config, param_path, sweep_values)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
