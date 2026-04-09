#!/usr/bin/env python3
"""
Parameter sweep for QMC.

Usage:
    python3 run_sweep.py <base_config.json> <param_path> <start> <stop> [step]

Example:
    python3 run_sweep.py configs/exciton_exciton.json params.R 1.0 6.0 0.5

The <param_path> uses dot notation to point at any value inside the config,
e.g. 'params.R', 'params.d', 'dmc.delta_tau', 'params.me'.
"""
import json
import subprocess
import os
import sys
import copy
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


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    base_config_path = sys.argv[1]
    param_path       = sys.argv[2]
    start            = float(sys.argv[3])
    stop             = float(sys.argv[4])
    step             = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    with open(base_config_path) as f:
        base_config = json.load(f)

    sweep_values = np.arange(start, stop, step)
    sweep_name   = os.path.splitext(os.path.basename(base_config_path))[0]
    sweep_dir    = os.path.join(RESULTS_DIR,
                                f"{sweep_name}_{param_path.replace('.', '_')}")

    sweep_results = []
    for val in sweep_values:
        cfg = copy.deepcopy(base_config)
        set_nested(cfg, param_path, float(val))

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


if __name__ == "__main__":
    main()
