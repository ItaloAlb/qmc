#!/usr/bin/env python3
import json
import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
QMC_EXECUTABLE = "./build/bin/qmc"
CONFIG_FILE = "config.json"
RESULTS_DIR = "results/sweep"

# Base configuration template
BASE_CONFIG = {
    "system": "exciton_exciton",
    "params": {
        "me": 1.0,
        "mh": 1.0,
        "d": 0.4,
        "R": 2.0,
        "charges": [-1.0, -1.0],
        "wf_alpha": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "wf_params_init": [3.4722, 10.4458, 13.1154, 19.7361, 0.565659, 1.34458, 6.41809, 1.34458, 6.41809],
        "nParticles": 2,
        "nDim": 2
    },
    "output": {
        "file": "output"
    },
    "optimizer": {
        "enabled": True,
        "learning_rate": 0.1,
        "max_epochs": 50,
        "samples_per_epoch": 100000
    },
    "vmc": {
        "enabled": True,
        "n_steps": 10000000,
        "n_equilibration": 1000000
    },
    "dmc": {
        "enabled": True,
        "delta_tau": 0.001,
        "fixed_node": True,
        "max_branch": True
    }
}

# Parameter to sweep
SWEEP_PARAM = "R"                        # key inside "params"
SWEEP_VALUES = np.arange(1.0, 6.0, 0.5)  # values to scan


def run_single(config, label):
    """Write config, run QMC, return parsed results."""
    run_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)

    output_base = os.path.join(run_dir, "qmc")
    config["output"]["file"] = output_base

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"[{label}] Running QMC ...")
    result = subprocess.run(
        [QMC_EXECUTABLE, config_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[{label}] ERROR:\n{result.stderr}")
        return None

    results_path = output_base + "_results.json"
    if not os.path.exists(results_path):
        print(f"[{label}] Results file not found: {results_path}")
        return None

    with open(results_path) as f:
        return json.load(f)


def main():
    import copy

    sweep_results = []

    for val in SWEEP_VALUES:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["params"][SWEEP_PARAM] = float(val)

        label = f"{SWEEP_PARAM}_{val:.4f}"
        data = run_single(cfg, label)

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
        print(f"[{label}] Done — DMC energy = {entry.get('dmc_energy', 'N/A')}")

    if not sweep_results:
        print("No results collected.")
        return

    # Save collected data
    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"\nSummary saved to {summary_path}")

    # Plot
    params = [r["param_value"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    if "dmc_energy" in sweep_results[0]:
        energies = [r["dmc_energy"] for r in sweep_results]
        errors   = [r["dmc_std_error"] for r in sweep_results]
        ax.errorbar(params, energies, yerr=errors, fmt="o-", label="DMC", capsize=3)

    if "vmc_energy" in sweep_results[0]:
        energies = [r["vmc_energy"] for r in sweep_results]
        errors   = [r["vmc_std_error"] for r in sweep_results]
        ax.errorbar(params, energies, yerr=errors, fmt="s--", label="VMC", capsize=3)

    ax.set_xlabel(SWEEP_PARAM)
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy vs {SWEEP_PARAM}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(RESULTS_DIR, "sweep_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
