#!/usr/bin/env python3
"""
Parameter sweep for QMC.

Usage:
    python3 run_sweep.py <config> <param_path> <start> <stop> [step]
    python3 run_sweep.py <config> --dt-max <dt_max>
    python3 run_sweep.py <config> <param_path> <start> <stop> [step] --dt-max <dt_max>

Examples:
    # Generic parameter sweep
    python3 run_sweep.py configs/exciton_exciton.json params.R 1.0 7.0 0.5

    # Time step extrapolation only (Lee thesis: dt_max and dt_max/4, effort 1:8)
    python3 run_sweep.py configs/exciton_exciton.json --dt-max 0.02

    # Parameter sweep + dt extrapolation at each point
    python3 run_sweep.py configs/exciton_exciton.json params.R 1.0 7.0 0.5 --dt-max 0.05

The <param_path> uses dot notation to point at any value inside the config,
e.g. 'params.R', 'params.d', 'dmc.delta_tau', 'params.me'.

When sweeping 'dmc.delta_tau', n_steps_per_block is automatically adjusted
to keep block_time constant across runs.

The --dt-max flag triggers the optimal 2-point extrapolation protocol:
  - Runs at dt_max (1/9 of total effort) and dt_max/4 (8/9 of effort).
    Effort is measured against accumulation_blocks * n_steps_per_block;
    the config's accumulation_blocks sets the dt_max run, and the
    dt_max/4 run is scaled by the 1:8 ratio.
  - Keeps block_time constant by adjusting n_steps_per_block.
  - The dt_max run equilibrates and writes a checkpoint; the dt_max/4
    run resumes from it with zero equilibration.
  - Extrapolates E(dt) -> E(0) using linear fit.

When combined with a parameter sweep, dt extrapolation is performed at
each sweep point (with its own checkpoint), and the extrapolated
E(dt->0) is plotted.
"""
import json
import subprocess
import os
import sys
import copy
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt

QMC_EXECUTABLE = "./build/bin/qmc"
RESULTS_DIR = "results/sweep"
MAX_RETRIES = 3


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


def dmc_is_valid(data, output_base):
    """Check if a DMC run produced valid results (population didn't die).
    Checks both the energy from the results JSON and the final population
    from the .dat file (column 4: nWalkers)."""
    if data is None or "dmc" not in data:
        return False
    dmc = data["dmc"]
    if dmc["energy"] == 0.0:
        return False
    dat_path = output_base + ".dat"
    if os.path.exists(dat_path):
        with open(dat_path) as f:
            lines = f.readlines()
        if lines:
            last_cols = lines[-1].split()
            if len(last_cols) >= 4:
                final_population = int(last_cols[3])
                if final_population == 0:
                    return False
    return True


def run_single(config, label, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    output_base = os.path.join(run_dir, "qmc")
    config["output"]["file"] = output_base

    config_path = os.path.join(run_dir, "config.json")

    for attempt in range(1, MAX_RETRIES + 1):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        if attempt == 1:
            print(f"[{label}] Running QMC...")
        else:
            print(f"[{label}] Retry {attempt}/{MAX_RETRIES} (population died)...")

        result = subprocess.run([QMC_EXECUTABLE, config_path])

        if result.returncode != 0:
            print(f"[{label}] ERROR (exit code {result.returncode})")
            continue

        results_path = output_base + "_results.json"
        if not os.path.exists(results_path):
            print(f"[{label}] Results file not found")
            continue

        with open(results_path) as f:
            data = json.load(f)

        if config.get("dmc", {}).get("enabled", False) and not dmc_is_valid(data, output_base):
            print(f"[{label}] Population died (energy=0 or population=0) - will retry")
            continue

        return data

    print(f"[{label}] FAILED after {MAX_RETRIES} attempts")
    return None


def dt_extrapolation(results):
    """
    Linear extrapolation E(dt) -> E(0) from two points.

    E(dt) = E_0 + kappa * dt
    """
    dt1, e1 = results[0]["dt"], results[0]["dmc_energy"]
    dt2, e2 = results[1]["dt"], results[1]["dmc_energy"]

    kappa = (e2 - e1) / (dt2 - dt1)
    E0 = e1 - kappa * dt1

    return E0, kappa


def compute_dt_runs(dt_max, block_time, base_accum_blocks):
    """Compute the two dt-extrapolation runs with proper effort-aware 1:8 split.

    block_time is kept constant: n_steps_per_block = round(block_time / dt).
    Effort = accumulation_blocks * n_steps_per_block, so runs at smaller dt
    cost more per block. The 1:8 optimal split (Lee thesis) requires:
        accum_small = 8 * accum_large * steps_large / steps_small
    For dt_small = dt_max/4 this gives accum_small = 2 * accum_large.
    """
    dt_small = dt_max / 4.0
    steps_large = max(1, round(block_time / dt_max))
    steps_small = max(1, round(block_time / dt_small))

    accum_large = base_accum_blocks
    accum_small = max(1, round(8 * accum_large * steps_large / steps_small))

    effort_large = accum_large * steps_large
    effort_small = accum_small * steps_small

    return [
        {"dt": dt_max,   "accumulation_blocks": accum_large,
         "n_steps_per_block": steps_large, "label": f"dt_{dt_max:.6f}"},
        {"dt": dt_small, "accumulation_blocks": accum_small,
         "n_steps_per_block": steps_small, "label": f"dt_{dt_small:.6f}"},
    ], effort_large, effort_small


def run_dt_extrapolation(base_config, dt_max, base_accum_blocks, resume_equilibration_blocks=0):
    """Run the 2-point dt extrapolation protocol from Lee's thesis.

    The first run (dt_max) performs full equilibration and saves a
    checkpoint. The second run (dt_max/4) resumes from that checkpoint,
    so it only pays the accumulation cost.
    """
    dt_small = dt_max / 4.0
    base_dt = base_config["dmc"].get("delta_tau", 0.01)
    base_nsb = base_config["dmc"].get("n_steps_per_block", 100)
    block_time = base_dt * base_nsb

    runs, effort_large, effort_small = compute_dt_runs(dt_max, block_time, base_accum_blocks)

    sweep_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown"
    sweep_dir = os.path.join(RESULTS_DIR, f"{sweep_name}_dt_extrapolation")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Time step extrapolation (Lee thesis protocol)")
    print(f"  dt_max     = {dt_max}  ({runs[0]['accumulation_blocks']} accum blocks, "
          f"{runs[0]['n_steps_per_block']} steps/block)")
    print(f"  dt_max/4   = {dt_small}  ({runs[1]['accumulation_blocks']} accum blocks, "
          f"{runs[1]['n_steps_per_block']} steps/block)")
    print(f"  block_time = {block_time:.4f}")
    print(f"  effort split: {effort_large} steps (dt_max, 1/9) + "
          f"{effort_small} steps (dt_max/4, 8/9)")
    print()

    sweep_results = []
    run_dirs = []
    for i, run in enumerate(runs):
        cfg = copy.deepcopy(base_config)
        cfg["dmc"]["enabled"] = True
        cfg["dmc"]["delta_tau"] = run["dt"]
        cfg["dmc"]["accumulation_blocks"] = run["accumulation_blocks"]
        cfg["dmc"]["n_steps_per_block"] = run["n_steps_per_block"]

        run_dir = os.path.join(sweep_dir, run["label"])
        os.makedirs(run_dir, exist_ok=True)
        run_dirs.append(run_dir)

        if i == 0:
            cfg["dmc"]["checkpoint"] = True
            cfg["dmc"]["resume_from_checkpoint"] = False
        else:
            cfg["dmc"]["checkpoint"] = False
            cfg["dmc"]["resume_from_checkpoint"] = True
            cfg["dmc"]["equilibration_blocks"] = resume_equilibration_blocks
            src_ckpt = os.path.join(run_dirs[0], "qmc_checkpoint.bin")
            dst_ckpt = os.path.join(run_dir, "qmc_checkpoint.bin")
            if os.path.exists(src_ckpt):
                shutil.copyfile(src_ckpt, dst_ckpt)
            else:
                print(f"  [{run['label']}] WARNING: checkpoint from first run not found, "
                      f"falling back to fresh equilibration")
                cfg["dmc"]["resume_from_checkpoint"] = False
                cfg["dmc"]["equilibration_blocks"] = base_config["dmc"].get("equilibration_blocks", 200)

        print(f"  [{run['label']}] n_steps_per_block = {run['n_steps_per_block']}, "
              f"block_time = {run['n_steps_per_block'] * run['dt']:.4f}, "
              f"accumulation_blocks = {run['accumulation_blocks']}")

        data = run_single(cfg, run["label"], run_dir)

        if data is None or "dmc" not in data:
            print(f"  [{run['label']}] FAILED - aborting extrapolation")
            return

        entry = {
            "dt": run["dt"],
            "n_block_steps": run["n_block_steps"],
            "dmc_energy": data["dmc"]["energy"],
            "dmc_variance": data["dmc"]["variance"],
        }
        sweep_results.append(entry)
        print(f"  [{run['label']}] E = {entry['dmc_energy']:.8f}")
        print()

    # Extrapolate
    E0, kappa = dt_extrapolation(sweep_results)

    print("=" * 55)
    print(f"  E(dt_max)   = {sweep_results[0]['dmc_energy']:.8f}")
    print(f"  E(dt_max/4) = {sweep_results[1]['dmc_energy']:.8f}")
    print(f"  kappa       = {kappa:.8f}")
    print(f"  E(dt->0)    = {E0:.8f}")
    print("=" * 55)

    # Save summary
    summary = {
        "dt_max": dt_max,
        "dt_small": dt_small,
        "E0": E0,
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

    ax.plot(dts, es, "ko", markersize=8, label="DMC runs")

    # Extrapolation line
    dt_line = np.linspace(0, dt_max * 1.1, 100)
    e_line = E0 + kappa * dt_line
    ax.plot(dt_line, e_line, "r--", alpha=0.7, label="linear fit")

    # E(0) marker
    ax.plot([0], [E0], "rs", markersize=10, label=f"$E_0$ = {E0:.6f}")

    ax.set_xlabel(r"$\delta\tau$")
    ax.set_ylabel(r"$E_{\mathrm{DMC}}$")
    ax.set_title(r"Time step extrapolation $\delta\tau \to 0$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(sweep_dir, "extrapolation_plot.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


def run_sweep_with_dt_extrapolation(base_config, param_path, sweep_values, dt_max, base_accum_blocks,
                                     resume_equilibration_blocks=0):
    """Run a parameter sweep where each point uses dt extrapolation.

    For each parameter value, the dt_max run equilibrates and saves a
    checkpoint; the dt_max/4 run resumes from it with zero equilibration.
    Checkpoints are per-parameter-value (a new Hamiltonian needs a new
    equilibrated ensemble).
    """
    dt_small = dt_max / 4.0
    base_dt = base_config["dmc"].get("delta_tau", 0.01)
    base_nsb = base_config["dmc"].get("n_steps_per_block", 100)
    block_time = base_dt * base_nsb

    dt_runs_template, effort_large, effort_small = compute_dt_runs(
        dt_max, block_time, base_accum_blocks)

    sweep_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown"
    sweep_dir = os.path.join(RESULTS_DIR,
                             f"{sweep_name}_{param_path.replace('.', '_')}_dt_extrap")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Parameter sweep with dt extrapolation")
    print(f"  param       = {param_path}")
    print(f"  values      = {sweep_values}")
    print(f"  dt_max      = {dt_max}  ({dt_runs_template[0]['accumulation_blocks']} accum blocks, "
          f"{dt_runs_template[0]['n_steps_per_block']} steps/block)")
    print(f"  dt_max/4    = {dt_small}  ({dt_runs_template[1]['accumulation_blocks']} accum blocks, "
          f"{dt_runs_template[1]['n_steps_per_block']} steps/block)")
    print(f"  block_time  = {block_time:.4f}")
    print(f"  effort split: {effort_large} steps (dt_max, 1/9) + "
          f"{effort_small} steps (dt_max/4, 8/9)")
    print()

    sweep_results = []
    for val in sweep_values:
        param_label = f"{param_path.split('.')[-1]}_{val:.4f}"
        param_dir = os.path.join(sweep_dir, param_label)
        os.makedirs(param_dir, exist_ok=True)

        print(f"=== {param_label} ===")

        dt_results = []
        failed = False
        run_dirs = []
        for i, run in enumerate(dt_runs_template):
            cfg = copy.deepcopy(base_config)
            set_nested(cfg, param_path, float(val))
            cfg["dmc"]["enabled"] = True
            cfg["dmc"]["delta_tau"] = run["dt"]
            cfg["dmc"]["accumulation_blocks"] = run["accumulation_blocks"]
            cfg["dmc"]["n_steps_per_block"] = run["n_steps_per_block"]

            run_dir = os.path.join(param_dir, run["label"])
            os.makedirs(run_dir, exist_ok=True)
            run_dirs.append(run_dir)

            if i == 0:
                cfg["dmc"]["checkpoint"] = True
                cfg["dmc"]["resume_from_checkpoint"] = False
            else:
                cfg["dmc"]["checkpoint"] = False
                cfg["dmc"]["resume_from_checkpoint"] = True
                cfg["dmc"]["equilibration_blocks"] = resume_equilibration_blocks
                src_ckpt = os.path.join(run_dirs[0], "qmc_checkpoint.bin")
                dst_ckpt = os.path.join(run_dir, "qmc_checkpoint.bin")
                if os.path.exists(src_ckpt):
                    shutil.copyfile(src_ckpt, dst_ckpt)
                else:
                    print(f"  [{param_label}/{run['label']}] WARNING: checkpoint not found, "
                          f"falling back to fresh equilibration")
                    cfg["dmc"]["resume_from_checkpoint"] = False
                    cfg["dmc"]["equilibration_blocks"] = base_config["dmc"].get("equilibration_blocks", 200)

            data = run_single(cfg, f"{param_label}/{run['label']}", run_dir)

            if data is None or "dmc" not in data:
                print(f"  [{param_label}/{run['label']}] FAILED - skipping this parameter value")
                failed = True
                break

            dt_results.append({
                "dt": run["dt"],
                "dmc_energy": data["dmc"]["energy"],
                "dmc_variance": data["dmc"]["variance"],
            })
            print(f"  [{run['label']}] E = {dt_results[-1]['dmc_energy']:.8f}")

        if failed or len(dt_results) < 2:
            continue

        E0, kappa = dt_extrapolation(dt_results)
        print(f"  E(dt->0) = {E0:.8f}")
        print()

        entry = {
            "param_value": float(val),
            "dmc_energy": E0,
            "kappa": kappa,
            "dt_runs": dt_results,
        }

        if "vmc" in (data or {}):
            entry["vmc_energy"] = data["vmc"]["energy"]
            entry["vmc_std_error"] = data["vmc"]["std_error"]

        sweep_results.append(entry)

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

    ax.plot(params,
            [r["dmc_energy"] for r in sweep_results],
            "o-", label=r"DMC $(\delta\tau \to 0)$")

    if "vmc_energy" in sweep_results[0]:
        ax.errorbar(params,
                    [r["vmc_energy"] for r in sweep_results],
                    yerr=[r["vmc_std_error"] for r in sweep_results],
                    fmt="s--", label="VMC", capsize=3)

    ax.set_xlabel(param_path)
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy vs {param_path} (dt extrapolated)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(sweep_dir, "sweep_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


def run_generic_sweep(base_config, param_path, sweep_values):
    """Run a generic parameter sweep."""
    sweep_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown"
    sweep_dir = os.path.join(RESULTS_DIR,
                             f"{sweep_name}_{param_path.replace('.', '_')}")
    os.makedirs(sweep_dir, exist_ok=True)

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
            entry["dmc_energy"] = data["dmc"]["energy"]
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
        ax.plot(params,
                [r["dmc_energy"] for r in sweep_results],
                "o-", label="DMC")

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
    parser.add_argument("--accumulation-blocks", type=int, default=None,
                        help="Override accumulation_blocks for the dt_max run "
                             "(default: from config accumulation_blocks)")
    parser.add_argument("--resume-equilibration-blocks", type=int, default=0,
                        help="Equilibration blocks for the dt_max/4 run that "
                             "resumes from checkpoint (default: 0). Raise if "
                             "the dt change introduces visible transient bias.")
    parser.add_argument("--binary", type=str, default=None,
                        help="Path to QMC binary")
    args = parser.parse_args()

    if args.binary:
        global QMC_EXECUTABLE
        QMC_EXECUTABLE = args.binary

    with open(args.config) as f:
        base_config = json.load(f)

    base_accum_blocks = (args.accumulation_blocks if args.accumulation_blocks is not None
                         else base_config["dmc"].get("accumulation_blocks", 1000))

    has_sweep = len(args.sweep_args) >= 3

    if has_sweep:
        param_path = args.sweep_args[0]
        start = float(args.sweep_args[1])
        stop = float(args.sweep_args[2])
        step = float(args.sweep_args[3]) if len(args.sweep_args) > 3 else 0.5
        sweep_values = np.arange(start, stop + 1e-10, step)

    if has_sweep and args.dt_max is not None:
        run_sweep_with_dt_extrapolation(
            base_config, param_path, sweep_values, args.dt_max, base_accum_blocks,
            args.resume_equilibration_blocks)
    elif args.dt_max is not None:
        run_dt_extrapolation(base_config, args.dt_max, base_accum_blocks,
                             args.resume_equilibration_blocks)
    elif has_sweep:
        run_generic_sweep(base_config, param_path, sweep_values)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
