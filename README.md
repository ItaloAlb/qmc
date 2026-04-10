# QMC Engine

Variational (VMC) and Diffusion (DMC) Monte Carlo engine for excitonic systems.

## Build

```bash
cmake -B build -S .
cmake --build build
```

The binary is placed at `build/bin/qmc`.

## Running

```bash
./build/bin/qmc configs/exciton_in_a_square_potential.json
```

The config file controls everything: which system to simulate, optimizer/VMC/DMC parameters, and output paths.

Output files are written relative to the working directory:
- `<output_file>.dat` — block-by-block DMC energies
- `<output_file>_results.json` — final energies (VMC and/or DMC) and parameters
- `<output_file>_walkers.bin` — walker positions (if `dump_walkers: true`)
- `<output_file>_descendants.bin` — descendant weighting data (if `descendant_weighting: true`)

## Config file structure

```jsonc
{
    "system": "exciton_in_a_square_potential",   // registered name in system_config.h
    "params": {
        // system-specific physics parameters (masses, charges, potentials, ...)
        // optional "pbc" block for periodic boundaries
        "wf_params_init": [...]   // initial wavefunction variational parameters
    },
    "output": {
        "file": "results/my_system/output"
    },
    "optimizer": {
        "enabled": true,
        "learning_rate": 0.1,
        "max_epochs": 50,
        "samples_per_epoch": 100000
    },
    "vmc": {
        "enabled": true,
        "n_steps": 10000000,
        "n_equilibration": 1000000
    },
    "dmc": {
        "enabled": true,
        "delta_tau": 0.01,
        "fixed_node": true,
        "max_branch": true,
        "n_block_steps": 1000,
        "n_steps_per_block": 100,
        "running_average_window": 100,
        "dump_walkers": false,
        "descendant_weighting": false,
        "t_lag_blocks": 10,
        "tagging_interval_blocks": 1
    }
}
```

### DMC parameters

| Parameter | Description |
|---|---|
| `delta_tau` | Imaginary time step |
| `fixed_node` | Enforce fixed-node approximation |
| `max_branch` | Cap branching multiplicity |
| `n_block_steps` | Total number of blocks |
| `n_steps_per_block` | Time steps per block (block_time = n_steps_per_block * delta_tau) |
| `running_average_window` | Blocks used for final energy average |
| `dump_walkers` | Write walker positions to binary file |
| `descendant_weighting` | Enable overlapping descendant weighting for pure |Phi_0|^2 sampling |
| `t_lag_blocks` | Propagation time (in blocks) before counting descendants |
| `tagging_interval_blocks` | How often (in blocks) to tag a new generation of walkers |

## Adding a new system

A system requires three things: a Hamiltonian, a WaveFunction, and a builder function registered in `system_config.h`.

### 1. Wavefunction

Create `include/wavefunctions/my_system_wf.h` inheriting from `WaveFunction`:

```cpp
#include "wavefunction.h"

class MySystemWF : public WaveFunction {
public:
    MySystemWF(const std::vector<double>& params, int nParticles, int dim)
        : WaveFunction(params, nParticles, dim) {}

    WaveFunction* clone() const override {
        return new MySystemWF(*this);
    }

    double trialWaveFunction(const double* position) const override {
        // Return Psi_T(position)
    }
};
```

The base class provides `getDrift`, `getLaplacian`, and `parameterGradient` via finite differences. Override them for analytical implementations.

### 2. Hamiltonian

Create `include/hamiltonians/my_system_hamiltonian.h` inheriting from `Hamiltonian`:

```cpp
#include "hamiltonian.h"

class MySystemHamiltonian : public Hamiltonian {
public:
    MySystemHamiltonian(int nParticles, int dim,
                        const std::vector<double>& masses,
                        const std::vector<double>& charges)
        : Hamiltonian(nParticles, dim, masses, charges) {}

    double getPotential(const double* position) const override {
        // Return V(position) in Hartree
    }
};
```

The kinetic energy is computed by the base class using the wavefunction's Laplacian.

### 3. Register the system

In `include/system_config.h`:

1. Include the new headers:
```cpp
#include "wavefunctions/my_system_wf.h"
#include "hamiltonians/my_system_hamiltonian.h"
```

2. Add a builder function that reads parameters from JSON and returns a `System`:
```cpp
System MySystem(const json& p) {
    double me = p.at("me");
    // ... read other params ...

    std::vector<double> masses  = { me };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> initP   = p.at("wf_params_init").get<std::vector<double>>();

    int nP = 1, nD = 2;
    auto ham = std::make_unique<MySystemHamiltonian>(nP, nD, masses, charges);
    auto wf  = std::make_unique<MySystemWF>(initP, nP, nD);
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}
```

3. Register the name in `buildSystem()`:
```cpp
else if (name == "my_system") return MySystem(p);
```

4. Include the wavefunction and hamiltonian headers in `src/main.cpp`.

### 4. Config file

Create `configs/my_system.json` using the structure above. The `"system"` field must match the name registered in `buildSystem()`.

## Analysis scripts

### Parameter sweep (`run_sweep.py`)

Generic sweep over any config parameter:

```bash
python3 run_sweep.py configs/exciton_exciton.json params.R 1.0 6.0 0.5
```

When sweeping `dmc.delta_tau`, the script automatically adjusts `n_steps_per_block` to keep `block_time` constant.

#### Time step extrapolation

Uses the protocol from Lee's thesis (Section 1.5): two runs at `dt_max` and `dt_max/4` with optimal effort ratio 1:8.

```bash
# Find dt_max first
python3 find_dt_max.py configs/my_system.json --dt-values 0.001 0.005 0.01 0.02 0.05

# Run the extrapolation
python3 run_sweep.py configs/my_system.json --dt-max 0.02
```

The `--dt-max` flag:
- Runs DMC at `dt_max` (1/9 of total blocks) and `dt_max/4` (8/9 of total blocks)
- Keeps `block_time` constant across runs
- Extrapolates `E(dt) -> E(0)` with error propagation
- Total effort defaults to `9 * n_block_steps` from the config (override with `--total-blocks`)

### Find dt_max (`find_dt_max.py`)

Sweeps multiple time steps and identifies where `E(dt)` departs from linearity:

```bash
python3 find_dt_max.py configs/my_system.json
python3 find_dt_max.py configs/my_system.json --dt-min 0.001 --dt-max 0.1 --n-points 8
python3 find_dt_max.py configs/my_system.json --dt-values 0.001 0.002 0.005 0.01 0.02 0.05
```

Outputs `dt_max`, a summary JSON, and a plot of `E(dt)` vs `dt` with residuals.

### Descendant weighting density (`plot_descendants.py`)

Visualizes the pure ground state density |Phi_0|^2 from descendant weighting data:

```bash
python3 plot_descendants.py results/my_system/output_descendants.bin
python3 plot_descendants.py results/my_system/output_descendants.bin --bins 80 -o density.png
```

For 2D exciton systems (stride=4), plots electron density, hole density, and relative e-h density.

### Walker distribution (`plot_walkers.py`)

Plots the mixed distribution f = Psi_T * Phi_0 from dumped walker positions:

```bash
python3 plot_walkers.py results/my_system/output_walkers.bin
```
