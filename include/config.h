#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <fstream>
#include <stdexcept>
#include <memory>
#include "nlohmann/json.hpp"
#include "constants.h"
#include "hamiltonian.h"
#include "wavefunction.h"
#include "periodic_boundary.h"
#include "utils.h"

using json = nlohmann::json;

struct OptimizerConfig {
    bool enabled;
    double learningRate;
    int maxEpochs;
    int samplesPerEpoch;
};

struct VMCConfig {
    bool enabled;
    int nSteps;
    int nEquilibration;
};

struct DMCConfig {
    bool enabled;
    double deltaTau;
    bool fixedNode;
    bool maxBranch;
    bool dumpWalkers;
    bool descendantWeighting;
    int tLagBlocks;
    int taggingIntervalBlocks;
    int equilibrationBlocks;
    int accumulationBlocks;
    int nStepsPerBlock;
    int nWalkersTarget;
};

struct System {
    std::unique_ptr<Hamiltonian>      hamiltonian;
    std::unique_ptr<WaveFunction>     wf;
    std::unique_ptr<PeriodicBoundary> pbc;   // optional, may be null
    int nParticles;
    int nDim;
};

struct QMCConfig {
    std::string     systemName;
    json            params;      // raw — passed to the system builder
    OptimizerConfig optimizer;
    VMCConfig       vmc;
    DMCConfig       dmc;
    std::string     outputFile;

    static QMCConfig fromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open config: " + path);

        json j = json::parse(f);
        QMCConfig cfg;

        cfg.systemName  = j.at("system");
        cfg.params      = j.at("params");
        cfg.outputFile  = j.at("output").at("file");

        auto& o = j.at("optimizer");
        cfg.optimizer = {
            o.at("enabled"),
            o.at("learning_rate"),
            o.at("max_epochs"),
            o.at("samples_per_epoch")
        };

        auto& v = j.at("vmc");
        cfg.vmc = { v.at("enabled"), v.at("n_steps"), v.at("n_equilibration") };

        auto& d = j.at("dmc");
        cfg.dmc = {
            d.at("enabled"),
            d.at("delta_tau"),
            d.at("fixed_node"),
            d.at("max_branch"),
            d.value("dump_walkers", false),
            d.value("descendant_weighting", false),
            d.value("t_lag_blocks", 10),
            d.value("tagging_interval_blocks", 1),
            d.value("equilibration_blocks", 200),
            d.value("accumulation_blocks", 800),
            d.value("n_steps_per_block", 100),
            d.value("n_walkers_target", Constants::N_WALKERS_TARGET)
        };

        return cfg;
    }
};

#endif