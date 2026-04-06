#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <fstream>
#include <stdexcept>
#include <memory>
#include "nlohmann/json.hpp"
#include "hamiltonian.h"
#include "wavefunction.h"
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
};

struct System {
    std::unique_ptr<Hamiltonian>  hamiltonian;
    std::unique_ptr<WaveFunction> wf;
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
            d.at("max_branch")
        };

        return cfg;
    }
};

#endif