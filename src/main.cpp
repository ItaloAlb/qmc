// src/main.cpp
#include <filesystem>

#include "dmc.h"
#include "vmc.h"
#include "optimizer.h"
#include "constants.h"
#include "utils.h"
#include "periodic_boundary.h"
#include "wavefunctions/wavefunction_pool.h"
#include "hamiltonians/hamiltonian_pool.h"

#include "system_config.h"

int main(int argc, char* argv[]) {
    std::string configPath = argc > 1 ? argv[1] : "config.json";
    QMCConfig cfg = QMCConfig::fromFile(configPath);

    System sys = buildSystem(cfg);
    auto& ham  = *sys.hamiltonian;
    auto& wf   = *sys.wf;

    std::random_device rd;
    Metropolis sampler(rd(), 1.0, sys.nParticles, sys.nDim);

    auto outputDir = std::filesystem::path(cfg.outputFile).parent_path();
    if (!outputDir.empty())
        std::filesystem::create_directories(outputDir);

    json results;
    results["params"] = cfg.params;
    results["initial_params"] = wf.getParameters();

    if (cfg.optimizer.enabled) {
        auto& o = cfg.optimizer;
        LinearMethodOptimizer opt(o.maxEpochs, o.samplesPerEpoch);
        opt.optimize(wf, ham, sampler);

        std::vector<double> optimized = wf.getParameters();
        results["optimized_params"] = optimized;

        std::cout << "\nOptimized wavefunction parameters:\n[";
        for (size_t i = 0; i < optimized.size(); ++i) {
            std::cout << optimized[i];
            if (i + 1 < optimized.size()) std::cout << ", ";
        }
        std::cout << "]\n" << std::endl;
    }

    if (cfg.vmc.enabled) {
        VMC vmc(ham, wf, sampler, cfg.vmc.nSteps, cfg.vmc.nEquilibration);
        vmc.run();
        results["vmc"] = {
            {"energy",    vmc.result.energy},
            {"variance",  vmc.result.variance},
            {"std_error", vmc.result.stdError},
            {"acceptance_rate",      vmc.result.acceptanceRate},
            {"metropolis_step_size", vmc.result.metropolisStepSize}
        };

        std::cout << "VMC output:\n"
                  << "  Energy    = " << vmc.result.energy    << "\n"
                  << "  Variance  = " << vmc.result.variance  << "\n"
                  << "  StdError  = " << vmc.result.stdError  << "\n"
                  << "  AccRate   = " << vmc.result.acceptanceRate << "\n"
                  << "  StepSize  = " << vmc.result.metropolisStepSize << "\n" << std::endl;
    }

    if (cfg.dmc.enabled) {
        std::string datFile = cfg.outputFile + ".dat";
        DMC dmc(ham, wf, cfg.dmc.deltaTau, sys.pbc.get(),
                cfg.dmc.nWalkersTarget, cfg.dmc.fixedNode, cfg.dmc.maxBranch,
                cfg.dmc.dumpWalkers, cfg.dmc.descendantWeighting,
                cfg.dmc.tLagBlocks, cfg.dmc.taggingIntervalBlocks,
                cfg.dmc.equilibrationBlocks, cfg.dmc.accumulationBlocks,
                cfg.dmc.nStepsPerBlock);
        DMCResult dmcResult = dmc.run(datFile);
        results["dmc"] = {
            {"energy",    dmcResult.energy},
            {"variance",  dmcResult.variance}
        };

        std::cout << "DMC output\n"
                  << "  Energy    = " << dmcResult.energy    << "\n"
                  << "  Variance  = " << dmcResult.variance  << "\n" << std::endl;
    }

    std::string resultsFile = cfg.outputFile + "_results.json";
    std::ofstream out(resultsFile);
    out << results.dump(4) << std::endl;
    out.close();
    std::cout << "Results written to: " << resultsFile << std::endl;

    return 0;
}