// src/main.cpp
#include <filesystem>

#include "dmc.h"
#include "vmc.h"
#include "optimizer.h"
#include "constants.h"
#include "utils.h"
#include "periodic_boundary.h"

#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "wavefunctions/monolayer_exciton_wf.h"
#include "wavefunctions/monolayer_trion_wf.h"
#include "wavefunctions/monolayer_biexciton_wf.h"
#include "wavefunctions/twisted_bilayer_exciton_wf.h"
#include "wavefunctions/bilayer_exciton_wf.h"
#include "wavefunctions/twisted_bilayer_exciton_gaussian_wf.h"
#include "wavefunctions/exciton_exciton_wf.h"
#include "wavefunctions/exciton_exciton_non_interact_wf.h"
#include "wavefunctions/exciton_in_a_square_potential_wf.h"
#include "wavefunctions/exciton_in_a_triangle_potential_wf.h"

#include "hamiltonians/coulomb_hamiltonian.h"
#include "hamiltonians/efficient_rk_hamiltonian.h"
#include "hamiltonians/twisted_heterobilayer_hamiltonian.h"
#include "hamiltonians/heterobilayer_hamiltonian.h"
#include "hamiltonians/exciton_exciton_coulomb_hamiltonian.h"
#include "hamiltonians/exciton_exciton_non_interact_hamiltonian.h"
#include "hamiltonians/square_hamiltonian.h"
#include "hamiltonians/exciton_triangle_hamiltonian.h"

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
                cfg.dmc.nBlockSteps, cfg.dmc.nStepsPerBlock,
                cfg.dmc.runningAverageWindow);
        DMCResult dmcResult = dmc.run(datFile);
        results["dmc"] = {
            {"energy",    dmcResult.energy},
            {"variance",  dmcResult.variance},
            {"std_error", dmcResult.stdError}
        };

        std::cout << "DMC output\n"
                  << "  Energy    = " << dmcResult.energy    << "\n"
                  << "  Variance  = " << dmcResult.variance  << "\n"
                  << "  StdError  = " << dmcResult.stdError  << "\n" << std::endl;
    }

    std::string resultsFile = cfg.outputFile + "_results.json";
    std::ofstream out(resultsFile);
    out << results.dump(4) << std::endl;
    out.close();
    // std::cout << "Results written to: " << resultsFile << std::endl;

    return 0;
}

// int main() {
//     std::cout << "=================================\n";
//     std::cout << "   Exciton-Exciton Interaction   \n";
//     std::cout << "=================================\n";

//     const PeriodicBoundary* pbc = nullptr;

//     double me = 1.0;
//     double mh = 1.0;
//     double mu = (me * mh) / (me + mh);
//     double d = 0.2 * me / mu;
//     double R = 2.0 * me / mu;

//     std::vector<double> masses = {mu,  mu};
//     std::vector<double> charges = {-1.0, -1.0};

//     std::vector<double> alpha = {1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

//     std::vector<double> params = {3.4722, 10.4458, 13.1154, 19.7361, 0.565659, 1.34458, 6.41809, 1.34458, 6.41809};

//     int nParticles = 2;
//     int nDim = 2;

//     ExcitonExcitonCoulombHamiltonian hamiltonian(nParticles, nDim, masses, charges, me, mh, d, R);
//     ExcitonExcitonWF wf(alpha, nParticles, nDim, me, mh, d, R);

//     wf.setParameters(params);

//     std::cout << "==========\n";
//     std::cout << "   BFGS   \n";
//     std::cout << "==========\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << ", "
//               << optParams[3] << ", "
//               << optParams[4] << ", "
//               << optParams[5] << ", "
//               << optParams[6] << ", "
//               << optParams[7] << ", "
//               << optParams[8] << "]\n\n";



//     std::cout << "=========\n";
//     std::cout << "   VMC   \n";
//     std::cout << "=========\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "=========\n";
//     std::cout << "   DMC   \n";
//     std::cout << "=========\n";

//     double deltaTau = 0.001;
//     bool useFixedNode = true;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, pbc, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "==================================\n";
//     std::cout << "   Exciton-Exciton Non-Interact   \n";
//     std::cout << "==================================\n";

//     const PeriodicBoundary* pbc = nullptr;

//     double me = 1.0;
//     double mh = 1.0;
//     double mu = (me * mh) / (me + mh);
//     double d = 0.2 * me / mu;

//     std::vector<double> masses = {mu,  mu};
//     std::vector<double> charges = {-1.0, -1.0};

//     std::vector<double> alpha = {1.0, -1.0, 1.0, -1.0, 1.0};

//     std::vector<double> params = {1.0, -1.0, 1.0, -1.0, 1.0};

//     int nParticles = 2;
//     int nDim = 2;

//     ExcitonExcitonNonInteractHamiltonian hamiltonian(nParticles, nDim, masses, charges, me, mh, d);
//     ExcitonExcitonNonInteractWF wf(alpha, nParticles, nDim, me, mh, d);

//     wf.setParameters(params);

//     std::cout << "==========\n";
//     std::cout << "   BFGS   \n";
//     std::cout << "==========\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << ", "
//               << optParams[3] << ", "
//               << optParams[4] << "]\n\n";



//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = true;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, pbc, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "=======================\n";
//     std::cout << "   Moiré exciton (X)   \n";
//     std::cout << "=======================\n";

//     const PeriodicBoundary* pbc = nullptr;

//     double thickness = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;
//     double theta = 0.5;
//     double eField = -100.0;

//     TwistedBilayerSystem moire(theta, eField, thickness);

//     double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     std::vector<double> initParams = {35.8, 37.8, 33.5, 36.2};

//     int nParticles = 2;
//     int nDim = 2;

//     TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
//     TwistedBilayerExcitonGaussianWF wf(initParams, nParticles, nDim, 
//         324.8537752460106, 
//         275.7926272419918,
//         324.84973392640273,
//         277.110522144819);


//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);

//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << ", "
//               << params[2] << ", "
//               << params[3] << "]\n\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.1;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, pbc, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "=======================\n";
//     std::cout << "   Moiré exciton (X)   \n";
//     std::cout << "=======================\n";

//     double thickness = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;
//     double theta = 0.5;
//     double eField = -0.0;

//     TwistedBilayerSystem moire(theta, eField, thickness);

//     double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     std::vector<double> initParams = {40.2, 40.2, 37.5, 37.5};

//     int nParticles = 2;
//     int nDim = 2;

//     TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
//     TwistedBilayerExcitonGaussianWF wf(initParams, nParticles, nDim, 
//         243.70652695413165, 
//         140.90486103351043,
//         243.6684947401863,
//         140.77958422241366);


//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);

//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << ", "
//               << params[2] << ", "
//               << params[3] << "]\n\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.1;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }


// int main() {
//     std::cout << "============================\n";
//     std::cout << "   Interlayer exciton (X)   \n";
//     std::cout << "============================\n";

//     double d = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;

//     double rho0 = alpha * 2 * d * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;

//     std::vector<double> initParams = {c1, 0.1, 0.1};

//     std::vector<double> optParams = {-2.30259, -3.175};

//     int nParticles = 2;
//     int nDim = 2;

//     HeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, d, rho0, eps1, eps2);
//     BilayerExcitonWF wf(initParams, nParticles, nDim, d);

//     wf.setParameters(optParams);

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.2, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << "]\n\n";

//     std::cout << "--- Rodando VMC ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "==============\n";
//     std::cout << "   WS2 (XX)   \n";
//     std::cout << "==============\n";

//     double X2D = 6.393 / Constants::a0;
//     double rho0 = 2 * Constants::PI * X2D;

//     std::vector<double> masses = {0.32,  0.32, 0.35, 0.35};
//     std::vector<double> charges = {-1.0, -1.0, 1.0, 1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;
//     std::vector<double> alpha = {c1, 0.1, 0.1, c4, 0.1};

//     std::vector<double> params = {-0.0621872, -3.85307, -2.40632};

//     int nParticles = 4;
//     int nDim = 2;

//     EfficientRKHamiltonian hamiltonian(nParticles, nDim, masses, charges, rho0);
//     MonolayerBiexcitonWF wf(alpha, nParticles, nDim);

//     wf.setParameters(params);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << "]\n\n";



//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = true;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "========================================\n";
//     std::cout << "   DMC TEST: HYDROGEN ATOM (Ground State)\n";
//     std::cout << "========================================\n";

//     std::vector<double> masses = {10000.0, 1.0};
//     std::vector<double> charges = {1.0, -1.0};
//     std::vector<double> alpha = {1.0};


//     CoulombHamiltonian hamiltonian(2, 2, masses, charges);
//     HydrogenWF wf(alpha, 2, 2);

//     double deltaTau = 0.01;
    
//     bool useFixedNode = false;
//     bool useMaxBranch = false;

//     DMC dmc(hamiltonian, wf, deltaTau, 20000, useFixedNode, useMaxBranch);

//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "========================================\n";
//     std::cout << "   QMC TEST: HELIUM ATOM (Ground State)\n";
//     std::cout << "========================================\n";

//     std::vector<double> masses = {1e12, 1.0, 1.0};
//     std::vector<double> charges = {2.0, -1.0, -1.0};
    
//     std::vector<double> alpha = {0.5, 0.5}; 

//     int nParticles = 3;
//     int nDim = 2; 

//     CoulombHamiltonian hamiltonian(nParticles, nDim, masses, charges);
//     HeliumWF wf(alpha, nParticles, nDim);

//     std::random_device rd;
//     unsigned int randomSeed = rd();

//     Metropolis sampler(randomSeed, 1.0, nParticles, nDim);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     JastrowBFGSOptimizer opt(0.001, 100, 100000); 
    
//     opt.optimize(wf, hamiltonian, sampler);

//     std::cout << "Parametros Otimizados: [" 
//               << wf.getParameters()[0] << ", "
//               << wf.getParameters()[1] << "]\n\n";

//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, sampler, 1e5, 1e4); 
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";

//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);

//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "==============\n";
//     std::cout << "   WSe2 (X^-)   \n";
//     std::cout << "==============\n";

//     double X2D =  7.571 / Constants::a0;
//     double rho0 = 2 * Constants::PI * X2D;

//     std::vector<double> masses = {0.34,  0.34, 0.36};
//     std::vector<double> charges = {-1.0, -1.0, 1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;
//     std::vector<double> alpha = {c1, 0.1, 0.1, c4, 0.1};

//     std::vector<double> params = {-0.0621872, -3.85307, -2.40632};

//     int nParticles = 3;
//     int nDim = 2;

//     EfficientRKHamiltonian hamiltonian(nParticles, nDim, masses, charges, rho0);
//     MonolayerTrionWF wf(alpha, nParticles, nDim);

//     wf.setParameters(params);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.05, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << "]\n\n";



//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }