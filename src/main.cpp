// src/main.cpp
#include "dmc.h"
#include "vmc.h"
#include "optimizer.h"
#include "constants.h"
#include "utils.h"

#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "wavefunctions/monolayer_exciton_wf.h"
#include "wavefunctions/monolayer_trion_wf.h"
#include "wavefunctions/monolayer_biexciton_wf.h"
#include "wavefunctions/twisted_bilayer_exciton_wf.h"

#include "hamiltonians/coulomb_hamiltonian.h"
#include "hamiltonians/efficient_rk_hamiltonian.h"
#include "hamiltonians/twisted_heterobilayer_hamiltonian.h"

int main() {
    std::cout << "=======================\n";
    std::cout << "   Moiré exciton (X)   \n";
    std::cout << "=======================\n";

    double thickness = 6.15 / Constants::a0;
    double alpha = 1.5;
    double eps = 14.0;
    double eps1 = 4.5;
    double eps2 = 4.5;
    double theta = 0.5;
    double eField = 0.0;

    TwistedBilayerSystem moire(theta, eField, thickness);

    // double X2D = 6.393 / Constants::a0;
    double rho0 = alpha * thickness * eps / (eps1 + eps2);

    double me = 0.43;
    double mh = 0.35;

    std::vector<double> masses = {me,  mh};
    std::vector<double> charges = {-1.0, +1.0};

    double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
    double c4 = - masses[0] / 4;

    std::vector<double> initParams = {c1, 0.1, 0.1, 0.1, 0.1, 0.1};

    std::vector<double> optParams = {-1.30937, -3.30797, -0.0144402, -0.0144402, -0.0144402};

    int nParticles = 2;
    int nDim = 2;

    TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
    TwistedBilayerExcitonWF wf(initParams, nParticles, nDim, moire);

    wf.setParameters(optParams);

    std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

    std::random_device rd;
    unsigned int randomSeed = rd();
    
    Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

    JastrowBFGSOptimizer optVariance(0.3, 50, 1e5);
    optVariance.optimize(wf, hamiltonian, optimizerSampler);


    optParams = wf.getParameters();
    std::cout << "Parametros Otimizados: [" 
              << optParams[0] << ", "
              << optParams[1] << ", "
              << optParams[2] << ", "
              << optParams[3] << ", "
              << optParams[4] << "]\n\n";



    std::cout << "--- Rodando VMC de Producao ---\n";
    
    VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
    vmc.run();

    std::cout << "Energy: "             << vmc.result.energy             << "\n";
    std::cout << "Variance: "           << vmc.result.variance           << "\n";
    std::cout << "StdError: "           << vmc.result.stdError           << "\n";
    std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
    std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


    std::cout << "--- Rodando DMC ---\n";
    double deltaTau = 0.05;
    bool useFixedNode = false;
    bool useMaxBranch = true;

    DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
    dmc.run();

    return 0;
}

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