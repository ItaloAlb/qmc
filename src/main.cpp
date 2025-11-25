// src/main.cpp
#include "dmc.h"
#include "vmc.h"
#include "optimizer.h"
#include "constants.h"

#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "wavefunctions/monolayer_exciton_wf.h"
#include "wavefunctions/monolayer_trion_wf.h"

#include "hamiltonians/coulomb_hamiltonian.h"
#include "hamiltonians/efficient_rk_hamiltonian.h"

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

int main() {
    std::cout << "==============\n";
    std::cout << "   WS2 (X^-) \n";
    std::cout << "==============\n";

    double X2D =  6.393 / Constants::a0;
    double rho0 = 2 * Constants::PI * X2D;

    std::vector<double> masses = {0.32 ,  0.32 , 0.35};
    std::vector<double> charges = {-1.0, -1.0, 1.0};

    double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
    double c4 = - masses[0] / 4;
    std::vector<double> alpha = {c1, 0.2, 0.2, c4, 0.2};

    // std::vector<double> params = {-1.59201, -1.61131, -1.6036};

    int nParticles = 3;
    int nDim = 2;

    EfficientRKHamiltonian hamiltonian(nParticles, nDim, masses, charges, rho0);
    MonolayerTrionWF wf(alpha, nParticles, nDim);

    // wf.setParameters(params);

    std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

    std::random_device rd;
    unsigned int randomSeed = rd();
    
    Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

    JastrowBFGSOptimizer optVariance(0.01, 100, 100000, false);
    optVariance.optimize(wf, hamiltonian, optimizerSampler);


    std::vector<double> optParams = wf.getParameters();
    std::cout << "Parametros Otimizados (log): [" 
              << optParams[0] << ", "
              << optParams[1] << ", "
              << optParams[2] << "]\n\n";



    std::cout << "--- Rodando VMC de Producao ---\n";
    
    VMC vmc(hamiltonian, wf, optimizerSampler, 1e5, 1e4);
    vmc.run();

    std::cout << "Energy: "             << vmc.result.energy             << "\n";
    std::cout << "Variance: "           << vmc.result.variance           << "\n";
    std::cout << "StdError: "           << vmc.result.stdError           << "\n";
    std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
    std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


    std::cout << "--- Rodando DMC ---\n";
    double deltaTau = 0.01;
    bool useFixedNode = false;
    bool useMaxBranch = true;

    DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
    dmc.run();

    return 0;
}