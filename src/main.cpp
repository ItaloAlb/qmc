// src/main.cpp
#include "dmc.h"
#include "vmc.h"
#include "constants.h"

#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "wavefunctions/monolayer_exciton_wf.h"

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
//     std::vector<double> alpha = {1.0, 1.0};


//     std::vector<double> alphaStart = {0.1, 0.1};
//     std::vector<double> alphaEnd   = {2.0, 2.0};
//     std::vector<double> alphaStep  = {0.1, 0.1};


//     CoulombHamiltonian hamiltonian(3, 2, masses, charges);
//     // HydrogenWF wf(alpha, 2, 2);
//     HeliumWF wf(alpha, 3, 2);

//     double deltaTau = 0.0001;
    
//     bool useFixedNode = false;
//     bool useMaxBranch = true;
//     bool isMinimizingVariance = true;

//     VMC vmc(hamiltonian, wf, 1e5, 1e4);
//     wf.setParameters(vmc.optimizeParameters(alphaStart, alphaEnd, alphaStep, isMinimizingVariance));

//     std::cout << "Parameters: [" << wf.getParameters()[0] << ", "
//                                << wf.getParameters()[1] << "]\n";

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";

//     DMC dmc(hamiltonian, wf, deltaTau, 20000, useFixedNode, useMaxBranch);

//     dmc.run();

//     return 0;
// }

int main() {
    std::cout << "==============\n";
    std::cout << "   MoS2 (X)   \n";
    std::cout << "==============\n";


    double X2D = 7.112;

    double rho0 = 2 * Constants::PI * X2D;

    std::vector<double> masses = {0.54, 0.47};
    std::vector<double> charges = {+1.0, -1.0};

    double c1 = masses[0] * masses[1] / 2 / (masses[0] + masses[1]);
    std::vector<double> alpha = {c1, 1.0, 1.0};


    std::vector<double> alphaStart = {c1, 0.1, 0.1};
    std::vector<double> alphaEnd   = {c1, 2.0, 2.0};
    std::vector<double> alphaStep  = {0.1, 0.1, 0.1};


    EfficientRKHamiltonian hamiltonian(2, 2, masses, charges, rho0);
    MonolayerExcitonWF wf(alpha, 2, 2);

    double deltaTau = 0.01;
    
    bool useFixedNode = false;
    bool useMaxBranch = true;
    bool isMinimizingVariance = true;

    VMC vmc(hamiltonian, wf, 1e5, 1e4);
    wf.setParameters(vmc.optimizeParameters(alphaStart, alphaEnd, alphaStep, isMinimizingVariance));

    std::cout << "Parameters: [" << wf.getParameters()[1] << ", "
                               << wf.getParameters()[2] << "]\n";

    std::cout << "Energy: "             << vmc.result.energy             << "\n";
    std::cout << "Variance: "           << vmc.result.variance           << "\n";
    std::cout << "StdError: "           << vmc.result.stdError           << "\n";
    std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
    std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";

    DMC dmc(hamiltonian, wf, deltaTau, 5000, useFixedNode, useMaxBranch);

    dmc.run();

    return 0;
}