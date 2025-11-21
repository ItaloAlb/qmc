// src/main.cpp
#include "dmc.h"
#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "hamiltonians/coulomb_hamiltonian.h"

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

int main() {
    std::cout << "========================================\n";
    std::cout << "   DMC TEST: HELIUM ATOM (Ground State)\n";
    std::cout << "========================================\n";

    std::vector<double> masses = {100000.0, 1.0, 1.0};
    std::vector<double> charges = {2.0, -1.0, -1.0};
    std::vector<double> alpha = {1.0, 1.0};


    CoulombHamiltonian hamiltonian(3, 3, masses, charges);
    // HydrogenWF wf(alpha, 2, 2);
    HeliumWF wf(alpha, 3, 3);

    double deltaTau = 0.01;
    
    bool useFixedNode = false;
    bool useMaxBranch = true;

    DMC dmc(hamiltonian, wf, deltaTau, 20000, useFixedNode, useMaxBranch);

    dmc.run();

    return 0;
}