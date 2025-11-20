#ifndef DMC_H
#define DMC_H

#include <vector>
#include <random>
#include <omp.h>
#include <deque>
#include <numeric>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "constants.h"
#include "hamiltonian.h"
#include "wavefunction.h"
#include "utils.h"


struct BlockResult {
    double energy;
    double variance;
    double stdError;
};

struct DMCResult {
    double energy;
    double variance;
    double stdError;
};

class DMC {
    private:
        const Hamiltonian& hamiltonian;
        const WaveFunction& wf;

        int nWalkers, nParticles, dim, stride;
        double deltaTau, referenceEnergy, instEnergy, meanEnergy;
        bool isFixedNode, isMaxBranch;

        std::vector<std::mt19937> gens;
        std::vector<double> positions;
        std::vector<double> drifts;
        std::vector<double> localEnergy;

        void initializeWalkers();
        void updateReferenceEnergy(double blockEnergy, double blockTime);

        double driftGreenFunction(const double* newPosition, const double* oldPosition, const double* oldDrift) const;
        double branchGreenFunction(double newLocalEnergy, double oldLocalEnergy) const;

        void timeStep();
        BlockResult blockStep(int nSteps);

    public: 
        DMC(const Hamiltonian& hamiltonian, 
            const WaveFunction& wf, 
            double deltaTau, 
            int nWalkers = Constants::N_WALKERS_TARGET,
            bool isFixedNode = false,
            bool isMaxBranch = false);

        void run();
};

#endif