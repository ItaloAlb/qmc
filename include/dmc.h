#ifndef DMC_H
#define DMC_H

#include <vector>
#include <random>
#include <omp.h>
#include <deque>
#include <numeric>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "constants.h"
#include "hamiltonian.h"
#include "wavefunction.h"
#include "periodic_boundary.h"
#include "utils.h"


struct BlockResult {
    double energy;
    double variance;
    double stdError;
    double acceptanceRatio;
};

struct DMCResult {
    double energy;
    double variance;
    double stdError;
};

class DMC {
    private:
        const Hamiltonian& hamiltonian;
        WaveFunction& wf;
        const PeriodicBoundary* pbc;

        int nWalkers, nWalkersTarget, nParticles, dim, stride, blockTotalMoves, blockAcceptedMoves;
        int nBlockSteps, nStepsPerBlock, runningAverageWindow;
        int tLagBlocks;
        double deltaTau, invDeltaTau, referenceEnergy, instEnergy, meanEnergy;
        bool isFixedNode, isMaxBranch, dumpWalkers, descendantWeighting;

        std::vector<std::mt19937> gens;
        std::vector<double> positions;
        std::vector<double> drifts;
        std::vector<double> localEnergy;
        std::vector<double> newPositionsScratch;
        std::vector<double> newDriftsScratch;
        std::vector<double> newLocalEnergiesScratch;
        std::deque<std::vector<int>> ancestorsHistory;
        std::deque<std::vector<int>> newAncestorsHistory;
        std::deque<std::vector<double>> taggedPositionsHistory;
        std::deque<int> taggedCountHistory;
        std::deque<int> taggingBlocksHistory;
        int taggingIntervalBlocks;

        void initializeWalkers();
        void updateReferenceEnergy(double blockEnergy, double blockTime);

        double driftGreenFunction(const double* newPosition, const double* oldPosition, const double* oldDrift) const;
        double branchGreenFunction(double newLocalEnergy, double oldLocalEnergy) const;

        void timeStep();
        BlockResult blockStep(int nSteps);

    public:
        DMC(const Hamiltonian& hamiltonian,
            WaveFunction& wf,
            double deltaTau,
            const PeriodicBoundary* pbc = nullptr,
            int nWalkersTarget = Constants::N_WALKERS_TARGET,
            bool isFixedNode = false,
            bool isMaxBranch = false,
            bool dumpWalkers = false,
            bool descendantWeighting = false,
            int tLagBlocks = 10,
            int taggingIntervalBlocks = 1,
            int nBlockSteps = 1000,
            int nStepsPerBlock = 100,
            int runningAverageWindow = 100);

        DMCResult run(const std::string& outputFile = "qmc.dat");
};

#endif
