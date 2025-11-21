#ifndef VMC_H
#define VMC_H

#include <vector>
#include <random>
#include <omp.h>
#include <numeric>
#include <memory>

#include "hamiltonian.h"
#include "wavefunction.h"


struct VMCResult {
    double energy;
    double variance;
    double stdError;
    double acceptanceRate;
    double metropolisStepSize;
};

class VMC {
    private:
        const Hamiltonian& hamiltonian;
        WaveFunction& wf;

        int nParticle, nDim, nSteps, nEquilibration, stride;

        std::vector<std::mt19937> gens; 
        std::uniform_real_distribution<double> uniformDist;

    public:
        VMCResult result;
        VMC(const Hamiltonian& hamiltonian, 
            WaveFunction& wf, 
            int nSteps, 
            int nEquilibration);

        VMCResult run(const std::vector<double>& alpha, std::mt19937& local_rng);

        std::vector<double> optimizeParameters(
            const std::vector<double>& alphaStart, 
            const std::vector<double>& alphaEnd, 
            const std::vector<double>& alphaStep,
            bool isMinimizeVariance);
};

#endif