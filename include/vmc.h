#include <vector>
#include <random>

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
        const WaveFunction& wf;

        int nParticle, nDim, nSteps, nEquilibration, stride;

        std::vector<std::mt19937> gens; 
        std::uniform_real_distribution<double> uniformDist;

    public:
        VMCResult result;
        VMC(int nParticle, int nDim, int nSteps, int nEquilibration);
        VMCResult run(const double* alpha, std::mt19937& local_rng);
        void localEnergyPerStep(const double* alpha, std::mt19937& local_rng, const std::string& filename);
        std::vector<double> optimizeParameters(
            const std::vector<double>& alphaStart, 
            const std::vector<double>& alphaEnd, 
            const std::vector<double>& alphaStep,
            bool isMinimizeVariance);
};