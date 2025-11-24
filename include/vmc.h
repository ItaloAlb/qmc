#ifndef VMC_H
#define VMC_H

#include <vector>
#include <random>
#include <omp.h>
#include <numeric>
#include <memory>

#include "hamiltonian.h"
#include "wavefunction.h"
#include "utils.h"


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
    Utils::Metropolis& sampler;

    int nSteps;
    int nEquilibration;
    int stride;

public:
    VMCResult result;

    VMC(const Hamiltonian& hamiltonian, 
        WaveFunction& wf, 
        Utils::Metropolis& sampler,
        int nSteps, 
        int nEquilibration);

    void run();
};
#endif