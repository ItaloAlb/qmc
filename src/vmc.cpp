#include "vmc.h"

VMC::VMC(const Hamiltonian& h, WaveFunction& w, Utils::Metropolis& s, int steps, int equil)
    : hamiltonian(h), wf(w), sampler(s), nSteps(steps), nEquilibration(equil) 
{
    stride = h.getStride();
}

void VMC::run() {
    double globalSumEnergy = 0.0;
    double globalSumEnergySq = 0.0;
    double globalAccepted = 0.0;
    double globalSamples = 0.0;
    double finalStepSize = 0.0;

    double initialStep = sampler.getStepSize();
    int nParticles = wf.getNParticles();
    int dim = wf.getDim();

    #pragma omp parallel reduction(+:globalSumEnergy, globalSumEnergySq, globalAccepted, globalSamples, finalStepSize)
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();

        int localSeed = 1234 + threadId; 
        Utils::Metropolis localSampler(localSeed, initialStep, nParticles, dim);

        std::vector<double> currentPosition(nParticles * dim);
        
        std::mt19937 initGen(localSeed);
        std::uniform_real_distribution<double> dist(-100.0, 100.0);
        for(auto& x : currentPosition) x = dist(initGen);

        double currentPsi = wf.trialWaveFunction(currentPosition.data());

        double targetAcc = 0.5;
        double adjustRate = 0.10;
        int adjustInterval = 100;
        int acceptedEquil = 0;

        for (int i = 0; i < nEquilibration; i++) {
            if (localSampler.step(wf, currentPosition, currentPsi)) {
                acceptedEquil++;
            }

            if ((i + 1) % adjustInterval == 0) {
                double accRate = (double)acceptedEquil / adjustInterval;
                double oldStep = localSampler.getStepSize();
                
                double newStep = oldStep * (1.0 + adjustRate * (accRate - targetAcc));

                localSampler.setStepSize(newStep);
                acceptedEquil = 0;
            }
        }
        #pragma omp for nowait
        for (int step = 0; step < nSteps; ++step) {
            
            bool accepted = localSampler.step(wf, currentPosition, currentPsi);
            if (accepted) {
                globalAccepted += 1.0;
            }
            globalSamples += 1.0;

            double localEnergy = hamiltonian.getLocalEnergy(wf, currentPosition.data());

            globalSumEnergy += localEnergy;
            globalSumEnergySq += localEnergy * localEnergy;
        }
        
        finalStepSize += localSampler.getStepSize();
        
    }


    double meanEnergy = globalSumEnergy / globalSamples;
    double meanEnergySq = globalSumEnergySq / globalSamples;
    
    double variance = meanEnergySq - (meanEnergy * meanEnergy);
    double stdError = std::sqrt(variance / globalSamples); 

    result.energy = meanEnergy;
    result.variance = variance;
    result.stdError = stdError;
    result.acceptanceRate = globalAccepted / globalSamples;
    
    int nThreadsUsed = omp_get_max_threads();
    result.metropolisStepSize = finalStepSize / nThreadsUsed;
}