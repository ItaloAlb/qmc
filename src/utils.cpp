#include "utils.h"
#include "wavefunction.h"
#include <cmath>

namespace Utils {

Metropolis::Metropolis(int seed, double step, int nParticles, int dim)
    : rng(seed), stepSize(step) {
    positionBuffer.resize(nParticles * dim);
}

bool Metropolis::step(WaveFunction& wf, std::vector<double>& currentR, double& currentPsi) {
    positionBuffer = currentR;

    std::uniform_real_distribution<double> dist(-stepSize, stepSize);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (double& position : positionBuffer) {
        position += dist(rng);
    }

    double proposedPsi = wf.trialWaveFunction(positionBuffer.data());

    double ratio = (proposedPsi * proposedPsi) / (currentPsi * currentPsi);

    if (ratio >= 1.0 || uniform(rng) < ratio) {
        currentR = positionBuffer;
        currentPsi = proposedPsi;
        return true;
    }
    return false;
}
}