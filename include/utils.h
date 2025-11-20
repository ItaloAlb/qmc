#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <cmath>
#include <functional>

#include "wavefunction.h"


namespace Utils {

    inline bool metropolisStep(
        const double* currentPos,
        double* proposedPos,
        double& currentPsi, 
        int stride,
        std::mt19937& gen,
        double step,
        const std::function<double(const double*)>& psiFunc 
    ) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::uniform_real_distribution<double> delta(-step, +step);

        for (int i = 0; i < stride; i++)
            proposedPos[i] = currentPos[i] + delta(gen);

        double psiProposed = psiFunc(proposedPos);

        double w = (psiProposed * psiProposed) / (currentPsi * currentPsi);

        if (w >= uniform(gen)) {
            currentPsi = psiProposed;
            return true;
        }
        return false;
    }
}

#endif

