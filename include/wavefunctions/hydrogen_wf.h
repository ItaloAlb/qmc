#include "wavefunction.h"

class HydrogenWF : public WaveFunction {
public:
    HydrogenWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}

    double trialWaveFunction(const double* position) const override {
        double alpha = params[0]; 
        
        double r2 = 0.0;

        for(int k = 0; k < dim; k++) {
            double dist = position[k] - position[k + dim];
            r2 += dist * dist;
        }
        double r = std::sqrt(r2);
        return std::exp(-alpha * r);
    }
};