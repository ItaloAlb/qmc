#include "wavefunction.h"

class HydrogenWF : public WaveFunction {
public:
    HydrogenWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}


    WaveFunction* clone() const override {
        return new HydrogenWF(*this);
    }
    
    double trialWaveFunction(const double* position) const override {
        double alpha = params[0]; 
        
        double r2 = 0.0;

        double r;
        if (pbc) {
            r = pbc->getDistance(position, position + dim);
        } else {
            double r2 = 0.0;
            for (int k = 0; k < dim; ++k) {
                double d = position[k] - position[k + dim];
                r2 += d * d;
            }
            r = std::sqrt(r2);
        }
        return std::exp(-alpha * r);
    }
};