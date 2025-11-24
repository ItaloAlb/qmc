#include "wavefunction.h"

class MonolayerExcitonWF : public WaveFunction {
public:
    MonolayerExcitonWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}


    WaveFunction* clone() const override {
        return new MonolayerExcitonWF(*this);
    }

    double jastrowFactor(double r, double r2) const {
        double c1 = params[0];
        double c2 = params[1];
        double c3 = params[2];

        double argexp = - c2 * r2;
        double exp = std::exp(argexp);
        
        return std::exp(c1 * r2 * std::log(r) * exp - c3 * r * (1 - exp));
    }
    
    double trialWaveFunction(const double* position) const override {
        double r2 = 0.0;

        for(int k = 0; k < dim; k++) {
            double dist = position[k] - position[k + dim];
            r2 += dist * dist;
        }
        double r = std::sqrt(r2);
        return jastrowFactor(r, r2);
    }
};