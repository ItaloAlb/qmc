#include "wavefunction.h"

class HeliumWF : public WaveFunction {
public:
    HeliumWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}

    WaveFunction* clone() const override {
        return new HeliumWF(*this);
    }

    double trialWaveFunction(const double* position) const override {
        double alpha = params[0];
        double beta = params[1];

        const double* p0   = &position[0];
        const double* p1   = &position[dim];
        const double* p2   = &position[2*dim];
        
        double r01 = 0.0;
        double r02 = 0.0;
        double r12 = 0.0;

        for (int k = 0; k < dim; k++) {
            double d1 = p1[k] - p0[k];
            r01 += d1 * d1;

            double d2 = p2[k] - p0[k];
            r02 += d2 * d2;

            double d12 = p1[k] - p2[k];
            r12 += d12 * d12;
        }
        double r1 = std::sqrt(r01);
        double r2 = std::sqrt(r02);
        double r3 = std::sqrt(r12);
        return std::exp(-alpha * (r1 + r2)) * std::exp(0.5 * r3 / (1 + beta * r3));
    }
};