#include "wavefunction.h"

class MonolayerExcitonWF : public WaveFunction {
public:
    MonolayerExcitonWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}


    WaveFunction* clone() const override {
        return new MonolayerExcitonWF(*this);
    }

    void setParameters(const std::vector<double>& newParams) override {
        if (newParams.size() == 2) {
            double p2 = newParams[0];
            double p3 = newParams[1];

            params[1] = std::exp(p2);
            params[2] = std::exp(p3);
        }
    }

    std::vector<double> getParameters() const override {

        return { std::log(params[1]), std::log(params[2]) };
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