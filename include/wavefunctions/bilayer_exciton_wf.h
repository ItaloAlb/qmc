#include "wavefunction.h"
#include "utils.h"
#include "complex"

using namespace std::complex_literals;

class BilayerExcitonWF : public WaveFunction {
private:
    double d, dSquared;
public:
    BilayerExcitonWF(const std::vector<double>& params, int nParticles, int dim, double dst_) 
        : WaveFunction(params, nParticles, dim), d(dst_ / Constants::a0) {
            dSquared = d * d;
        }

    WaveFunction* clone() const override {
        return new BilayerExcitonWF(*this);
    }

void setParameters(const std::vector<double>& newParams) override {
        if (newParams.size() == 2) {
            double p_c2 = newParams[0];
            double p_c3 = newParams[1];


            params[1] = std::exp(p_c2);
            params[2] = std::exp(p_c3);
        }
    }

    std::vector<double> getParameters() const override {
        return { 
            std::log(params[1]),
            std::log(params[2])
        };
    }

    double jastrowEH(double r, double r2) const {
        double c1 = params[0];
        double c2 = params[1];
        double c3 = params[2];

        double exp = std::exp(-c2 * r2);
        double term1 = c1 * r2 * std::log(r) * exp;
        double term2 = c3 * r * (1.0 - exp);
        return std::exp(term1 - term2);
    }

    
    double trialWaveFunction(const double* position) const override {
        int idx_e = 0;
        int idx_h = 1 * dim;

        double r2 = dSquared;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e + k] - position[idx_h + k];
            r2 += d * d;
        }
        double r = std::sqrt(r2);

        return jastrowEH(r, r2);
    }
};