#include "wavefunction.h"

class MonolayerTrionWF : public WaveFunction {
public:
    MonolayerTrionWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}

    WaveFunction* clone() const override {
        return new MonolayerTrionWF(*this);
    }

void setParameters(const std::vector<double>& newParams) override {
        if (newParams.size() == 3) {
            double p_c2 = newParams[0];
            double p_c3 = newParams[1];
            double p_c5 = newParams[2];

            params[1] = std::exp(p_c2);
            params[2] = std::exp(p_c3);
            params[4] = std::exp(p_c5);
        }
    }

    std::vector<double> getParameters() const override {
        return { 
            std::log(params[1]),
            std::log(params[2]),
            std::log(params[4])
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

    double jastrowEE(double r, double r2) const {
        double c4 = params[3];
        double c5 = params[4];

        double exparg = c4 * r2 * std::log(r) * std::exp(-c5 * r2);
        return std::exp(exparg);
    }
    
    double trialWaveFunction(const double* position) const override {
        // Índices no vetor position (assumindo layout flat: [x1, y1, x2, y2, xh, yh] para 2D)
        int idx_e1 = 0;
        int idx_e2 = 1 * dim;
        int idx_h  = 2 * dim;

        double r2_e1h = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e1 + k] - position[idx_h + k];
            r2_e1h += d * d;
        }
        double r_e1h = std::sqrt(r2_e1h);

        double r2_e2h = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e2 + k] - position[idx_h + k];
            r2_e2h += d * d;
        }
        double r_e2h = std::sqrt(r2_e2h);

        double r2_ee = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e1 + k] - position[idx_e2 + k];
            r2_ee += d * d;
        }
        double r_ee = std::sqrt(r2_ee);

        return jastrowEH(r_e1h, r2_e1h) * jastrowEH(r_e2h, r2_e2h) * jastrowEE(r_ee,  r2_ee);
    }
};