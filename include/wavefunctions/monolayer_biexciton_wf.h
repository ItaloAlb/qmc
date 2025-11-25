#include "wavefunction.h"

class MonolayerBiexcitonWF : public WaveFunction {
public:
    MonolayerBiexcitonWF(const std::vector<double>& params, int nParticles, int dim) 
        : WaveFunction(params, nParticles, dim) {}

    WaveFunction* clone() const override {
        return new MonolayerBiexcitonWF(*this);
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
        // [xe1, ye1, xe2, ye2, xh1, yh1, xh2, yh2]
        int idx_e1 = 0;
        int idx_e2 = 1 * dim;
        int idx_h1  = 2 * dim;
        int idx_h2 = 3 * dim;

        double r2_e1h1 = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e1 + k] - position[idx_h1 + k];
            r2_e1h1 += d * d;
        }
        double r_e1h1 = std::sqrt(r2_e1h1);

        double r2_e1h2 = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e1 + k] - position[idx_h2 + k];
            r2_e1h2 += d * d;
        }
        double r_e1h2 = std::sqrt(r2_e1h2);

        double r2_e2h1 = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e2 + k] - position[idx_h1 + k];
            r2_e2h1 += d * d;
        }
        double r_e2h1 = std::sqrt(r2_e2h1);

        double r2_e2h2 = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e2 + k] - position[idx_h2 + k];
            r2_e2h2 += d * d;
        }
        double r_e2h2 = std::sqrt(r2_e2h2);

        double r2_ee = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e1 + k] - position[idx_e2 + k];
            r2_ee += d * d;
        }
        double r_ee = std::sqrt(r2_ee);

        double r2_hh = 0.0;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_h1 + k] - position[idx_h2 + k];
            r2_hh += d * d;
        }
        double r_hh = std::sqrt(r2_hh);

        return jastrowEH(r_e1h1, r2_e1h1) * 
                jastrowEH(r_e1h2, r2_e1h2) * 
                jastrowEH(r_e2h1, r2_e2h1) * 
                jastrowEH(r_e2h2, r2_e2h2) * 
                jastrowEE(r_ee,  r2_ee) * 
                jastrowEE(r_hh,  r2_hh);
    }
};