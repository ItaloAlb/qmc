#include "wavefunction.h"
#include <cmath>
#include <vector>

class ExcitonExcitonWF : public WaveFunction {
private:
    double me;
    double mh;
    double mu;
    double d;

public:
    // Updated constructor to accept me and mh for the coordinate transformation
    ExcitonExcitonWF(const std::vector<double>& params, int nParticles, int dim, double me, double mh, double d) 
        : WaveFunction(params, nParticles, dim), me(me), mh(mh), d(d) {
        mu = (me * mh) / (me + mh);
    }

    WaveFunction* clone() const override {
        return new ExcitonExcitonWF(*this);
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
        std::vector<double> R(dim, 0.0);
        if (dim > 0) {
            R[0] = 1.0; 
        }

        double r2_e1h1 = 0.0;
        double r2_e2h2 = 0.0;
        double r2_ee   = 0.0;
        double r2_hh   = 0.0;
        double r2_e2h1 = 0.0;
        double r2_e1h2 = 0.0;
        double d_sq = d * d;

        // Calculate all squared distances in a single loop over dimensions
        for(int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];
            double R_k  = R[k];

            // Intra-exciton distances (e1-h1 and e2-h2)
            r2_e1h1 += r1_k * r1_k;
            r2_e2h2 += r2_k * r2_k;

            // Inter-exciton distances (e1-e2 and h1-h2)
            double d_ee = R_k + (mu / me) * (r1_k - r2_k);
            r2_ee += d_ee * d_ee;

            double d_hh = R_k - (mu / mh) * (r1_k - r2_k);
            r2_hh += d_hh * d_hh;

            // Cross-exciton distances (e2-h1 and e1-h2)
            double d_e2h1 = - R_k + mu * (r2_k / me + r1_k / mh);
            r2_e2h1 += d_e2h1 * d_e2h1;

            double d_e1h2 = R_k + mu * (r1_k / me + r2_k / mh);
            r2_e1h2 += d_e1h2 * d_e1h2;
        }

        double r_e1h1 = std::sqrt(r2_e1h1);
        double r_e2h2 = std::sqrt(r2_e2h2);
        double r_ee   = std::sqrt(r2_ee + d_sq );
        double r_hh   = std::sqrt(r2_hh + d_sq);
        double r_e2h1 = std::sqrt(r2_e2h1 + d_sq);
        double r_e1h2 = std::sqrt(r2_e1h2 + d_sq);

        return jastrowEH(r_e1h1, r2_e1h1) * jastrowEH(r_e1h2, r2_e1h2) * jastrowEH(r_e2h1, r2_e2h1) * jastrowEH(r_e2h2, r2_e2h2) * jastrowEE(r_ee,  r2_ee) * jastrowEE(r_hh,  r2_hh);
    }
};