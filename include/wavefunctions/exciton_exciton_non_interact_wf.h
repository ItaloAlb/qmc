#include "wavefunction.h"
#include <cmath>
#include <vector>

class ExcitonExcitonNonInteractWF : public WaveFunction {
private:
    double me;
    double mh;
    double mu;
    double d;
    double R;

public:
    // Updated constructor to accept me and mh for the coordinate transformation
    ExcitonExcitonNonInteractWF(const std::vector<double>& params, int nParticles, int dim, double me, double mh, double d) 
        : WaveFunction(params, nParticles, dim), me(me), mh(mh), d(d) {
        mu = (me * mh) / (me + mh);
    }

    WaveFunction* clone() const override {
        return new ExcitonExcitonNonInteractWF(*this);
    }

void setParameters(const std::vector<double>& newParams) override {
        if (newParams.size() == 9) {
            params[0] = newParams[0]; 

            params[2] = std::exp(newParams[2]); 
            params[4] = std::exp(newParams[4]); 

            params[1] = -std::exp(newParams[1]); 
            params[3] = -std::exp(newParams[3]); 
        }
    }

    std::vector<double> getParameters() const override {
        return { 
            params[0],              // c5
            std::log(-params[1]),   // log(-c6) para reverter o sinal
            std::log(params[2]),    // log(c7)
            std::log(-params[3]),   // log(-c8) para reverter o sinal
            std::log(params[4])     // log(c9)
        };
    }

    double jastrowEE(double r) const {
        double c1 = params[0];
        double c2 = params[1];
        return std::exp((c1 * r) / (1.0 + c2 * r));
    }

    double jastrowHH(double r) const {
        double c3 = params[2];
        double c4 = params[3];
        return std::exp((c3 * r) / (1.0 + c4 * r));
    }

    double jastrowEH(double r1a, double r2_1a, 
                               double r2b, double r2_2b) const {
        
        double c5 = params[0];
        double c6 = params[1];
        double c7 = params[2];
        double c8 = params[3];
        double c9 = params[4];

        auto calc_term = [c5](double r, double r2, double cx, double cy) {
            return (c5 * r + cx * r2) / (1.0 + cy * r);
        };

        double exp1_args = calc_term(r1a, r2_1a, c6, c7) + 
                           calc_term(r2b, r2_2b, c6, c7);

        double exp2_args = calc_term(r1a, r2_1a, c8, c9) + 
                           calc_term(r2b, r2_2b, c8, c9);

        return std::exp(exp1_args) + std::exp(exp2_args);
    }


    double trialWaveFunction(const double* position) const override {
        std::vector<double> r(dim, 0.0);
        r[0] = R;

        double r2_e1h1 = 0.0;
        double r2_e2h2 = 0.0;
        double r2_ee   = 0.0;
        double r2_hh   = 0.0;
        double r2_e2h1 = 0.0;
        double r2_e1h2 = 0.0;
        double d_sq = d * d;

        for(int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];

            r2_e1h1 += r1_k * r1_k;
            r2_e2h2 += r2_k * r2_k;
        }

        double r_e1h1 = std::sqrt(r2_e1h1 + d_sq);
        double r_e2h2 = std::sqrt(r2_e2h2 + d_sq);
        double r_ee   = std::sqrt(r2_ee);
        double r_hh   = std::sqrt(r2_hh);
        double r_e2h1 = std::sqrt(r2_e2h1 + d_sq);
        double r_e1h2 = std::sqrt(r2_e1h2 + d_sq);

        double psi_eh = jastrowEH(r_e1h1, r2_e1h1 + d_sq, r_e2h2, r2_e2h2 + d_sq);

        return psi_eh;
    }
};