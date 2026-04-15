#include "wavefunction.h"
#include <cmath>
#include <vector>

class ExcitonExcitonWF : public WaveFunction {
private:
    double me;
    double mh;
    double mu;
    double d;
    double R;
    bool interacting;

public:
    ExcitonExcitonWF(const std::vector<double>& params, int nParticles, int dim, double me, double mh, double d, double R, bool interacting = true)
        : WaveFunction(params, nParticles, dim), me(me), mh(mh), d(d), R(R), interacting(interacting) {
        mu = (me * mh) / (me + mh);
    }

    WaveFunction* clone() const override {
        return new ExcitonExcitonWF(*this);
    }

void setParameters(const std::vector<double>& newParams) override {
        if (interacting) {
            // c1, c3 e c5 podem assumir qualquer valor real
            params[0] = newParams[0];
            params[2] = newParams[2];
            params[4] = newParams[4];

            // c2, c4, c7 e c9 devem ser estritamente positivos (> 0)
            params[1] = std::exp(newParams[1]);
            params[3] = std::exp(newParams[3]);
            params[6] = std::exp(newParams[6]);
            params[8] = std::exp(newParams[8]);

            // c6 e c8 devem ser estritamente negativos (< 0)
            params[5] = -std::exp(newParams[5]);
            params[7] = -std::exp(newParams[7]);
        } else {
            // Non-interacting: only c5, c6, c7, c8, c9
            params[4] = newParams[0];              // c5
            params[5] = -std::exp(newParams[1]);   // c6 (negative)
            params[6] = std::exp(newParams[2]);    // c7 (positive)
            params[7] = -std::exp(newParams[3]);   // c8 (negative)
            params[8] = std::exp(newParams[4]);    // c9 (positive)
        }
    }

    std::vector<double> getParameters() const override {
        if (interacting) {
            return {
                params[0],              // c1
                std::log(params[1]),    // log(c2)
                params[2],              // c3
                std::log(params[3]),    // log(c4)
                params[4],              // c5
                std::log(-params[5]),   // log(-c6)
                std::log(params[6]),    // log(c7)
                std::log(-params[7]),   // log(-c8)
                std::log(params[8])     // log(c9)
            };
        } else {
            return {
                params[4],              // c5
                std::log(-params[5]),   // log(-c6)
                std::log(params[6]),    // log(c7)
                std::log(-params[7]),   // log(-c8)
                std::log(params[8])     // log(c9)
            };
        }
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

    // New Symmetrized Electron-Hole Jastrow
    // Takes r1a (e1-h1), r2b (e2-h2), r2a (e2-h1), r1b (e1-h2) and their squares
    double jastrowEH(double r1a, double r2_1a, 
                               double r2b, double r2_2b, 
                               double r2a, double r2_2a, 
                               double r1b, double r2_1b) const {
        
        double c5 = params[4];
        double c6 = params[5];
        double c7 = params[6];
        double c8 = params[7];
        double c9 = params[8];

        // A quick helper lambda to calculate the individual fractional terms cleanly
        auto calc_term = [c5](double r, double r2, double cx, double cy) {
            return (c5 * r + cx * r2) / (1.0 + cy * r);
        };

        // First exciton pairing configuration
        double exp1_args = calc_term(r1a, r2_1a, c6, c7) + 
                           calc_term(r2b, r2_2b, c6, c7) + 
                           calc_term(r2a, r2_2a, c8, c9) + 
                           calc_term(r1b, r2_1b, c8, c9);

        // Second exciton pairing configuration (swapped holes)
        double exp2_args = calc_term(r1a, r2_1a, c8, c9) + 
                           calc_term(r2b, r2_2b, c8, c9) + 
                           calc_term(r2a, r2_2a, c6, c7) + 
                           calc_term(r1b, r2_1b, c6, c7);

        return std::exp(exp1_args) + std::exp(exp2_args);
    }


    double trialWaveFunction(const double* position) const override {
        double d_sq = d * d;
        double r2_e1h1 = 0.0;
        double r2_e2h2 = 0.0;

        for(int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];
            r2_e1h1 += r1_k * r1_k;
            r2_e2h2 += r2_k * r2_k;
        }

        double r_e1h1 = std::sqrt(r2_e1h1 + d_sq);
        double r_e2h2 = std::sqrt(r2_e2h2 + d_sq);

        if (!interacting) {
            return jastrowEH(r_e1h1, r2_e1h1 + d_sq, r_e2h2, r2_e2h2 + d_sq,
                             0.0, 0.0, 0.0, 0.0);
        }

        std::vector<double> r(dim, 0.0);
        r[0] = R;
        double r2_ee = 0.0, r2_hh = 0.0, r2_e2h1 = 0.0, r2_e1h2 = 0.0;

        for(int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];
            double R_k  = r[k];

            double d_ee = R_k + (mu / me) * (r1_k - r2_k);
            r2_ee += d_ee * d_ee;

            double d_hh = R_k - (mu / mh) * (r1_k - r2_k);
            r2_hh += d_hh * d_hh;

            double d_e2h1 = - R_k + mu * (r2_k / me + r1_k / mh);
            r2_e2h1 += d_e2h1 * d_e2h1;

            double d_e1h2 = R_k + mu * (r1_k / me + r2_k / mh);
            r2_e1h2 += d_e1h2 * d_e1h2;
        }

        double r_ee   = std::sqrt(r2_ee);
        double r_hh   = std::sqrt(r2_hh);
        double r_e2h1 = std::sqrt(r2_e2h1 + d_sq);
        double r_e1h2 = std::sqrt(r2_e1h2 + d_sq);

        double psi_ee = jastrowEE(r_ee);
        double psi_hh = jastrowHH(r_hh);
        double psi_eh = jastrowEH(r_e1h1, r2_e1h1, r_e2h2, r2_e2h2,
                                   r_e2h1, r2_e2h1 + d_sq, r_e1h2, r2_e1h2 + d_sq);

        return psi_ee * psi_hh * psi_eh;
    }
};