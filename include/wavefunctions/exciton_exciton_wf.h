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
    int Nee, Nhh, Neh;
    int eeOffset, hhOffset, ehOffset;

public:
    ExcitonExcitonWF(const std::vector<double>& params, int nParticles, int dim,
                     double me, double mh, double d, double R, bool interacting = true,
                     int Nee = 0, int Nhh = 0, int Neh = 0)
        : WaveFunction(params, nParticles, dim), me(me), mh(mh), d(d), R(R),
          interacting(interacting), Nee(Nee), Nhh(Nhh), Neh(Neh) {
        mu = (me * mh) / (me + mh);

        // Layout after c1-c9 (indices 0-8):
        //   if Nee>0: [Lee, u1..uNee]
        //   if Nhh>0: [Lhh, v1..vNhh]
        //   if Neh>0: [Leh, w1..wNeh]
        int eeCount = (Nee > 0) ? 1 + Nee : 0;
        int hhCount = (Nhh > 0) ? 1 + Nhh : 0;
        int ehCount = (Neh > 0) ? 1 + Neh : 0;
        eeOffset = 9;
        hhOffset = eeOffset + eeCount;
        ehOffset = hhOffset + hhCount;
        this->params.resize(ehOffset + ehCount, 0.0);
    }

    WaveFunction* clone() const override {
        return new ExcitonExcitonWF(*this);
    }

    void setParameters(const std::vector<double>& newParams) override {
        if (interacting) {
            // c1, c3, c5: any real value
            params[0] = newParams[0];
            params[2] = newParams[2];
            params[4] = newParams[4];

            // c2, c4, c7, c9: strictly positive
            params[1] = std::exp(newParams[1]);
            params[3] = std::exp(newParams[3]);
            params[6] = std::exp(newParams[6]);
            params[8] = std::exp(newParams[8]);

            // c6, c8: strictly negative
            params[5] = -std::exp(newParams[5]);
            params[7] = -std::exp(newParams[7]);

            // Polynomial cutoff Jastrow: L (positive via exp) + coefficients
            int idx = 9;
            if (Nee > 0) {
                params[eeOffset] = std::exp(newParams[idx++]);
                for (int i = 0; i < Nee; i++)
                    params[eeOffset + 1 + i] = newParams[idx++];
            }
            if (Nhh > 0) {
                params[hhOffset] = std::exp(newParams[idx++]);
                for (int i = 0; i < Nhh; i++)
                    params[hhOffset + 1 + i] = newParams[idx++];
            }
            if (Neh > 0) {
                params[ehOffset] = std::exp(newParams[idx++]);
                for (int i = 0; i < Neh; i++)
                    params[ehOffset + 1 + i] = newParams[idx++];
            }
        } else {
            // Non-interacting: c5, c6, c7, c8, c9
            params[4] = newParams[0];              // c5
            params[5] = -std::exp(newParams[1]);   // c6 (negative)
            params[6] = std::exp(newParams[2]);    // c7 (positive)
            params[7] = -std::exp(newParams[3]);   // c8 (negative)
            params[8] = std::exp(newParams[4]);    // c9 (positive)

            // Only eh cutoff in non-interacting
            if (Neh > 0) {
                int idx = 5;
                params[ehOffset] = std::exp(newParams[idx++]);
                for (int i = 0; i < Neh; i++)
                    params[ehOffset + 1 + i] = newParams[idx++];
            }
        }
    }

    std::vector<double> getParameters() const override {
        if (interacting) {
            std::vector<double> result = {
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
            if (Nee > 0) {
                result.push_back(std::log(params[eeOffset]));
                for (int i = 0; i < Nee; i++)
                    result.push_back(params[eeOffset + 1 + i]);
            }
            if (Nhh > 0) {
                result.push_back(std::log(params[hhOffset]));
                for (int i = 0; i < Nhh; i++)
                    result.push_back(params[hhOffset + 1 + i]);
            }
            if (Neh > 0) {
                result.push_back(std::log(params[ehOffset]));
                for (int i = 0; i < Neh; i++)
                    result.push_back(params[ehOffset + 1 + i]);
            }
            return result;
        } else {
            std::vector<double> result = {
                params[4],              // c5
                std::log(-params[5]),   // log(-c6)
                std::log(params[6]),    // log(c7)
                std::log(-params[7]),   // log(-c8)
                std::log(params[8])     // log(c9)
            };
            if (Neh > 0) {
                result.push_back(std::log(params[ehOffset]));
                for (int i = 0; i < Neh; i++)
                    result.push_back(params[ehOffset + 1 + i]);
            }
            return result;
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

    // Symmetrized Electron-Hole Jastrow
    double jastrowEH(double r1a, double r2_1a,
                               double r2b, double r2_2b,
                               double r2a, double r2_2a,
                               double r1b, double r2_1b) const {

        double c5 = params[4];
        double c6 = params[5];
        double c7 = params[6];
        double c8 = params[7];
        double c9 = params[8];

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

    // Polynomial cutoff contribution: (r - L)^3 * Theta(L - r) * sum_{l=1}^{N} coeff_l * r^l
    double cutoffPoly(double r, double L, const double* coeffs, int N) const {
        if (N == 0 || r >= L) return 0.0;
        double cutoff = r - L;
        cutoff = cutoff * cutoff * cutoff;
        double poly = 0.0;
        double r_pow = r;
        for (int l = 0; l < N; l++) {
            poly += coeffs[l] * r_pow;
            r_pow *= r;
        }
        return cutoff * poly;
    }

    // Full polynomial cutoff Jastrow J
    double computeJ(double r_ee, double r_hh,
                    double r_e1h1, double r_e2h2,
                    double r_e2h1, double r_e1h2) const {
        double J = 0.0;

        if (Nee > 0) {
            J += cutoffPoly(r_ee, params[eeOffset], &params[eeOffset + 1], Nee);
        }
        if (Nhh > 0) {
            J += cutoffPoly(r_hh, params[hhOffset], &params[hhOffset + 1], Nhh);
        }
        if (Neh > 0) {
            double L = params[ehOffset];
            const double* w = &params[ehOffset + 1];
            J += cutoffPoly(r_e1h1, L, w, Neh);
            J += cutoffPoly(r_e2h2, L, w, Neh);
            J += cutoffPoly(r_e2h1, L, w, Neh);
            J += cutoffPoly(r_e1h2, L, w, Neh);
        }

        return J;
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
            double psi_eh = jastrowEH(r_e1h1, r2_e1h1 + d_sq, r_e2h2, r2_e2h2 + d_sq,
                             0.0, 0.0, 0.0, 0.0);
            double J_val = computeJ(0.0, 0.0, r_e1h1, r_e2h2, 0.0, 0.0);
            return std::exp(J_val) * psi_eh;
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

        double J_val = computeJ(r_ee, r_hh, r_e1h1, r_e2h2, r_e2h1, r_e1h2);

        return std::exp(J_val) * psi_ee * psi_hh * psi_eh;
    }
};
