#include "wavefunction.h"
#include "utils.h"
#include "complex"

using namespace std::complex_literals;

class TwistedBilayerExcitonWF : public WaveFunction {
private:
    TwistedBilayerSystem moire;
    const double thickness;
    const double thicknessSquared;
    const bool interacting;

public:
    TwistedBilayerExcitonWF(const std::vector<double>& params, int nParticles, int dim,
                            TwistedBilayerSystem moire_, double thickness_, bool interacting_ = true)
        : WaveFunction(params, nParticles, dim),
          moire(moire_),
          thickness(thickness_),
          thicknessSquared(thickness * thickness),
          interacting(interacting_) {}

    WaveFunction* clone() const override {
        return new TwistedBilayerExcitonWF(*this);
    }

    // internal params layout (always 7 entries):
    //   [0] c1   (jastrow, fixed by Kato cusp)
    //   [1] c2   (jastrow)
    //   [2] c3   (jastrow)
    //   [3] Ve1  (variational electron potential)
    //   [4] Ve2  (variational electron potential)
    //   [5] Vh1  (variational hole potential)
    //   [6] Vh2  (variational hole potential)
    //
    // optimizer-facing layout (what set/getParameters expose):
    //   interacting = true  -> 6 params: [log c2, log c3, Ve1, Ve2, Vh1, Vh2]
    //   interacting = false -> 4 params: [Ve1, Ve2, Vh1, Vh2]   (c1,c2,c3 frozen, jastrow disabled)
    void setParameters(const std::vector<double>& newParams) override {
        if (interacting) {
            params[1] = std::exp(newParams[0]);
            params[2] = std::exp(newParams[1]);
            params[3] = newParams[2];
            params[4] = newParams[3];
            params[5] = newParams[4];
            params[6] = newParams[5];
        } else {
            params[3] = newParams[0];
            params[4] = newParams[1];
            params[5] = newParams[2];
            params[6] = newParams[3];
        }
    }

    std::vector<double> getParameters() const override {
        if (interacting) {
            return {
                std::log(params[1]),
                std::log(params[2]),
                params[3],
                params[4],
                params[5],
                params[6],
            };
        } else {
            return {
                params[3],
                params[4],
                params[5],
                params[6],
            };
        }
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

    double variationalPotential(const double* position) const {
        double Ve1_var = params[3];
        double Ve2_var = params[4];
        double Vh1_var = params[5];
        double Vh2_var = params[6];

        double Ve = moire.getCarrierPotential(position[0], position[1], Ve1_var, Ve2_var);
        double Vh = moire.getCarrierPotential(position[2], position[3], Vh1_var, Vh2_var);

        return Ve + Vh;
    }

    double trialWaveFunction(const double* position) const override {
        double jastrow = 1.0;
        if (interacting) {
            int idx_e = 0;
            int idx_h = 1 * dim;
            double r2 = thicknessSquared;
            for (int k = 0; k < dim; k++) {
                double d = position[idx_e + k] - position[idx_h + k];
                r2 += d * d;
            }
            double r = std::sqrt(r2);
            jastrow = jastrowEH(r, r2);
        }
        return jastrow * (1 - variationalPotential(position));
    }
};