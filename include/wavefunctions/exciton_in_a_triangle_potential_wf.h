#include "wavefunction.h"
#include <cmath>

class ExcitonInATrianglePotentialWF : public WaveFunction {
private:
    const double a; // Lado do triângulo equilátero (em unidades de a0)
    const double thickness;
    const double thicknessSquared;
    const bool interacting;

public:
    ExcitonInATrianglePotentialWF(const std::vector<double>& params, int nParticles, int dim,
                                  double a_, double thickness_, bool interacting_ = true)
        : WaveFunction(params, nParticles, dim),
          a(a_ / a0),
          thickness(thickness_),
          thicknessSquared(thickness_ * thickness_),
          interacting(interacting_) {}

    WaveFunction* clone() const override {
        return new ExcitonInATrianglePotentialWF(*this);
    }

    // Layout dos parâmetros internos (sempre 5 posições):
    //   [0] c1       (jastrow, fixo por condição de cúspide)
    //   [1] c2       (jastrow, variacional)
    //   [2] c3       (jastrow, variacional)
    //   [3] lambda_e (peso variacional do potencial triangular para o elétron)
    //   [4] lambda_h (peso variacional do potencial triangular para o buraco)
    //
    // Layout para o otimizador:
    //   interacting = true  -> 4 parâmetros: [log c2, log c3, lambda_e, lambda_h]
    //   interacting = false -> 2 parâmetros: [lambda_e, lambda_h]

    void setParameters(const std::vector<double>& newParams) override {
        if (interacting) {
            params[1] = std::exp(newParams[0]);
            params[2] = std::exp(newParams[1]);
            params[3] = newParams[2];
            params[4] = newParams[3];
        } else {
            params[3] = newParams[0];
            params[4] = newParams[1];
        }
    }

    std::vector<double> getParameters() const override {
        if (interacting) {
            return {
                std::log(params[1]),
                std::log(params[2]),
                params[3],
                params[4],
            };
        } else {
            return {
                params[3],
                params[4],
            };
        }
    }

    double jastrowEH(double r, double r2) const {
        double c1 = params[0];
        double c2 = params[1];
        double c3 = params[2];
        double exp_term = std::exp(-c2 * r2);
        double term1 = c1 * r2 * std::log(r) * exp_term;
        double term2 = c3 * r * (1.0 - exp_term);
        return std::exp(term1 - term2);
    }

    double variationalPotential(const double* position) const {
        double lambda_e = params[3];
        double lambda_h = params[4];

        // Smooth phi for the equilateral-triangle tiling: sum of three
        // sinusoids along the K-point reciprocal vectors. Normalized so
        // the function is +1 at up-triangle centroids and -1 at
        // down-triangle centroids. Origin at a lattice vertex (matching
        // the Hamiltonian) so the tiling aligns with PBC edges.
        double invSqrt3 = 1.0 / std::sqrt(3.0);
        double k = 4.0 * M_PI * invSqrt3 / a; // |G| = 4π / (a√3)
        double norm = 2.0 / (3.0 * std::sqrt(3.0));

        // G1 = k*(0, 1)
        // G2 = k*(-√3/2, -1/2)
        // G3 = k*(+√3/2, -1/2)
        auto triangular = [k, norm](double x, double y) {
            double halfSqrt3 = 0.5 * std::sqrt(3.0);
            double g1 = k * y;
            double g2 = k * (-halfSqrt3 * x - 0.5 * y);
            double g3 = k * ( halfSqrt3 * x - 0.5 * y);
            return norm * (std::sin(g1) + std::sin(g2) + std::sin(g3));
        };

        double Ve = lambda_e * triangular(position[0], position[1]);
        double Vh = lambda_h * triangular(position[2], position[3]);

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

        return jastrow * (1.0 - variationalPotential(position));
    }
};
