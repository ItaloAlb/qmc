#include "wavefunction.h"
#include <cmath>

class ExcitonInASquarePotentialWF : public WaveFunction {
private:
    const double a; // Parâmetro de rede do tabuleiro (tamanho da célula)
    const double thickness;
    const double thicknessSquared;
    const bool interacting;

public:
    // Atualizado para receber 'a_' ao invés do sistema moiré
    ExcitonInASquarePotentialWF(const std::vector<double>& params, int nParticles, int dim,
                          double a_, double thickness_, bool interacting_ = true)
        : WaveFunction(params, nParticles, dim),
          a(a_),
          thickness(thickness_),
          thicknessSquared(thickness_ * thickness_),
          interacting(interacting_) {}

    WaveFunction* clone() const override {
        return new ExcitonInASquarePotentialWF(*this);
    }

    // Layout dos parâmetros internos (sempre 5 posições):
    //   [0] c1       (jastrow, fixo por condição de cúspide)
    //   [1] c2       (jastrow, variacional)
    //   [2] c3       (jastrow, variacional)
    //   [3] lambda_e (peso variacional do potencial xadrez para o elétron)
    //   [4] lambda_h (peso variacional do potencial xadrez para o buraco)
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

        // Smooth phi: cos(pi*x/a)*cos(pi*y/a) reproduces the
        // +1/-1 pattern of the step-function but is C-infinity,
        // so the numerical Laplacian stays finite everywhere.
        double ka = M_PI / a;
        auto checkerboard = [ka](double x, double y) {
            return std::cos(ka * x) * std::cos(ka * y);
        };

        double Ve = lambda_e * checkerboard(position[0], position[1]);
        double Vh = lambda_h * checkerboard(position[2], position[3]);

        return Ve + Vh;
    }

    double trialWaveFunction(const double* position) const override {
        double jastrow = 1.0;
        if (interacting) {
            int idx_e = 0;
            int idx_h = 1 * dim; // Assumindo dim=2, o buraco começa no índice 2
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