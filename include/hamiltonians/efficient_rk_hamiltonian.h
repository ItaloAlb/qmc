#include "hamiltonian.h"

class EfficientRKHamiltonian : public Hamiltonian {
    private:
        const double EULER_MASCHERONI = 0.5772156649;
        const double LN2 = 0.69314718056;

        double rho0;
        double invrho0;
    public:
        EfficientRKHamiltonian(int nParticles, int dim,
                        const std::vector<double>& masses,
                        const std::vector<double>& charges,
                        double rho0_)
            : Hamiltonian(nParticles, dim, masses, charges),
            rho0(rho0_) {
                invrho0 = 1.0 / rho0;
            }

        double getPotential(const double* position) const override {
            double r2 = 0.0;

            for(int k = 0; k < dim; k++) {
                double dist = position[k] - position[k + dim];
                r2 += dist * dist;
            }
            double r = std::sqrt(r2);

            return - invrho0 * (std::log(r / (r + rho0)) + (EULER_MASCHERONI - LN2) * std::exp(- r * invrho0));
        }
};