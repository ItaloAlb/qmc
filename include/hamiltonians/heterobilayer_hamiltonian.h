#include "hamiltonian.h"
#include "utils.h"

using namespace Constants;
using namespace Utils;

class HeterobilayerHamiltonian : public Hamiltonian {
    private:
        double eField;
        double eps1, eps2;
        double rho0;
        double invrho0;
        double d, dSquared;
    public:
        HeterobilayerHamiltonian(int nParticles, int dim,
                        const std::vector<double>& masses,
                        const std::vector<double>& charges,
                        double dst_, double rho0_, double eps1_, double eps2_)
            : Hamiltonian(nParticles, dim, masses, charges),
            rho0(rho0_), 
            d(dst_ / Constants::a0),
            eps1(eps1_),
            eps2(eps2_) {
                invrho0 = 1.0 / rho0;
                dSquared = d * d;
            }

        double getHeterobilayerRytovaKeldysh(const double* position) const {
            double r2 = dSquared;

            for(int k = 0; k < dim; k++) {
                double dist = position[k] - position[k + dim];
                r2 += dist * dist;
            }
            double r = std::sqrt(r2);
            double r_times_inv_rho0 = r * invrho0;
            
            return - PI / (eps1 + eps2) * invrho0 * (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
        }

        double getPotential(const double* position) const override {
            return getHeterobilayerRytovaKeldysh(position);
        }
};