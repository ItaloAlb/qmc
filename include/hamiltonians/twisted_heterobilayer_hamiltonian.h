#include "hamiltonian.h"
#include "utils.h"
#include <complex>

using namespace Constants;
using namespace Utils;
using namespace std::complex_literals;

class TwistedHeterobilayerHamiltonian : public Hamiltonian {
private:
    const double eps1, eps2;
    const double rho0;
    const double invrho0;
    const double thickness;
    const double thicknessSquared;
    const bool interacting;
    TwistedBilayerSystem moire;

public:
    TwistedHeterobilayerHamiltonian(int nParticles, int dim,
                                    const std::vector<double>& masses,
                                    const std::vector<double>& charges,
                                    TwistedBilayerSystem moire_,
                                    double rho0_, double thickness_, double eps1_, double eps2_,
                                    bool interacting_ = true)
        : Hamiltonian(nParticles, dim, masses, charges),
          eps1(eps1_),
          eps2(eps2_),
          rho0(rho0_),
          moire(moire_),
          thickness(thickness_ / a0),
          thicknessSquared(thickness * thickness),
          interacting(interacting_),
          invrho0(1.0 / rho0) {}

    double getHeterobilayerRytovaKeldysh(const double* position) const {
        double r2 = 0.0;
        for (int k = 0; k < dim; k++) {
            double dist = position[k] - position[k + dim];
            r2 += dist * dist;
        }
        double r = std::sqrt(r2 + thicknessSquared);
        double r_times_inv_rho0 = r * invrho0;
        return -PI / (eps1 + eps2) * invrho0 *
               (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
    }

    double getPotential(const double* position) const override {
        double V = moire.getExcitonMoirePotential(position);
        if (interacting) V += getHeterobilayerRytovaKeldysh(position);
        return V;
    }
};