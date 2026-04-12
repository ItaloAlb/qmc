#include "hamiltonian.h"
#include <cmath>

class TriangleHamiltonian : public Hamiltonian {
private:
    double V0;
    double a;

    double eps1;
    double eps2;
    double thicknessSquared;
    double invrho0;

    bool interacting;

public:
    TriangleHamiltonian(int nParticles, int dim,
                        const std::vector<double>& masses,
                        const std::vector<double>& charges,
                        double V0_, double side_,
                        double eps1_, double eps2_, double rho0_, double thickness_, bool interacting_ = true)
        : Hamiltonian(nParticles, dim, masses, charges),
          V0(V0_ / HARTREE), a(side_ / a0), eps1(eps1_), eps2(eps2_), interacting(interacting_) {

        this->thicknessSquared = thickness_ * thickness_;
        this->invrho0 = 1.0 / rho0_;
    }

    double getPotential(const double* position) const override {
        double xe = position[0];
        double ye = position[1];
        double xh = position[2];
        double yh = position[3];

        // Equilateral-triangle tiling with primitive vectors
        //   a1 = (a, 0),  a2 = (a/2, a*sqrt(3)/2).
        // The origin is placed at the centroid of an "up" triangle so that
        // (0,0) sits at a +V0 maximum — mirrors the square version where the
        // origin is the center of a cell.
        auto Phi = [this](double x, double y) {
            double invSqrt3 = 1.0 / std::sqrt(3.0);

            // Centroid of the reference "up" triangle {0, a1, a2} is
            // (a/2, a*sqrt(3)/6). Shift the point so that origin maps
            // to that centroid.
            double xs = x + 0.5 * this->a;
            double ys = y + this->a * invSqrt3 * 0.5;  // a/(2*sqrt(3)) = a*sqrt(3)/6

            double u = xs / this->a - ys * invSqrt3 / this->a;
            double v = 2.0 * ys * invSqrt3 / this->a;

            double u_frac = u - std::floor(u);
            double v_frac = v - std::floor(v);

            if (u_frac + v_frac < 1.0) {
                return this->V0;   // up triangle
            } else {
                return -this->V0;  // down triangle
            }
        };

        double Phi_electron = Phi(xe, ye);
        double Phi_hole     = Phi(xh, yh);

        double externalPotential = (-1.0 * Phi_electron) + (1.0 * Phi_hole);

        if (interacting) { return externalPotential + getMonolayerRytovaKeldysh(position); }

        return externalPotential;
    }

    double getHeterobilayerRytovaKeldysh(const double* position) const {
        double r2 = 0.0;

        for (int k = 0; k < 2; k++) {
            double dist = position[k] - position[k + 2];
            r2 += dist * dist;
        }

        double r = std::sqrt(r2 + thicknessSquared);
        double r_times_inv_rho0 = r * invrho0;

        return -M_PI / (eps1 + eps2) * invrho0 *
               (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
    }

    double getMonolayerRytovaKeldysh(const double* position) const {
        double r2 = 0.0;

        for (int k = 0; k < 2; k++) {
            double dist = position[k] - position[k + 2];
            r2 += dist * dist;
        }

        double r = std::sqrt(r2);
        double r_times_inv_rho0 = r * invrho0;

        return -M_PI / (eps1 + eps2) * invrho0 *
               (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
    }
};
