#include "hamiltonian.h"
#include <cmath> // Necessário para std::floor, std::abs, e M_PI

class SquareHamiltonian : public Hamiltonian {
private:
    double V0;
    double a;
    
    double eps1;
    double eps2;
    double thicknessSquared;
    double invrho0;

    bool interacting;

public:
    SquareHamiltonian(int nParticles, int dim,
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

        auto Phi = [this](double x, double y) {
            // Cells centered on the origin: cell (0,0) spans [-a/2, a/2) × [-a/2, a/2).
            int cell_x = static_cast<int>(std::floor(x / this->a + 0.5));
            int cell_y = static_cast<int>(std::floor(y / this->a + 0.5));

            if (std::abs(cell_x + cell_y) % 2 == 0) {
                return this->V0;
            } else {
                return -this->V0;
            }
        };

        double Phi_electron = Phi(xe, ye);
        double Phi_hole     = Phi(xh, yh);

        double externalPotential = (-1.0 * Phi_electron) + (1.0 * Phi_hole);

        if(interacting) { return externalPotential + getHeterobilayerRytovaKeldysh(position); }

        return externalPotential;
    }

    double getHeterobilayerRytovaKeldysh(const double* position) const {
        double r2 = 0.0;
        
        for (int k = 0; k < 2; k++) { // Alterado para 2 direto, pois xe,ye,xh,yh assumem 2D
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
        
        for (int k = 0; k < 2; k++) { // Alterado para 2 direto, pois xe,ye,xh,yh assumem 2D
            double dist = position[k] - position[k + 2]; 
            r2 += dist * dist;
        }
        
        double r = std::sqrt(r2);
        double r_times_inv_rho0 = r * invrho0;
        
        return -M_PI / (eps1 + eps2) * invrho0 *
               (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
    }
};