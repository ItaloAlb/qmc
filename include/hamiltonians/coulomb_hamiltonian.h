#include "hamiltonian.h"

class CoulombHamiltonian : public Hamiltonian {
public:
    CoulombHamiltonian(int nParticles, int dim, std::vector<double> masses, std::vector<double> charges)
        : Hamiltonian(nParticles, dim, masses, charges) {}

    double getPotential(const double* position) const override {
        double potentialEnergy = 0.0;

        for (int i = 0; i < nParticles; i++) {
            for (int j = i + 1; j < nParticles; j++) {
                
                double r2 = 0.0;
                int idx_i = i * dim;
                int idx_j = j * dim;

                for (int d = 0; d < dim; d++) {
                    double delta = position[idx_i + d] - position[idx_j + d];
                    r2 += delta * delta;
                }

                double r = std::sqrt(r2);
                if (r < Constants::MIN_DISTANCE) r = Constants::MIN_DISTANCE;

                potentialEnergy += (charges[i] * charges[j]) / r;
            }
        }
        return potentialEnergy;
    }
};