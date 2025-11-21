#include "hamiltonian.h"

class CoulombHamiltonian : public Hamiltonian {
public:
    CoulombHamiltonian(int nParticles, int dim,
                       const std::vector<double>& masses,
                       const std::vector<double>& charges)
        : Hamiltonian(nParticles, dim, masses, charges) {}

    double getPotential(const double* position) const override {
        double potentialEnergy = 0.0;

        for (int i = 0; i < nParticles; i++) {
            int idx_i = i * dim;

            for (int j = i + 1; j < nParticles; j++) {
                int idx_j = j * dim;

                double r2 = 0.0;
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