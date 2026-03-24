#include "hamiltonian.h"

class ExcitonExcitonCoulombHamiltonian : public Hamiltonian {
public:
    ExcitonExcitonCoulombHamiltonian(int nParticles, int dim,
                       const std::vector<double>& masses,
                       const std::vector<double>& charges)
        : Hamiltonian(nParticles, dim, masses, charges) {}

    double getPotential(const double* position) const override {
        std::vector<double> R(dim, 0.0);
        if (dim > 0) {
            R[0] = 1.0;
        }

        double me = masses[0];
        double mh = masses[1];

        double r1_sq = 0.0;
        double r2_sq = 0.0;
        double term3_sq = 0.0;
        double term4_sq = 0.0;
        double term5_sq = 0.0;
        double term6_sq = 0.0;

        for (int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];
            double R_k = R[k];

            r1_sq += r1_k * r1_k;
            r2_sq += r2_k * r2_k;

            double v3_k = R_k + (mu / me) * (-r2_k + r1_k);
            term3_sq += v3_k * v3_k;

            double v4_k = R_k + (mu / mh) * (-r1_k + r2_k);
            term4_sq += v4_k * v4_k;

            double v5_k = R_k - (mu / mh) * r1_k - (mu / me) * r2_k;
            term5_sq += v5_k * v5_k;

            double v6_k = R_k + (mu / me) * r1_k + (mu / mh) * r2_k;
            term6_sq += v6_k * v6_k;
        }

        double d_sq = d * d;
        double potentialEnergy = 0.0;

        potentialEnergy -= 1.0 / std::sqrt(r1_sq + d_sq);

        potentialEnergy -= 1.0 / std::sqrt(r2_sq + d_sq);

        double mag3 = std::sqrt(term3_sq);
        if (mag3 < Constants::MIN_DISTANCE) mag3 = Constants::MIN_DISTANCE;
        potentialEnergy += 1.0 / mag3;

        double mag4 = std::sqrt(term4_sq);
        if (mag4 < Constants::MIN_DISTANCE) mag4 = Constants::MIN_DISTANCE;
        potentialEnergy += 1.0 / mag4;

        potentialEnergy -= 1.0 / std::sqrt(term5_sq + d_sq);

        potentialEnergy -= 1.0 / std::sqrt(term6_sq + d_sq);

        return potentialEnergy;
    }
};