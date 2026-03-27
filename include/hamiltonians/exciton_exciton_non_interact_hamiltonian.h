#include "hamiltonian.h"

class ExcitonExcitonNonInteractHamiltonian : public Hamiltonian {
private:
    double me;
    double mh;
    double d;
    double R;
public:
    ExcitonExcitonNonInteractHamiltonian(int nParticles, int dim,
                       const std::vector<double>& masses,
                       const std::vector<double>& charges,
                       double me, double mh, double d)
        : Hamiltonian(nParticles, dim, masses, charges), me(me), mh(mh), d(d) {}

    double getPotential(const double* position) const override {
        double mu = masses[0];

        double r1_sq = 0.0;
        double r2_sq = 0.0;
        double term3_sq = 0.0;
        double term4_sq = 0.0;
        double term5_sq = 0.0;
        double term6_sq = 0.0;

        for (int k = 0; k < dim; k++) {
            double r1_k = position[0 * dim + k];
            double r2_k = position[1 * dim + k];

            r1_sq += r1_k * r1_k;
            r2_sq += r2_k * r2_k;
        }

        double d_sq = d * d;
        double potentialEnergy = 0.0;

        potentialEnergy -= 1.0 / std::sqrt(r1_sq + d_sq);

        potentialEnergy -= 1.0 / std::sqrt(r2_sq + d_sq);

        return potentialEnergy;
    }
};