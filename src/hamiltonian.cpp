#include "hamiltonian.h"

Hamiltonian::Hamiltonian(int nParticles_, int dim_, std::vector<double> masses_, std::vector<double> charges_) 
    : nParticles(nParticles_), dim(dim_), masses(masses_), charges(charges_)
{
    stride = nParticles_ * dim_;
}

double Hamiltonian::getLocalEnergy(const WaveFunction& wf, const double* position) const {
    double potentialEnergy = getPotential(position);

    std::vector<double> laplacian = wf.getLaplacian(position);

    double kineticEnergy = 0.0;
    for(int i = 0; i < nParticles; i++) {
        double _t = laplacian[i];
        kineticEnergy -= 0.5 / masses[i] * _t;
    }

    return kineticEnergy + potentialEnergy;
}