#include "wavefunction.h"

class Hamiltonian {
    protected:
        int nParticles, dim, stride;

        std::vector<double> masses;

        std::vector<double> charges;

    public:
        Hamiltonian(int nParticles, int dim, std::vector<double> masses, std::vector<double> charges);
        virtual ~Hamiltonian() = default;

        virtual double getPotential(const double* position) const = 0;

        double getLocalEnergy(const WaveFunction& wf, const double* position) const;

        int getNParticles() const { return nParticles; };

        int getDim() const { return dim; };

        int getStride() const { return stride; };

        const std::vector<double>& getMasses() const {return this->masses; };

        const std::vector<double>& getCharges() const {return this->charges; };
};