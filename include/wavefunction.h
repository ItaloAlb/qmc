#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <vector>
#include <cmath>

#include "constants.h"

class PeriodicBoundary;

class WaveFunction {
    protected:
        int nParticles, dim, stride;

        const PeriodicBoundary* pbc;
        

        std::vector<double> params;
    public:
        WaveFunction(std::vector<double> params,
            int nParticles = Constants::DEFAULT_N_PARTICLE, 
            int dim = Constants::DEFAULT_N_DIM,
            const PeriodicBoundary* pbc = nullptr);

        virtual WaveFunction* clone() const = 0;

        virtual ~WaveFunction() = default; 

        std::vector<double> getDrift(const double* position) const;

        virtual void setParameters(const std::vector<double>& newParams);
        
        virtual std::vector<double> getParameters() const;

        virtual std::vector<double> getDrift(const double* position, const double* masses) const;

        virtual std::vector<double> getLaplacian(const double* position) const;

        std::vector<double> parameterGradient(const double* position);

        virtual double trialWaveFunction(const double* position) const = 0;

        int getNParticles() const { return nParticles; };

        int getDim() const { return dim; };

        int getStride() const { return stride; };
};

#endif