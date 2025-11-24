#include "hamiltonian.h"
#include "wavefunction.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void optimize(WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler) = 0;
};