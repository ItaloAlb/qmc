#include "dmc.h"

using namespace Utils;

DMC::DMC(const Hamiltonian& hamiltonian_, 
        WaveFunction& wf_, 
        double deltaTau_, 
        const PeriodicBoundary* pbc_,
        int nWalkers_, 
        bool isFixedNode_, 
        bool isMaxBranch_)
    : hamiltonian(hamiltonian_),
      wf(wf_),
      pbc(pbc_),
      deltaTau(deltaTau_),
      nWalkers(nWalkers_),
      isFixedNode(isFixedNode_),
      isMaxBranch(isMaxBranch_),
      referenceEnergy(0.0),
      instEnergy(0.0),
      meanEnergy(0.0),
      blockTotalMoves(0),
      blockAcceptedMoves(0)
{
    this->invDeltaTau = 1 / deltaTau;
    this->stride = hamiltonian.getStride();
    this->nParticles = hamiltonian.getNParticles();
    this->dim = hamiltonian.getDim();

    int nThreads = omp_get_max_threads();
    if (nThreads < 1) nThreads = 1;

    omp_set_num_threads(nThreads);

    gens.resize(nThreads);

    std::random_device rd;
    for (int i = 0; i < nThreads; i++) {
        gens[i].seed(rd() + i);
    }
    
    initializeWalkers();
}


void DMC::timeStep(){
    // Create a temporary array to store the new generation of walkers
    std::vector<double> newPositions(Constants::MAX_N_WALKERS * stride);
    std::vector<double> newDrifts(Constants::MAX_N_WALKERS * stride);
    std::vector<double> newLocalEnergies(Constants::MAX_N_WALKERS);
    // Counter for the number of walkers in the new generation
    int newNWalkers = 0;
    // Accumulator for the total local energy of the new ensemble
    double ensembleEnergy = 0.0;

    #pragma omp parallel reduction(+:ensembleEnergy)
    {
        int threadId = omp_get_thread_num();
        auto& gen = gens[threadId];
        std::normal_distribution<double> dist(0.0, 1);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        // Iterate over each walker in the current ensemble
        #pragma omp for
        for(int i = 0; i < nWalkers; i++) {
            std::vector<double> newPosition(stride);
            std::vector<double> position(stride);
            std::vector<double> drift(stride);
            // Propose a new position for the walker using a random walk step and drift term
            for(int j = 0; j < stride; j++){
                position[j] = positions[i * stride + j];
                drift[j] = std::clamp(drifts[i * stride + j], - invDeltaTau, invDeltaTau);

                double chi = dist(gen) * std::sqrt(deltaTau / hamiltonian.getMasses()[j / dim]); // Random number from a normal distribution (diffusion term)
                // Update the position component: newPosition = oldPosition + diffusion_term + drift_term * time_step
                newPosition[j] = position[j] + chi + deltaTau * drift[j];
            }

            if (pbc) {
                for (int p = 0; p < nParticles; ++p) {
                    pbc->applyPeriodicBoundary(&newPosition[p * dim]);
                }
            }

            // Calculate the trial wave function at the old and new positions
            double oldPsi = wf.trialWaveFunction(&positions[i * stride]);
            double newPsi = wf.trialWaveFunction(&newPosition[0]);

            // Calculate the local energy at the old and new positions
            double oldLocalEnergy = hamiltonian.getLocalEnergy(wf, &positions[i * stride]); 
            double newLocalEnergy = hamiltonian.getLocalEnergy(wf, &newPosition[0]);
            
            // Check if the proposed move crosses a nodal surface (where Psi changes sign)
            // Moves that cross nodal surfaces are typically rejected in fixed-node DMC
            bool crossedNodalSurface;
            if (isFixedNode) {
                crossedNodalSurface = (oldPsi > 0 && newPsi < 0) || (oldPsi < 0 && newPsi > 0);
            }
            else {
                crossedNodalSurface = false;
            }

            // If the nodal surface is not crossed, proceed with the Metropolis-Hastings acceptance step
            std::vector<double> newDrift(stride); // Declare newDrift here
            if (!crossedNodalSurface) {
                // Calculate the drift at the new proposed position
                newDrift = wf.getDrift(&newPosition[0], &hamiltonian.getMasses()[0]);
                for(auto& d : newDrift) {
                    d = std::clamp(d, -invDeltaTau, invDeltaTau);
                }
                // Calculate the forward Green's function for the drift term
                double forwardDriftGreenFunction = driftGreenFunction(&newPosition[0], &positions[i * stride], &drifts[i * stride]);
                // Calculate the backward Green's function for the drift term
                double backwardDriftGreenFunction = driftGreenFunction(&positions[i * stride], &newPosition[0], &newDrift[0]);

                // Calculate the acceptance probability for the Metropolis-Hastings step
                // This ensures that the walkers sample the distribution proportional to Psi^2
                double acceptanceProbability = 
                std::min(1.0, (backwardDriftGreenFunction * newPsi * newPsi) / (forwardDriftGreenFunction * oldPsi * oldPsi));
                
                // Accept or reject the proposed move based on the acceptance probability
                if (uniform(gen) < acceptanceProbability) {
                    // If accepted, update the walker's position, drift, and local energy
                    blockAcceptedMoves++;
                    
                    for (int j = 0; j < stride; j++) {
                        position[j] = newPosition[j];
                        positions[i * stride + j] = newPosition[j];
                        drifts[i * stride + j] = newDrift[j];
                        drift[j] = newDrift[j];
                    }
                    localEnergy[i] = newLocalEnergy;
                }
                // If rejected, the walker remains at its old position with its old drift and local energy
            }
            
            // Determine the branching factor (number of copies of the walker)
            // This is based on the local energy and the reference energy (implicitly via branchGreenFunction)
            double eta = uniform(gen); // Random number for stochastic branching
            // The branch factor determines how many copies of the walker are made
            // It's typically an integer, calculated from the Green's function and a random number
            int branchFactor = static_cast<int>(eta + branchGreenFunction(localEnergy[i], oldLocalEnergy));
            if (isMaxBranch) {
                branchFactor = std::min(branchFactor, Constants::MAX_BRANCH_FACTOR);
            }
        
            // If the branch factor is positive, create copies of the walker
            if (branchFactor > 0) {
                #pragma omp critical
                for (int n = 0; n < branchFactor; n++) {
                    if (newNWalkers >= Constants::MAX_N_WALKERS) break;

                    ensembleEnergy += localEnergy[i];

                    std::copy(position.begin(), position.end(), newPositions.begin() + newNWalkers * stride);
                    std::copy(drift.begin(), drift.end(), newDrifts.begin() + newNWalkers * stride);
                    newLocalEnergies[newNWalkers] = localEnergy[i];

                    newNWalkers++;
                }
            }
        }
    }

    blockTotalMoves += nWalkers;

    // Update the instantaneous energy of the ensemble
    instEnergy = newNWalkers > 0 ? ensembleEnergy / newNWalkers: 0.0;
    // Replace the old generation of walkers with the new generation
    positions.assign(newPositions.begin(), newPositions.begin() + newNWalkers * stride);
    drifts.assign(newDrifts.begin(), newDrifts.begin() + newNWalkers * stride);
    localEnergy.assign(newLocalEnergies.begin(), newLocalEnergies.begin() + newNWalkers);
    // Update the total number of walkers
    nWalkers = newNWalkers;
}

BlockResult DMC::blockStep(int nSteps) {
    double mean = 0.0;
    double mean2 = 0.0;

    blockTotalMoves = 0;
    blockAcceptedMoves = 0;

    double delta, delta2;
    for (int n = 1; n < nSteps + 1; ++n) {
        timeStep();
        
        delta = instEnergy - mean;
        mean += delta / n;
        delta2 = instEnergy - mean;
        mean2 += delta * delta2;
    }

    BlockResult result;
    result.energy = mean;

    if (blockTotalMoves > 0) {
        result.acceptanceRatio = static_cast<double>(blockAcceptedMoves) / static_cast<double>(blockTotalMoves);
    } else {
        result.acceptanceRatio = 0.0;
    }

    if (nSteps > 1) {
        result.variance = mean2 / (nSteps - 1); 
        result.stdError = std::sqrt(result.variance / nSteps);
    } else {
        result.variance = 0.0;
        result.stdError = 0.0;
    }
    return result;
}

double DMC::driftGreenFunction(const double* newPosition, 
                               const double* oldPosition, 
                               const double* oldDrift) const {
    
    double exparg = 0.0;
    double prefactor = 1.0;
    std::vector<double> disp_min(stride);
    
    if (pbc) {
        pbc->getDisplacement(newPosition, oldPosition, disp_min.data());
    } else {
        for (int j = 0; j < stride; j++) {
            disp_min[j] = newPosition[j] - oldPosition[j];
        }
    }

    for (int j = 0; j < stride; j++) {
        double m = hamiltonian.getMasses()[j / dim];
        
        double diff = disp_min[j] - deltaTau * oldDrift[j];

        exparg -= (m * diff * diff) / (2.0 * deltaTau);

        prefactor *= std::sqrt(m / (2.0 * PI * deltaTau));
    }

    return prefactor * std::exp(exparg);
}

double DMC::branchGreenFunction(double newLocalEnergy,
                                double oldLocalEnergy) const {
    // exp(- τ/2 [E_L(R) + E_L(R') - 2E_T])
    return std::exp(- 0.5 * deltaTau * (newLocalEnergy + oldLocalEnergy - 2.0 * referenceEnergy));
}

void DMC::updateReferenceEnergy(double blockEnergy, double blockTime) {
    double ratio = static_cast<double>(nWalkers) / static_cast<double>(Constants::N_WALKERS_TARGET);
    if (ratio < Constants::MIN_POPULATION_RATIO) ratio = Constants::MIN_POPULATION_RATIO;
    referenceEnergy = blockEnergy - 1 / blockTime * std::log(ratio);
}

void DMC::initializeWalkers() {
    positions.resize(nWalkers * stride);
    drifts.resize(nWalkers * stride);
    localEnergy.resize(nWalkers);

    std::uniform_real_distribution<double> initDist(-100.0, +100.0);

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        std::random_device rd;
        
        unsigned int seed = rd() + threadId;

        Utils::Metropolis sampler(seed, 1.0, nParticles, dim); 
        
        std::mt19937 initGen(seed); 

        std::vector<double> current(stride);

        std::uniform_real_distribution<double> fracDist(0.0, 1.0);

        #pragma omp for
        for (int w = 0; w < nWalkers; w++) {

            if (pbc) {
                // Sample uniformly inside the simulation cell: fractional
                // coords in [0,1) mapped to Cartesian via the lattice matrix.
                std::vector<double> frac(dim);
                for (int p = 0; p < nParticles; ++p) {
                    for (int i = 0; i < dim; ++i) frac[i] = fracDist(initGen);
                    pbc->fractionalToCartesian(frac.data(), &current[p * dim]);
                }
            } else {
                for (int i = 0; i < stride; i++) current[i] = initDist(initGen);
            }

            double currentPsi = wf.trialWaveFunction(current.data());
            
            sampler.setStepSize(1.0); 

            int accepted = 0;
            int adjustInterval = 100;
            double targetAcc = 0.5;

            for (int j = 0; j < Constants::EQUILIBRATION_STEPS; j++) {
                
                bool accept = sampler.step(wf, current, currentPsi);

                if (accept) {
                    accepted++;
                }

                if ((j + 1) % adjustInterval == 0) {
                    double accRate = accepted / double(adjustInterval);
                    double oldStep = sampler.getStepSize();
                    double newStep = oldStep * (1.0 + 0.1 * (accRate - targetAcc));
                    
                    if (newStep < Constants::MIN_METROPOLIS_STEP) 
                        newStep = Constants::MIN_METROPOLIS_STEP;
                    
                    sampler.setStepSize(newStep);
                    accepted = 0;
                }
            }

            for (int i = 0; i < stride; i++) {
                positions[w * stride + i] = current[i];
            }

            std::vector<double> drift = wf.getDrift(current.data(), hamiltonian.getMasses().data());
            for (int i = 0; i < stride; i++) {
                drifts[w * stride + i] = drift[i];
            }

            localEnergy[w] = hamiltonian.getLocalEnergy(wf, current.data());
        }
    }

    double totalEnergy = 0.0;
    for (double e : localEnergy) totalEnergy += e;
    
    referenceEnergy = totalEnergy / nWalkers;
    meanEnergy = referenceEnergy;
}

DMCResult DMC::run(const std::string& outputFile) {
    double blockTime = deltaTau * Constants::N_STEPS_PER_BLOCK;

    std::ofstream fout(outputFile);
    std::deque<double> energyQueue;
    
    std::vector<double> blockMeanEnergies;
    blockMeanEnergies.reserve(Constants::N_BLOCK_STEPS);

    for (int j = 0; j < Constants::N_BLOCK_STEPS; j++) {
        BlockResult blockResult = blockStep(Constants::N_STEPS_PER_BLOCK);

        energyQueue.push_back(blockResult.energy);
        if (energyQueue.size() > Constants::RUNNING_AVERAGE_WINDOW) {
            energyQueue.pop_front();
        }

        meanEnergy = std::accumulate(energyQueue.begin(), energyQueue.end(), 0.0) / energyQueue.size();

        updateReferenceEnergy(blockResult.energy, blockTime); 

        fout << j << " "
             << blockResult.energy << " "
             << referenceEnergy << " "
             << meanEnergy << " "
             << nWalkers << " "
             << blockResult.variance << " "
             << blockResult.stdError << "\n";

        std::cout << "Block " << std::setw(4) << j
                    << " | Block Energy = " << std::fixed << std::setprecision(8) << blockResult.energy
                    << " | Reference Energy = " << std::fixed << std::setprecision(8) << referenceEnergy
                    << " | Mean Energy = " << std::fixed << std::setprecision(8) << meanEnergy
                    << " | Population = " << std::fixed << nWalkers
                    << " | Variance = " << std::fixed << std::setprecision(8) << blockResult.variance
                    << " | Acceptance Ratio = " << std::fixed << std::setprecision(8) << blockResult.acceptanceRatio
                    << std::endl;
    }

    fout.close();

    double variance = 0.0;
    for (double energy : energyQueue) {
        variance += (energy - meanEnergy) * (energy - meanEnergy);
    }
    
    double stdError = 0.0;
    if (energyQueue.size() > 1) {
        variance /= (energyQueue.size() - 1); 
        
        stdError = std::sqrt(variance / energyQueue.size());
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Mean Energy: " << meanEnergy << std::endl;
    std::cout << "Variance: " << variance << std::endl;
    std::cout << "Standard Error: " << stdError << std::endl;

    return { meanEnergy, variance, stdError };
}
