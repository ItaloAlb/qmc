#include "vmc.h"

VMC::VMC(const Hamiltonian& hamiltonian_, WaveFunction& wf_, int nSteps_, int nEquilibration_)
    : hamiltonian(hamiltonian_),
      wf(wf_),
      nSteps(nSteps_),
      nEquilibration(nEquilibration_),
      uniformDist(0.0, 1.0)
{
    this->stride = hamiltonian.getStride();

    int nThreads = omp_get_max_threads();
    if (nThreads < 1) nThreads = 1;
    omp_set_num_threads(nThreads);

    gens.resize(nThreads);
    std::random_device rd;
    for (int i = 0; i < nThreads; i++) {
        gens[i].seed(rd() + i);
    }
}

VMCResult VMC::run(const std::vector<double>& alpha, std::mt19937& local_rng) {
    wf.setParameters(alpha);

    std::vector<double> currentPosition(stride);
    std::vector<double> proposedPosition(stride);

    double L = 1.0;
    double step = 1.0;

    std::uniform_real_distribution<double> rStep(-1.0, 1.0); 
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);
    std::uniform_real_distribution<double> initialDistribution(-L, +L);

    for (int i = 0; i < stride; ++i) {
        currentPosition[i] = initialDistribution(local_rng);
    }

    double targetAcc = 0.5;
    double adjustRate = 0.10;
    int adjustInterval = 100;
    int acceptedEquil = 0;

    for (int i = 0; i < nEquilibration; i++) {
        for (int j = 0; j < stride; ++j) {
            proposedPosition[j] = currentPosition[j] + rStep(local_rng) * step;
        }

        double proposedPsi = wf.trialWaveFunction(&proposedPosition[0]);

        double currentPsi  = wf.trialWaveFunction(&currentPosition[0]);

        double w = (proposedPsi * proposedPsi) / (currentPsi * currentPsi);

        if (w >= uniformDist(local_rng)) {
            acceptedEquil++;
            for (int j = 0; j < stride; ++j) {
                currentPosition[j] = proposedPosition[j];
            }
        }

        if ((i + 1) % adjustInterval == 0) {
            double accRate = (double)acceptedEquil / adjustInterval;
            step *= (1.0 + adjustRate * (accRate - targetAcc));

            if (step < 1e-4) step = 1e-4;

            acceptedEquil = 0;
        }
    }

    int nBlocks = 100;
    if (nSteps < nBlocks) nBlocks = 1; 

    int blockSize = nSteps / nBlocks;
    int effectiveSteps = nBlocks * blockSize;

    std::vector<double> blockEnergies;
    blockEnergies.reserve(nBlocks);

    double totalEnergySum = 0.0;
    double totalEnergySquaredSum = 0.0;
    int acceptedMoves = 0;

    for (int iBlock = 0; iBlock < nBlocks; ++iBlock) {
        double currentBlockEnergySum = 0.0;

        for (int iStep = 0; iStep < blockSize; ++iStep) {
            
            // Move
            for (int k = 0; k < stride; ++k) {
                proposedPosition[k] = currentPosition[k] + rStep(local_rng) * step;
            }

            double proposedPsi = wf.trialWaveFunction(&proposedPosition[0]);
            double currentPsi  = wf.trialWaveFunction(&currentPosition[0]);

            double w = (proposedPsi * proposedPsi) / (currentPsi * currentPsi);

            if (w >= uniformDist(local_rng)) {
                acceptedMoves++;
                for (int k = 0; k < stride; ++k) {
                    currentPosition[k] = proposedPosition[k];
                }
            }
            
            // Measure
            double localEnergy = hamiltonian.getLocalEnergy(wf, &currentPosition[0]);

            currentBlockEnergySum += localEnergy;
            totalEnergySum += localEnergy;
            totalEnergySquaredSum += localEnergy * localEnergy;
        }

        blockEnergies.push_back(currentBlockEnergySum / blockSize);
    }

    VMCResult result;
    result.energy = totalEnergySum / effectiveSteps;
    
    double energyMeanSquared = totalEnergySquaredSum / effectiveSteps;
    result.variance = energyMeanSquared - (result.energy * result.energy);

    if (nBlocks > 1) {
        double blockSum = 0.0;
        double blockSumSquared = 0.0;
        for (double blockEnergy : blockEnergies) {
            blockSum += blockEnergy;
            blockSumSquared += blockEnergy * blockEnergy;
        }

        double blockMean = blockSum / nBlocks;
        double blockMeanSquared = blockSumSquared / nBlocks;
        double blockVariance = blockMeanSquared - (blockMean * blockMean);

        result.stdError = std::sqrt(blockVariance / (nBlocks - 1.0));
    } else {
        result.stdError = 0.0;
    }

    result.metropolisStepSize = step;
    result.acceptanceRate = (double)acceptedMoves / (double)effectiveSteps;

    return result;
}

std::vector<double> VMC::optimizeParameters(
    const std::vector<double>& alphaStart, 
    const std::vector<double>& alphaEnd, 
    const std::vector<double>& alphaStep,
    bool isMinimizeVariance
)
{
    size_t D = alphaStart.size();
    if (D == 0) return {};

    double bestObjective = 1e20;
    std::vector<double> bestAlpha(D);
    VMCResult bestResult;

    std::vector<int> stepsPerDim(D);
    std::vector<long long> cumSteps(D);
    long long totalIterations = 1;

    for (size_t d = 0; d < D; ++d) {
        stepsPerDim[d] = static_cast<int>(
            std::round((alphaEnd[d] - alphaStart[d]) / alphaStep[d] + 1e-9)
        ) + 1;
        cumSteps[d] = totalIterations;
        totalIterations *= stepsPerDim[d];
    }

    #pragma omp parallel
    {
        std::unique_ptr<WaveFunction> local_wf(this->wf.clone());

        VMC local_vmc(this->hamiltonian, *local_wf, this->nSteps, this->nEquilibration);

        std::vector<double> currentAlpha(D);
        int thread_id = omp_get_thread_num();
        std::mt19937& local_rng = gens[thread_id];

        double threadBestObj = 1e20;
        std::vector<double> threadBestAlpha(D);
        VMCResult threadBestResult;

        #pragma omp for nowait
        for (long long k = 0; k < totalIterations; ++k) {
            long long tmp = k;
            for (size_t d = D; d-- > 0; ) {
                int idx = tmp / cumSteps[d];
                currentAlpha[d] = alphaStart[d] + idx * alphaStep[d];
                tmp -= idx * cumSteps[d];
            }

            VMCResult _result = local_vmc.run(currentAlpha, local_rng);

            double obj = isMinimizeVariance ? _result.variance : _result.energy;

            if (obj < threadBestObj) {
                threadBestObj = obj;
                threadBestAlpha = currentAlpha;
                threadBestResult = _result;
            }
        }

        #pragma omp critical
        {
            if (threadBestObj < bestObjective) {
                bestObjective = threadBestObj;
                bestAlpha = threadBestAlpha;
                bestResult = threadBestResult;
            }
        }
    }

    this->result = bestResult;
    return bestAlpha;
}