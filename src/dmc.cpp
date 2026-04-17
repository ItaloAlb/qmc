#include "dmc.h"

using namespace Utils;

DMC::DMC(const Hamiltonian& hamiltonian_,
        WaveFunction& wf_,
        double deltaTau_,
        const PeriodicBoundary* pbc_,
        int nWalkersTarget_,
        bool isFixedNode_,
        bool isMaxBranch_,
        bool dumpWalkers_,
        bool descendantWeighting_,
        bool checkpoint_,
        bool resumeFromCheckpoint_,
        int tLagBlocks_,
        int taggingIntervalBlocks_,
        int equilibrationBlocks_,
        int accumulationBlocks_,
        int nStepsPerBlock_)
    : hamiltonian(hamiltonian_),
      wf(wf_),
      pbc(pbc_),
      deltaTau(deltaTau_),
      nWalkers(nWalkersTarget_),
      nWalkersTarget(nWalkersTarget_),
      isFixedNode(isFixedNode_),
      isMaxBranch(isMaxBranch_),
      dumpWalkers(dumpWalkers_),
      descendantWeighting(descendantWeighting_),
      checkpoint(checkpoint_),
      resumeFromCheckpoint(resumeFromCheckpoint_),
      equilibrationBlocks(equilibrationBlocks_),
      accumulationBlocks(accumulationBlocks_),
      nStepsPerBlock(nStepsPerBlock_),
      tLagBlocks(tLagBlocks_),
      taggingIntervalBlocks(taggingIntervalBlocks_),
      referenceEnergy(0.0),
      instEnergy(0.0),
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
    int newNWalkers = 0;
    double ensembleEnergy = 0.0;

    #pragma omp parallel reduction(+:ensembleEnergy)
    {
        int threadId = omp_get_thread_num();
        auto& gen = gens[threadId];
        std::normal_distribution<double> dist(0.0, 1);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        #pragma omp for
        for(int i = 0; i < nWalkers; i++) {
            thread_local std::vector<double> newPosition(stride);
            thread_local std::vector<double> position(stride);
            thread_local std::vector<double> drift(stride);
            // Propose a new position for the walker using a random walk step and drift term
            for(int j = 0; j < stride; j++){
                position[j] = positions[i * stride + j];
                drift[j] = std::clamp(drifts[i * stride + j], - invDeltaTau, invDeltaTau);

                double chi = dist(gen) * std::sqrt(deltaTau / hamiltonian.getMasses()[j / dim]);
                newPosition[j] = position[j] + chi + deltaTau * drift[j];
            }

            if (pbc) {
                for (int p = 0; p < nParticles; ++p) {
                    pbc->applyPeriodicBoundary(&newPosition[p * dim]);
                }
            }

            double oldPsi = wf.trialWaveFunction(&positions[i * stride]);
            double newPsi = wf.trialWaveFunction(&newPosition[0]);

            double oldLocalEnergy = hamiltonian.getLocalEnergy(wf, &positions[i * stride]);
            double newLocalEnergy = hamiltonian.getLocalEnergy(wf, &newPosition[0]);

            bool crossedNodalSurface;
            if (isFixedNode) {
                crossedNodalSurface = (oldPsi > 0 && newPsi < 0) || (oldPsi < 0 && newPsi > 0);
            }
            else {
                crossedNodalSurface = false;
            }

            thread_local std::vector<double> newDrift(stride);
            if (!crossedNodalSurface) {
                newDrift = wf.getDrift(&newPosition[0], &hamiltonian.getMasses()[0]);
                for(auto& d : newDrift) {
                    d = std::clamp(d, -invDeltaTau, invDeltaTau);
                }
                double forwardDriftGreenFunction = driftGreenFunction(&newPosition[0], &positions[i * stride], &drifts[i * stride]);
                double backwardDriftGreenFunction = driftGreenFunction(&positions[i * stride], &newPosition[0], &newDrift[0]);

                double acceptanceProbability =
                std::min(1.0, (backwardDriftGreenFunction * newPsi * newPsi) / (forwardDriftGreenFunction * oldPsi * oldPsi));

                if (uniform(gen) < acceptanceProbability) {
                    blockAcceptedMoves++;

                    for (int j = 0; j < stride; j++) {
                        position[j] = newPosition[j];
                        positions[i * stride + j] = newPosition[j];
                        drifts[i * stride + j] = newDrift[j];
                        drift[j] = newDrift[j];
                    }
                    localEnergy[i] = newLocalEnergy;
                }
            }

            double eta = uniform(gen);
            int branchFactor = static_cast<int>(eta + branchGreenFunction(localEnergy[i], oldLocalEnergy));
            if (isMaxBranch) {
                branchFactor = std::min(branchFactor, Constants::MAX_BRANCH_FACTOR);
            }

            if (branchFactor > 0) {
                #pragma omp critical
                for (int n = 0; n < branchFactor; n++) {
                    if (newNWalkers >= Constants::MAX_N_WALKERS) break;

                    ensembleEnergy += localEnergy[i];

                    std::copy(position.begin(), position.end(), newPositionsScratch.begin() + newNWalkers * stride);
                    std::copy(drift.begin(), drift.end(), newDriftsScratch.begin() + newNWalkers * stride);
                    newLocalEnergiesScratch[newNWalkers] = localEnergy[i];
                    if (!ancestorsHistory.empty()) {
                        for (size_t k = 0; k < ancestorsHistory.size(); ++k) {
                            newAncestorsHistory[k][newNWalkers] = ancestorsHistory[k][i];
                        }
                    }

                    newNWalkers++;
                }
            }
        }
    }

    blockTotalMoves += nWalkers;

    instEnergy = newNWalkers > 0 ? ensembleEnergy / newNWalkers: 0.0;
    positions.assign(newPositionsScratch.begin(), newPositionsScratch.begin() + newNWalkers * stride);
    drifts.assign(newDriftsScratch.begin(), newDriftsScratch.begin() + newNWalkers * stride);
    localEnergy.assign(newLocalEnergiesScratch.begin(), newLocalEnergiesScratch.begin() + newNWalkers);
    if (!ancestorsHistory.empty()) {
        for (size_t k = 0; k < ancestorsHistory.size(); ++k) {
            ancestorsHistory[k].assign(newAncestorsHistory[k].begin(), newAncestorsHistory[k].begin() + newNWalkers);
        }
    }
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
    } else {
        result.variance = 0.0;
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
    double ratio = static_cast<double>(nWalkers) / static_cast<double>(nWalkersTarget);
    if (ratio < Constants::MIN_POPULATION_RATIO) ratio = Constants::MIN_POPULATION_RATIO;
    referenceEnergy = blockEnergy - 1 / blockTime * std::log(ratio);
}

void DMC::initializeWalkers() {
    positions.resize(nWalkers * stride);
    drifts.resize(nWalkers * stride);
    localEnergy.resize(nWalkers);
    newPositionsScratch.resize(Constants::MAX_N_WALKERS * stride);
    newDriftsScratch.resize(Constants::MAX_N_WALKERS * stride);
    newLocalEnergiesScratch.resize(Constants::MAX_N_WALKERS);

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
                    if (pbc) {
                        for (int p = 0; p < nParticles; ++p)
                            pbc->applyPeriodicBoundary(&current[p * dim]);
                        currentPsi = wf.trialWaveFunction(current.data());
                    }
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
}

void DMC::saveCheckpoint(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Warning: could not open checkpoint file for writing: " << path << std::endl;
        return;
    }

    int32_t st = stride;
    int32_t np = nParticles;
    int32_t dm = dim;
    int32_t nw = nWalkers;
    double dt = deltaTau;
    double refE = referenceEnergy;

    f.write(reinterpret_cast<const char*>(&st), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&np), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&dm), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&nw), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&dt), sizeof(double));
    f.write(reinterpret_cast<const char*>(&refE), sizeof(double));
    f.write(reinterpret_cast<const char*>(positions.data()),
            static_cast<std::streamsize>(nWalkers) * stride * sizeof(double));
}

bool DMC::loadCheckpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    int32_t st, np, dm, nw;
    double dt, refE;
    f.read(reinterpret_cast<char*>(&st), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&np), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dm), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&nw), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&dt), sizeof(double));
    f.read(reinterpret_cast<char*>(&refE), sizeof(double));

    if (!f) {
        std::cerr << "Warning: checkpoint header read failed: " << path << std::endl;
        return false;
    }

    if (st != stride || np != nParticles || dm != dim) {
        std::cerr << "Warning: checkpoint shape mismatch (stride/nParticles/dim). Ignoring." << std::endl;
        return false;
    }

    if (nw <= 0 || nw > Constants::MAX_N_WALKERS) {
        std::cerr << "Warning: checkpoint walker count out of range: " << nw << ". Ignoring." << std::endl;
        return false;
    }

    positions.assign(static_cast<size_t>(nw) * stride, 0.0);
    f.read(reinterpret_cast<char*>(positions.data()),
           static_cast<std::streamsize>(nw) * stride * sizeof(double));
    if (!f) {
        std::cerr << "Warning: checkpoint positions read failed: " << path << std::endl;
        return false;
    }

    nWalkers = nw;
    referenceEnergy = refE;

    drifts.assign(static_cast<size_t>(nWalkers) * stride, 0.0);
    localEnergy.assign(nWalkers, 0.0);
    for (int w = 0; w < nWalkers; ++w) {
        std::vector<double> d = wf.getDrift(&positions[w * stride], hamiltonian.getMasses().data());
        for (int i = 0; i < stride; ++i) drifts[w * stride + i] = d[i];
        localEnergy[w] = hamiltonian.getLocalEnergy(wf, &positions[w * stride]);
    }

    std::cout << "Resumed from checkpoint: " << path
              << " | nWalkers = " << nWalkers
              << " | referenceEnergy = " << std::fixed << std::setprecision(8) << referenceEnergy
              << " | saved dt = " << dt
              << " | current dt = " << deltaTau
              << std::endl;

    return true;
}

DMCResult DMC::run(const std::string& outputFile) {
    double blockTime = deltaTau * nStepsPerBlock;

    std::ofstream fout;
    std::ofstream walkerFile;
    std::ofstream descFile;

    std::string checkpointPath = outputFile.substr(0, outputFile.rfind('.')) + "_checkpoint.bin";

    if (resumeFromCheckpoint) {
        if (!loadCheckpoint(checkpointPath)) {
            std::cout << "No usable checkpoint at " << checkpointPath
                      << " — running full equilibration." << std::endl;
        }
    }

    for (int j = 0; j < equilibrationBlocks; j++) {
        BlockResult blockResult = blockStep(nStepsPerBlock);
        updateReferenceEnergy(blockResult.energy, blockTime);

        std::cout << "Block " << std::setw(4) << j
                    << " | Block Energy = " << std::fixed << std::setprecision(8) << blockResult.energy
                    << " | Reference Energy = " << std::fixed << std::setprecision(8) << referenceEnergy
                    << " | Population = " << std::fixed << nWalkers
                    << " | Variance = " << std::fixed << std::setprecision(8) << blockResult.variance
                    << " | Acceptance Ratio = " << std::fixed << std::setprecision(8) << blockResult.acceptanceRatio
                    << std::endl;
    }

    if (checkpoint) {
        saveCheckpoint(checkpointPath);
        std::cout << "Checkpoint written: " << checkpointPath << std::endl;
    }

    fout.open(outputFile);
    if (dumpWalkers) {
        std::string walkerPath = outputFile.substr(0, outputFile.rfind('.')) + "_walkers.bin";
        walkerFile.open(walkerPath, std::ios::binary);
    }

    // Descendant weighting file (binary).
    // Format per tagging event:
    //   int32  nTagged
    //   int32  stride
    //   double taggedPositions[nTagged * stride]
    //   int32  descendantCounts[nTagged]
    if (descendantWeighting) {
        std::string descPath = outputFile.substr(0, outputFile.rfind('.')) + "_descendants.bin";
        descFile.open(descPath, std::ios::binary);
    }

    std::vector<double> blockEnergies;
    blockEnergies.reserve(accumulationBlocks);

    for (int k = 0; k < accumulationBlocks; k++) {
        int j = equilibrationBlocks + k;

        if (descendantWeighting && descFile.is_open() && (k % taggingIntervalBlocks == 0)) {
            ancestorsHistory.emplace_back(nWalkers);
            newAncestorsHistory.emplace_back(Constants::MAX_N_WALKERS);
            taggedPositionsHistory.emplace_back(nWalkers * stride);
            taggedCountHistory.emplace_back(nWalkers);
            taggingBlocksHistory.emplace_back(k);

            std::vector<int>& currentAncestors = ancestorsHistory.back();
            std::vector<double>& currentTaggedPositions = taggedPositionsHistory.back();

            for (int i = 0; i < nWalkers; ++i) {
                currentAncestors[i] = i;
            }
            std::copy(positions.begin(), positions.begin() + nWalkers * stride, currentTaggedPositions.begin());
        }

        BlockResult blockResult = blockStep(nStepsPerBlock);

        // --- Descendant weighting: harvest completed tagging events ---
        while (!taggingBlocksHistory.empty() && (k - taggingBlocksHistory.front()) >= tLagBlocks) {
            int currentNTagged = taggedCountHistory.front();
            std::vector<double>& currentTaggedPositions = taggedPositionsHistory.front();

            std::vector<int32_t> descendantCount(currentNTagged, 0);
            for (int i = 0; i < nWalkers; ++i) {
                int a = ancestorsHistory.front()[i];
                if (a >= 0 && a < currentNTagged) descendantCount[a]++;
            }

            int32_t nt = currentNTagged;
            int32_t st = stride;
            descFile.write(reinterpret_cast<const char*>(&nt), sizeof(int32_t));
            descFile.write(reinterpret_cast<const char*>(&st), sizeof(int32_t));
            descFile.write(reinterpret_cast<const char*>(currentTaggedPositions.data()),
                           currentNTagged * stride * sizeof(double));
            descFile.write(reinterpret_cast<const char*>(descendantCount.data()),
                           currentNTagged * sizeof(int32_t));

            ancestorsHistory.pop_front();
            newAncestorsHistory.pop_front();
            taggedPositionsHistory.pop_front();
            taggedCountHistory.pop_front();
            taggingBlocksHistory.pop_front();
        }

        updateReferenceEnergy(blockResult.energy, blockTime);

        blockEnergies.push_back(blockResult.energy);

        fout << j << " "
             << blockResult.energy << " "
             << referenceEnergy << " "
             << nWalkers << " "
             << blockResult.variance << " "
             << blockResult.acceptanceRatio << "\n";

        if (dumpWalkers && walkerFile.is_open()) {
            int32_t nw = nWalkers;
            int32_t st = stride;
            walkerFile.write(reinterpret_cast<const char*>(&nw), sizeof(int32_t));
            walkerFile.write(reinterpret_cast<const char*>(&st), sizeof(int32_t));
            walkerFile.write(reinterpret_cast<const char*>(positions.data()),
                             nWalkers * stride * sizeof(double));
        }

        std::cout << "Block " << std::setw(4) << j
                    << " | Block Energy = " << std::fixed << std::setprecision(8) << blockResult.energy
                    << " | Reference Energy = " << std::fixed << std::setprecision(8) << referenceEnergy
                    << " | Population = " << std::fixed << nWalkers
                    << " | Variance = " << std::fixed << std::setprecision(8) << blockResult.variance
                    << " | Acceptance Ratio = " << std::fixed << std::setprecision(8) << blockResult.acceptanceRatio
                    << std::endl;
    }

    fout.close();
    if (walkerFile.is_open()) walkerFile.close();
    if (descFile.is_open()) descFile.close();

    double mean = 0.0;
    for (double e : blockEnergies) mean += e;
    if (!blockEnergies.empty()) mean /= blockEnergies.size();

    double variance = 0.0;
    for (double e : blockEnergies) variance += (e - mean) * (e - mean);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Mean Energy: " << mean << std::endl;
    std::cout << "Variance: " << variance << std::endl;

    return { mean, variance };
}
