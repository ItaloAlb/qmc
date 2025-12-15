#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include "hamiltonian.h"
#include "wavefunction.h"
#include "utils.h"
#include "constants.h"

using namespace Utils;


class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void optimize(WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler) = 0;
};


class JastrowBFGSOptimizer : public Optimizer {
private:
    double learningRate;
    int maxEpochs;
    int samplesPerEpoch;

    std::vector<double> currentPosition;
    bool isInitialized = false;
    double currentStepSize;

    void adjustStepSize(WaveFunction& wf, Metropolis& sampler, double& currentPsi) {
        double targetAcc = 0.5;
        int adjustInterval = 100;
        int acceptedCount = 0;
        
        for (int i = 0; i < Constants::EQUILIBRATION_STEPS; ++i) { 
            if (sampler.step(wf, currentPosition, currentPsi)) {
                acceptedCount++;
            }
            
            if ((i + 1) % adjustInterval == 0) {
                double accRate = (double)acceptedCount / adjustInterval;
                double oldStep = sampler.getStepSize();
                
                double newStep = oldStep * (1.0 + 0.5 * (accRate - targetAcc));

                sampler.setStepSize(newStep);
                currentStepSize = newStep;
                acceptedCount = 0;
            }
        }
    }

    std::pair<double, std::vector<double>> computeGradients(
        WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler, const std::vector<double>& params) 
    {
        int nParams = params.size();

        double currentPsi = wf.trialWaveFunction(currentPosition.data());

        for(int i=0; i<Constants::EQUILIBRATION_STEPS; ++i) {
            sampler.step(wf, currentPosition, currentPsi);
        }

        double sumE = 0.0;
        std::vector<double> sumO(nParams, 0.0);
        std::vector<double> sumEO(nParams, 0.0);
        
        double accepted = 0.0;

        for (int step = 0; step < samplesPerEpoch; ++step) {
            bool acc = sampler.step(wf, currentPosition, currentPsi);
            if(acc) accepted += 1.0;
            
            double EL = ham.getLocalEnergy(wf, currentPosition.data());
            std::vector<double> O = wf.parameterGradient(currentPosition.data());

            sumE += EL;


            for (int i = 0; i < nParams; ++i) {
                sumO[i]   += O[i];
                sumEO[i]  += EL * O[i];
            }
        }

        double accRate = accepted / samplesPerEpoch;
        if (accRate < 0.2 || accRate > 0.8) {
             adjustStepSize(wf, sampler, currentPsi);
        }

        double avgE = sumE / samplesPerEpoch;
        std::vector<double> finalGrad(nParams);

        for (int i = 0; i < nParams; ++i) {
            double avgO = sumO[i] / samplesPerEpoch;
            double avgEO = sumEO[i] / samplesPerEpoch;
            finalGrad[i] = 2.0 * (avgEO - (avgE * avgO));
        }

        return {avgE, finalGrad};
    }
    public:
        JastrowBFGSOptimizer(double lr, int epochs, int samples)
        : learningRate(lr), maxEpochs(epochs), samplesPerEpoch(samples) {}

        void optimize(WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler) override {
            std::vector<double> currentParams = wf.getParameters();
            int nParams = currentParams.size();

            currentPosition.resize(wf.getStride());
            std::mt19937 tempGen(1234);
            std::uniform_real_distribution<double> dist(-1.0, 1.0); 
            for (auto& pos : currentPosition) pos = dist(tempGen);
            
            currentStepSize = sampler.getStepSize(); 
            if(currentStepSize < 0.1) currentStepSize = 1.0;
            sampler.setStepSize(currentStepSize);

            double psiTmp = wf.trialWaveFunction(currentPosition.data());
            adjustStepSize(wf, sampler, psiTmp);

            std::vector<std::vector<double>> H(nParams, std::vector<double>(nParams, 0.0));
            for (int i = 0; i < nParams; ++i) H[i][i] = 1.0;

            std::cout << "BFGS" << std::endl;
            auto [energyOld, gradOld] = computeGradients(wf, ham, sampler, currentParams);
            std::cout << "Energia Inicial: " << energyOld << std::endl;

            for (int epoch = 0; epoch < maxEpochs; ++epoch) {
                
                std::vector<double> p = matVecMul(H, gradOld);
                for (auto& val : p) val = -val;

                std::vector<double> newParams = currentParams;
                for (int i = 0; i < nParams; ++i) {
                    newParams[i] += learningRate * p[i];
                }
                wf.setParameters(newParams);

                auto [energyNew, gradNew] = computeGradients(wf, ham, sampler, newParams);

                std::vector<double> s(nParams), y(nParams);
                for (int i = 0; i < nParams; ++i) {
                    s[i] = newParams[i] - currentParams[i];
                    y[i] = gradNew[i] - gradOld[i];
                }

                double rho = dot(y, s);
                if (std::abs(rho) > 1e-10) {
                    rho = 1.0 / rho;
                    
                    std::vector<std::vector<double>> term1(nParams, std::vector<double>(nParams));
                    std::vector<std::vector<double>> term2(nParams, std::vector<double>(nParams));
                    
                    for(int i=0; i<nParams; ++i) {
                        for(int j=0; j<nParams; ++j) {
                            double I_ij = (i==j ? 1.0 : 0.0);
                            term1[i][j] = I_ij - rho * s[i] * y[j];
                            term2[i][j] = I_ij - rho * y[i] * s[j];
                        }
                    }
                    
                    std::vector<std::vector<double>> temp(nParams, std::vector<double>(nParams, 0.0));
                    for(int i=0; i<nParams; ++i)
                        for(int j=0; j<nParams; ++j)
                            for(int k=0; k<nParams; ++k)
                                temp[i][j] += term1[i][k] * H[k][j];
                                
                    std::vector<std::vector<double>> H_new(nParams, std::vector<double>(nParams, 0.0));
                    for(int i=0; i<nParams; ++i) {
                        for(int j=0; j<nParams; ++j) {
                            double val = 0.0;
                            for(int k=0; k<nParams; ++k) val += temp[i][k] * term2[k][j];
                            H_new[i][j] = val + rho * s[i] * s[j];
                        }
                    }
                    H = H_new;
                }

                currentParams = newParams;
                gradOld = gradNew;

                std::cout << "Epoch " << epoch+1 << " | E = " << energyNew << std::endl;
            }
        }
};

#endif