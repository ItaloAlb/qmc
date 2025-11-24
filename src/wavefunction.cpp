#include "wavefunction.h"

WaveFunction::WaveFunction(std::vector<double>params_, int nParticles_, int dim_)
    : nParticles(nParticles_),
      dim(dim_),
      params(params_)
{
    stride = nParticles_ * dim_;
}

std::vector<double> WaveFunction::getDrift(const double* position, const double* masses) const {
    std::vector<double> drift(stride, 0.0);

    std::vector<double> Rp(position, position + stride);
    std::vector<double> Rm(position, position + stride);
    for (int i = 0; i < stride; i++) {
        Rp[i] += Constants::FINITE_DIFFERENCE_STEP;
        Rm[i] -= Constants::FINITE_DIFFERENCE_STEP;

        double forwardPsi = std::log(std::abs(trialWaveFunction(&Rp[0])));
        double backwardPsi = std::log(std::abs(trialWaveFunction(&Rm[0])));

        Rp[i] -= Constants::FINITE_DIFFERENCE_STEP;
        Rm[i] += Constants::FINITE_DIFFERENCE_STEP;

        double lnDiff = forwardPsi - backwardPsi;
        drift[i] = lnDiff / (2.0 * Constants::FINITE_DIFFERENCE_STEP) / masses[i / dim];
    }
    return drift;
}

void WaveFunction::setParameters(const std::vector<double>& newParams) {
    this->params = newParams;
}

const std::vector<double>& WaveFunction::getParameters() const {
    return this->params;
}

std::vector<double> WaveFunction::getLaplacian(const double* position) const {
    std::vector<double> laplacians(nParticles, 0.0);

    double lnPsiCenter = std::log(std::abs(trialWaveFunction(position)));

    double h = Constants::FINITE_DIFFERENCE_STEP;
    double h2 = Constants::FINITE_DIFFERENCE_STEP_2;
    double twoH = 2.0 * h;

    std::vector<double> pos(position, position + stride);

    for (int p = 0; p < nParticles; p++) {
        double lapLn = 0.0; 
        double gradLnSq = 0.0;

        int idx = p * dim;
        
        for (int d = 0; d < dim; d++) {
            int i = idx + d; 

            pos[i] += h;
            double forwardLnPsi = std::log(std::abs(trialWaveFunction(&pos[0])));
            
            pos[i] -= twoH;
            double backwardLnPsi = std::log(std::abs(trialWaveFunction(&pos[0])));
            
            pos[i] += h; 

            lapLn += (forwardLnPsi - 2.0 * lnPsiCenter + backwardLnPsi) / h2;

            double deriv = (forwardLnPsi - backwardLnPsi) / twoH;
            gradLnSq += deriv * deriv;
        }
        laplacians[p] = lapLn + gradLnSq;
    }

    return laplacians;
}

std::vector<double> WaveFunction::parameterGradient(const double* position) {
    std::vector<double> optParams = this->getParameters();
    
    std::vector<double> gradients(optParams.size(), 0.0);
    
    double eps = Constants::FINITE_DIFFERENCE_STEP; 

    for (size_t i = 0; i < optParams.size(); ++i) {
        double originalVal = optParams[i];
        optParams[i] = originalVal + eps;
        this->setParameters(optParams);
        
        double psiPlus = trialWaveFunction(position);

        optParams[i] = originalVal - eps;
        this->setParameters(optParams);
        
        double psiMinus = trialWaveFunction(position);

        gradients[i] = (psiPlus - psiMinus) / (2.0 * eps);

        optParams[i] = originalVal;
        this->setParameters(optParams);
    }

    return gradients;
}