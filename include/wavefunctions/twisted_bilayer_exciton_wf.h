#include "wavefunction.h"
#include "utils.h"
#include "complex"

using namespace std::complex_literals;

class TwistedBilayerExcitonWF : public WaveFunction {
private:
    TwistedBilayerSystem moire;
public:
    TwistedBilayerExcitonWF(const std::vector<double>& params, int nParticles, int dim, TwistedBilayerSystem moire_) 
        : WaveFunction(params, nParticles, dim), moire(moire_) {}

    WaveFunction* clone() const override {
        return new TwistedBilayerExcitonWF(*this);
    }

void setParameters(const std::vector<double>& newParams) override {
        if (newParams.size() == 5) {
            double p_c2 = newParams[0];
            double p_c3 = newParams[1];
            double p_c4 = newParams[2];
            double p_c5 = newParams[2];
            double p_c6 = newParams[2];

            params[1] = std::exp(p_c2);
            params[2] = std::exp(p_c3);
            params[3] = p_c4;
            params[4] = p_c5;
            params[5] = p_c6;
        }
    }

    std::vector<double> getParameters() const override {
        return { 
            std::log(params[1]),
            std::log(params[2]),
            params[3], params[4], params[5]
        };
    }

    double jastrowEH(double r, double r2) const {
        double c1 = params[0];
        double c2 = params[1];
        double c3 = params[2];

        double exp = std::exp(-c2 * r2);
        double term1 = c1 * r2 * std::log(r) * exp;
        double term2 = c3 * r * (1.0 - exp);
        return std::exp(term1 - term2);
    }

    double jastrowEE(double r, double r2) const {
        double c4 = params[3];
        double c5 = params[4];

        double exparg = c4 * r2 * std::log(r) * std::exp(-c5 * r2);
        return std::exp(exparg);
    }

    double variationalPotential(const double* position) const {
        double xe = position[0];
        double ye = position[1];
        double xh = position[2];
        double yh = position[3];

        double c4 = params[3];
        double c5 = params[4];
        double c6 = params[5];

        double K1_dot_re = moire.k1x * xe + moire.k1y * ye;
        double K2_dot_re = moire.k2x * xe + moire.k2y * ye;
        double K3_dot_re = moire.k3x * xe + moire.k3y * ye;

        std::complex<double> f1e = (std::exp(-1i * K1_dot_re) + 
                                    std::exp(-1i * K2_dot_re) + 
                                    std::exp(-1i * K3_dot_re)) / 3.0;

        std::complex<double> f2e = (std::exp(-1i * K1_dot_re) + 
                                    std::exp(-1i * (K2_dot_re + moire.HALF_PHASE)) + 
                                    std::exp(-1i * (K3_dot_re + moire.PHASE))) / 3.0;

        double f1eSquared = std::norm(f1e);
        double f2eSquared = std::norm(f2e);

        double de = moire.d0 + moire.d1 * f1eSquared + moire.d2 * f2eSquared;

        double Ve = c4 * (f1eSquared + f2eSquared) + c6 * moire.eField * de * 0.5;

        double K1_dot_rh = moire.k1x * xh + moire.k1y * yh;
        double K2_dot_rh = moire.k2x * xh + moire.k2y * yh;
        double K3_dot_rh = moire.k3x * xh + moire.k3y * yh;

        std::complex<double> f1h = (std::exp(-1i * K1_dot_rh) + 
                                        std::exp(-1i * K2_dot_rh) + 
                                        std::exp(-1i * K3_dot_rh)) / 3.0;

        std::complex<double> f2h = (std::exp(-1i * K1_dot_rh) + 
                                    std::exp(-1i * (K2_dot_rh + moire.HALF_PHASE)) + 
                                    std::exp(-1i * (K3_dot_rh + moire.PHASE))) / 3.0;

        double f1hSquared = std::norm(f1h);
        double f2hSquared = std::norm(f2h);

        double dh = moire.d0 + moire.d1 * f1hSquared + moire.d2 * f2hSquared;

        double Vh = c5 * (f1hSquared + f2hSquared) + c6 * moire.eField * dh * 0.5;
        
        return Ve + Vh;
    }
    
    double trialWaveFunction(const double* position) const override {
        int idx_e = 0;
        int idx_h = 1 * dim;

        double r2 = moire.thicknessSquared;
        for(int k = 0; k < dim; k++) {
            double d = position[idx_e + k] - position[idx_h + k];
            r2 += d * d;
        }
        double r = std::sqrt(r2);

        return jastrowEH(r, r2) * (1 - variationalPotential(position));
    }
};