#include "wavefunction.h"
#include "utils.h"

class TwistedBilayerExcitonGaussianWF : public WaveFunction {
private:
    double xe0, ye0;
    double xh0, yh0;

    inline double gaussian2D(double x, double y, double x0, double y0,
                                   double sigma_x, double sigma_y) const {
        double dx = x - x0;
        double dy = y - y0;
        double exponent = -0.5 * ( (dx * dx) / (sigma_x * sigma_x)
                                 + (dy * dy) / (sigma_y * sigma_y) );
        return std::exp(exponent);
    }

public:
    TwistedBilayerExcitonGaussianWF(
        const std::vector<double>& params,
        int nParticles,
        int dim,
        double xe0_, double ye0_,
        double xh0_, double yh0_
    ) : WaveFunction(params, nParticles, dim),
        xe0(xe0_), ye0(ye0_),
        xh0(xh0_), yh0(yh0_)
    {}

    WaveFunction* clone() const override {
        return new TwistedBilayerExcitonGaussianWF(*this);
    }

    void setParameters(const std::vector<double>& newParams) override {
        params[0] = newParams[0];
        params[1] = newParams[1];
        params[2] = newParams[2];
        params[3] = newParams[3];
    }

    std::vector<double> getParameters() const override {
        return {
            params[0],
            params[1],
            params[2],
            params[3]
        };
    }

    double trialWaveFunction(const double* position) const override {
        double xe = position[0];
        double ye = position[1];
        double xh = position[2];
        double yh = position[3];

        double sigma_xe = params[0];
        double sigma_ye = params[1];
        double sigma_xh = params[2];
        double sigma_yh = params[3];

        double phi_e = gaussian2D(xe, ye, xe0, ye0, sigma_xe, sigma_ye);
        double phi_h = gaussian2D(xh, yh, xh0, yh0, sigma_xh, sigma_yh);

        return phi_e * phi_h;
    }
};