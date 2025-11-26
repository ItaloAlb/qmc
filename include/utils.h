#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <cmath>
#include <functional>

#include "wavefunction.h"
#include "constants.h"

using namespace Constants;

struct MoireSystem {
    const double Vh1 = -107.1 / HARTREE;
    const double Vh2 = -(107.1 - 16.9) / HARTREE; 
    const double Ve1 = -17.3 / HARTREE;
    const double Ve2 = -(17.3 - 3.5) / HARTREE;

    const double d0 = 6.387 / a0;
    const double d1 = 0.544 / a0;
    const double d2 = 0.042 / a0;

    const double a10 = 3.282;
    const double a20 = 3.160;
    const double delta = std::abs(a10 - a20) / a10;


    const double HALF_PHASE = 2.0 * PI / 3.0;
    const double PHASE = 4.0 * PI / 3.0;

    double theta;
    double eField;
    
    double moireLength;
    double absK;
    double k1x, k1y, k2x, k2y, k3x, k3y;
    
    MoireSystem(double theta_, double eField_) 
        : theta(theta_), eField(eField_)
    {
        moireLength = a10 / std::sqrt(theta*theta + delta*delta) / a0;
        absK = (4.0 * PI) / (3.0 * moireLength);

        k1x = absK * 1.0;
        k1y = absK * 0.0;
        k2x = absK * (-0.5);
        k2y = absK * (std::sqrt(3.0) / 2.0);
        k3x = absK * (-0.5);
        k3y = absK * (-std::sqrt(3.0) / 2.0);
    }
};


namespace Utils {

    class Metropolis {
        private:
            std::mt19937 rng;
            double stepSize;
            std::vector<double> positionBuffer; 

        public:
            Metropolis(int seed, double step, int nParticles, int dim) 
                : rng(seed), stepSize(step) {
                positionBuffer.resize(nParticles * dim);
            }

            bool step(WaveFunction& wf, std::vector<double>& currentR, double& currentPsi) {
                
                positionBuffer = currentR; 

                std::uniform_real_distribution<double> dist(-stepSize, stepSize);
                std::uniform_real_distribution<double> uniform(0.0, 1.0);

                for (double& position : positionBuffer) {
                    position += dist(rng);
                }

                double proposedPsi = wf.trialWaveFunction(positionBuffer.data());

                double w = (proposedPsi * proposedPsi) / (currentPsi * currentPsi);

                if (w >= uniform(rng)) {
                    currentR = positionBuffer;
                    currentPsi = proposedPsi;
                    return true;
                }
                return false;
            }
            
            double getStepSize() const { return stepSize; }
            void setStepSize(double s) { stepSize = s; }
        };

    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

    inline std::vector<double> matVecMul(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
        std::vector<double> res(v.size(), 0.0);
        for (size_t i = 0; i < M.size(); ++i) {
            for (size_t j = 0; j < M[i].size(); ++j) {
                res[i] += M[i][j] * v[j];
            }
        }
        return res;
    }

    inline double stvh0(double x) {
        double s = 1.0;
        double r = 1.0;
        double sh0;

        if (x <= 20.0) {
            double a0 = 2.0 * x / PI;
            for (int k = 1; k <= 60; ++k) {
                r = -r * (x / (2.0 * k + 1.0)) * (x / (2.0 * k + 1.0));
                s = s + r;
                if (std::abs(r) < std::abs(s) * 1.0e-12) {
                    break;
                }
            }
            sh0 = a0 * s;
        } else {
            int km = static_cast<int>(0.5 * (x + 1.0));
            if (x >= 50.0) {
                km = 25;
            }
            for (int k = 1; k <= km; ++k) {
                double term = (2.0 * k - 1.0) / x;
                r = -r * term * term;
                s = s + r;
                if (std::abs(r) < std::abs(s) * 1.0e-12) {
                    break;
                }
            }

            double t = 4.0 / x;
            double t2 = t * t;
            double p0 = ((((-0.37043e-5 * t2 + 0.173565e-4) * t2 - 0.487613e-4) * t2 + 0.17343e-3) * t2 - 0.1753062e-2) * t2 + 0.3989422793;
            double q0 = t * (((((0.32312e-5 * t2 - 0.142078e-4) * t2 + 0.342468e-4) * t2 - 0.869791e-4) * t2 + 0.4564324e-3) * t2 - 0.0124669441);
            double ta0 = x - 0.25 * PI;
            double by0 = 2.0 / std::sqrt(x) * (p0 * std::sin(ta0) + q0 * std::cos(ta0));
            sh0 = 2.0 / (PI * x) * s + by0;
        }

        return sh0;
    }

    inline double jy0b(double x) {
        if (x == 0.0) {
            return -1.0e300; 
        } else if (x <= 4.0) {
            double t = x / 4.0;
            double t2 = t * t;

            double bj0 = (((((( -0.5014415e-3*t2 + 0.76771853e-2 )*t2 - 0.0709253492 )*t2
                    + 0.4443584263 )*t2 - 1.7777560599 )*t2 + 3.9999973021 )*t2
                    - 3.9999998721 )*t2 + 1.0;

            double by0 = (((((( -0.567433e-4*t2 + 0.859977e-3 )*t2 - 0.94855882e-2 )*t2
                    + 0.0772975809 )*t2 - 0.4261737419 )*t2 + 1.4216421221 )*t2
                    - 2.3498519931 )*t2 + 1.0766115157;
            by0 = by0 * t2 + 0.3674669052;
            by0 = 2.0 / PI * std::log(x / 2.0) * bj0 + by0;

            return by0;
        } else {
            double t = 4.0 / x;
            double t2 = t * t;
            double a0 = std::sqrt(2.0 / (PI * x));

            double p0 = ((((-0.9285e-5*t2 + 0.43506e-4 )*t2 - 0.122226e-3 )*t2
                    + 0.434725e-3 )*t2 - 0.4394275e-2 )*t2 + 0.999999997;
            double q0 = t*((((( 0.8099e-5*t2 - 0.35614e-4 )*t2 + 0.85844e-4 )*t2
                    - 0.218024e-3 )*t2 + 0.1144106e-2 )*t2 - 0.031249995);
            double ta0 = x - 0.25 * PI;
            double by0 = a0 * (p0 * std::sin(ta0) + q0 * std::cos(ta0));

            return by0;
        }
    }
}

#endif

