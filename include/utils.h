#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <functional>

#include "constants.h"

class WaveFunction;

using namespace Constants;

struct TwistedBilayerSystem {
    const double Vh1;
    const double Vh2;
    const double Ve1;
    const double Ve2;

    const double d0;
    const double d1;
    const double d2;

    const double a10;
    const double a20;

    const double HALF_PHASE = 2.0 * PI / 3.0;
    const double PHASE = 4.0 * PI / 3.0;

    double theta;
    double eField;

    double moireLength;
    double absK;
    double k1x, k1y, k2x, k2y, k3x, k3y;

    TwistedBilayerSystem(double a10_, double a20_,
                         double theta_, double eField_,
                         double Vh1_, double Vh2_,
                         double Ve1_, double Ve2_,
                         double d0_, double d1_, double d2_)
        : Vh1(Vh1_ / HARTREE),
          Vh2(Vh2_ / HARTREE),
          Ve1(Ve1_ / HARTREE),
          Ve2(Ve2_ / HARTREE),
          d0(d0_ / a0),
          d1(d1_ / a0),
          d2(d2_ / a0),
          a10(a10_ / a0),
          a20(a20_ / a0),
          theta(theta_ * PI / 180),
          eField(eField_ * a0 / HARTREE)
    {
        double delta = std::abs(a10 - a20);
        moireLength = a10 / std::sqrt(theta*theta + delta*delta);
        absK = (4.0 * PI) / (3.0 * moireLength);

        k1x = absK;        k1y = 0.0;
        k2x = -0.5 * absK; k2y =  absK * std::sqrt(3.0) / 2.0;
        k3x = -0.5 * absK; k3y = -absK * std::sqrt(3.0) / 2.0;
    }

    double getCarrierPotential(double x, double y, double V1, double V2) const {
        using namespace std::complex_literals;

        double K1r = k1x * x + k1y * y;
        double K2r = k2x * x + k2y * y;
        double K3r = k3x * x + k3y * y;

        std::complex<double> f1 = (std::exp(-1i * K1r) +
                                   std::exp(-1i * K2r) +
                                   std::exp(-1i * K3r)) / 3.0;

        std::complex<double> f2 = (std::exp(-1i * K1r) +
                                   std::exp(-1i * (K2r + HALF_PHASE)) +
                                   std::exp(-1i * (K3r + PHASE))) / 3.0;

        double f1Sq = std::norm(f1);
        double f2Sq = std::norm(f2);

        double d = d0 + d1 * f1Sq + d2 * f2Sq;

        return V1 * f1Sq + V2 * f2Sq + eField * d * 0.5;
    }

    double getExcitonMoirePotential(const double* position) const {
        double Ve = getCarrierPotential(position[0], position[1], Ve1, Ve2);
        double Vh = getCarrierPotential(position[2], position[3], Vh1, Vh2);
        return Ve + Vh;
    }
};

namespace Utils {

class Metropolis {
private:
    std::mt19937 rng;
    double stepSize;
    std::vector<double> positionBuffer;

public:
    Metropolis(int seed, double step, int nParticles, int dim);

    bool step(WaveFunction& wf, std::vector<double>& currentR, double& currentPsi);

    double getStepSize() const { return stepSize; }
    void setStepSize(double s) { stepSize = s; }
};

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return sum;
    }

inline std::vector<double> matVecMul(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
        std::vector<double> res(v.size(), 0.0);
        for (int i = 0; i < M.size(); ++i) {
            for (int j = 0; j < M[i].size(); ++j) {
                res[i] += M[i][j] * v[j];
            }
        }
        return res;
    }

inline std::vector<double> invertMatrix(const std::vector<double>& M) {
        int totalSize = M.size();
        int dim = static_cast<int>(std::sqrt(totalSize));

        std::vector<double> temp = M;
        std::vector<double> inv(totalSize, 0.0);
        for (int i = 0; i < dim; ++i) {
            inv[i * dim + i] = 1.0;
        }

        const double EPSILON = 1e-12;

        for (int i = 0; i < dim; ++i) {
            
            double pivotValue = temp[i * dim + i];
            int pivotRow = i;

            for (int k = i + 1; k < dim; ++k) {
                if (std::abs(temp[k * dim + i]) > std::abs(pivotValue)) {
                    pivotValue = temp[k * dim + i];
                    pivotRow = k;
                }
            }

            if (pivotRow != i) {
                for (int j = 0; j < dim; ++j) {
                    std::swap(temp[i * dim + j], temp[pivotRow * dim + j]);
                    std::swap(inv[i * dim + j],  inv[pivotRow * dim + j]);
                }
            }
            
            pivotValue = temp[i * dim + i]; 

            double invPivot = 1.0 / pivotValue;
            for (int j = 0; j < dim; ++j) {
                temp[i * dim + j] *= invPivot;
                inv[i * dim + j]  *= invPivot;
            }

            for (int k = 0; k < dim; ++k) {
                if (k != i) {
                    double factor = temp[k * dim + i];
                    for (int j = 0; j < dim; ++j) {
                        temp[k * dim + j] -= factor * temp[i * dim + j];
                        inv[k * dim + j]  -= factor * inv[i * dim + j];
                    }
                }
            }
        }

        return inv;
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

} // namespace Utils

#endif
