#include "periodic_boundary.h"

#include <stdexcept>

using namespace Utils;

PeriodicBoundary::PeriodicBoundary(const std::vector<std::vector<double>>& latticeVectors) {
    dim = static_cast<int>(latticeVectors.size());
    if (dim <= 0) {
        throw std::invalid_argument("PeriodicBoundary: need at least one lattice vector");
    }

    matrixCell.assign(dim * dim, 0.0);
    for (int j = 0; j < dim; ++j) {
        if (static_cast<int>(latticeVectors[j].size()) != dim) {
            throw std::invalid_argument(
                "PeriodicBoundary: each lattice vector must have length dim");
        }
        // Store lattice vector j as column j of the row-major matrix.
        for (int i = 0; i < dim; ++i) {
            matrixCell[i * dim + j] = latticeVectors[j][i];
        }
    }

    invMatrixCell = invertMatrix(matrixCell);
}

void PeriodicBoundary::applyPeriodicBoundary(double* position) const {
    std::vector<double> s(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            s[i] += invMatrixCell[i * dim + j] * position[j];
        }
    }

    for (int i = 0; i < dim; ++i) {
        s[i] -= std::floor(s[i]);
    }

    for (int i = 0; i < dim; ++i) {
        position[i] = 0.0;
        for (int j = 0; j < dim; ++j) {
            position[i] += matrixCell[i * dim + j] * s[j];
        }
    }
}

void PeriodicBoundary::getDisplacement(const double* r1, const double* r2, double* outDisplacement) const {
    std::vector<double> s(dim);
    for (int i = 0; i < dim; ++i) s[i] = r1[i] - r2[i];

    for (int i = 0; i < dim; ++i) {
        double tmp = 0.0;
        for (int j = 0; j < dim; ++j) {
            tmp += invMatrixCell[i * dim + j] * s[j];
        }
        s[i] = tmp - std::round(tmp);
    }

    for (int i = 0; i < dim; ++i) {
        outDisplacement[i] = 0.0;
        for (int j = 0; j < dim; ++j) {
            outDisplacement[i] += matrixCell[i * dim + j] * s[j];
        }
    }
}

double PeriodicBoundary::getDistanceSq(const double* r1, const double* r2) const {
    std::vector<double> disp(dim);
    getDisplacement(r1, r2, disp.data());
    double sum = 0.0;
    for (double x : disp) sum += x * x;
    return sum;
}

double PeriodicBoundary::getDistance(const double* r1, const double* r2) const {
    return std::sqrt(getDistanceSq(r1, r2));
}

void PeriodicBoundary::fractionalToCartesian(const double* frac, double* cart) const {
    for (int i = 0; i < dim; ++i) {
        cart[i] = 0.0;
        for (int j = 0; j < dim; ++j) {
            cart[i] += matrixCell[i * dim + j] * frac[j];
        }
    }
}
