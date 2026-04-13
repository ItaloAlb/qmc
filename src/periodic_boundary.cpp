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
        s[i] -= std::round(s[i]);
    }

    for (int i = 0; i < dim; ++i) {
        position[i] = 0.0;
        for (int j = 0; j < dim; ++j) {
            position[i] += matrixCell[i * dim + j] * s[j];
        }
    }
}

void PeriodicBoundary::getDisplacement(const double* r1, const double* r2, double* outDisplacement) const {
    // 1. Raw Cartesian displacement
    std::vector<double> cart_disp(dim);
    for (int i = 0; i < dim; ++i) cart_disp[i] = r1[i] - r2[i];

    // 2. Fractional displacement wrapped to the home parallelepiped [-0.5, 0.5]
    std::vector<double> frac_disp(dim);
    for (int i = 0; i < dim; ++i) {
        double tmp = 0.0;
        for (int j = 0; j < dim; ++j) {
            tmp += invMatrixCell[i * dim + j] * cart_disp[j];
        }
        frac_disp[i] = tmp - std::round(tmp); 
    }

    // 3. Convert wrapped fractional displacement back to a base Cartesian displacement
    std::vector<double> base_cart_disp(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            base_cart_disp[i] += matrixCell[i * dim + j] * frac_disp[j];
        }
    }

    // 4. Search neighboring images for the absolute shortest Cartesian distance
    double min_dist_sq = -1.0;
    std::vector<int> offset(dim, -1); // Initialize offset vector to [-1, -1, ...]
    
    // Base-3 counter loop to check all {-1, 0, 1} combinations dynamically for any 'dim'
    while (true) {
        // Apply the integer fractional offset converted to Cartesian space
        std::vector<double> candidate_disp = base_cart_disp;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                candidate_disp[i] += matrixCell[i * dim + j] * static_cast<double>(offset[j]);
            }
        }
        
        // Calculate squared distance for this image
        double dist_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            dist_sq += candidate_disp[i] * candidate_disp[i];
        }
        
        // Track the shortest vector found
        if (min_dist_sq < 0 || dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            for (int i = 0; i < dim; ++i) {
                outDisplacement[i] = candidate_disp[i];
            }
        }

        // Increment offset like a base-3 counter (-1, 0, 1)
        int idx = 0;
        while (idx < dim) {
            offset[idx]++;
            if (offset[idx] > 1) {
                offset[idx] = -1; // Reset this digit and carry over
                idx++;
            } else {
                break; // No carry, stop incrementing
            }
        }
        if (idx == dim) break; // We have exhausted all 3^dim combinations
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
