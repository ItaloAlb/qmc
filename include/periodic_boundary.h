#ifndef PERIODIC_BOUNDARY_H
#define PERIODIC_BOUNDARY_H

#include <vector>
#include "utils.h"

// Parallelogram / parallelepiped simulation cell with periodic boundaries.
//
// The cell is defined by `dim` lattice vectors (each of length `dim`) passed
// explicitly at construction. The stored `matrixCell` has the lattice vectors
// as its columns (row-major flattening), so
//     Cartesian   = matrixCell    · fractional
//     fractional  = invMatrixCell · Cartesian
//
// `applyPeriodicBoundary` wraps a position into the fundamental cell
// (fractional coords in [0, 1)). `getDisplacement` returns the minimum-image
// displacement r1 − r2 and is correct for any sufficiently orthogonal cell
// (the standard `s − round(s)` convention).
class PeriodicBoundary {
    protected:
        int dim;
        std::vector<double> matrixCell;
        std::vector<double> invMatrixCell;

    public:
        // Each entry of `latticeVectors` is one lattice vector in Cartesian
        // coordinates. The number of vectors must equal the length of each
        // vector (= dim).
        //
        // Example (2D square box of side L):
        //     PeriodicBoundary pbc({{L, 0}, {0, L}});
        //
        // Example (2D hexagonal cell with lattice constant a):
        //     PeriodicBoundary pbc({{a, 0}, {a/2, a*std::sqrt(3.0)/2}});
        PeriodicBoundary(const std::vector<std::vector<double>>& latticeVectors);
        virtual ~PeriodicBoundary() = default;

        int getDim() const { return dim; }
        const std::vector<double>& getMatrixCell() const { return matrixCell; }

        // Wrap a position into the fundamental cell (fractional [0,1)).
        void applyPeriodicBoundary(double* position) const;

        // Minimum-image displacement r1 − r2.
        void getDisplacement(const double* r1, const double* r2, double* disp) const;

        double getDistance(const double* r1, const double* r2) const;
        double getDistanceSq(const double* r1, const double* r2) const;

        // Map fractional coordinates (length dim, in [0,1)) to Cartesian.
        // Convenience helper for uniformly seeding walkers inside the cell.
        void fractionalToCartesian(const double* frac, double* cart) const;
};

#endif
