#ifndef PERIODIC_BOUNDARY_H
#define PERIODIC_BOUNDARY_H

#include <vector>
#include "utils.h"

class PeriodicBoundary {
    protected:
        int dim;
        std::vector<double> matrixCell;
        std::vector<double> invMatrixCell;

    public:
        PeriodicBoundary(const std::vector<double>& latticeVectors);
        virtual ~PeriodicBoundary() = default;

        void applyPeriodicBoundary(double* position) const;

        void getDisplacement(const double* r1, const double* r2, double* disp) const;

        double getDistance(const double* r1, const double* r2) const;

        double getDistanceSq(const double* r1, const double* r2) const;
};

#endif