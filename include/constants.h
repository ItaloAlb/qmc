#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants {
    const double PI = 3.14159265359;

    //QMC
    const int DEFAULT_N_PARTICLE = 2;
    const int DEFAULT_N_DIM = 2;


    //DMC
    const int MAX_N_WALKERS = 100000;
    const int N_WALKERS_TARGET = 20000;
    const int MAX_BRANCH_FACTOR = 3;
    const double MIN_POPULATION_RATIO = 1e-4;


    //VMC

    

    //UTILS
    const double EQUILIBRATION_STEPS = 1e4;
    const double MIN_METROPOLIS_STEP = 1e-4;
    const double MIN_DISTANCE = 1e-8;
    const double FINITE_DIFFERENCE_STEP = 1e-4;
    const double FINITE_DIFFERENCE_STEP_2 = FINITE_DIFFERENCE_STEP * FINITE_DIFFERENCE_STEP;
}

#endif