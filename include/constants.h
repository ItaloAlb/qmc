#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants {
    const double PI = 3.14159265359;

    //QMC
    const int DEFAULT_N_PARTICLE = 2;
    const int DEFAULT_N_DIM = 2;


    //DMC
    const int MAX_N_WALKERS = 200000;
    const int N_WALKERS_TARGET = 10000;
    const int MAX_BRANCH_FACTOR = 3;
    const double MIN_POPULATION_RATIO = 1e-4;

    const int N_BLOCK_STEPS = 10000;
    const int N_STEPS_PER_BLOCK = 100;
    const int RUNNING_AVERAGE_WINDOW = 500;


    //VMC

    

    //UTILS
    const double EQUILIBRATION_STEPS = 1e4;
    const double MIN_METROPOLIS_STEP = 1e-4;
    const double MIN_DISTANCE = 1e-8;
    const double FINITE_DIFFERENCE_STEP = 1e-4;
    const double FINITE_DIFFERENCE_STEP_2 = FINITE_DIFFERENCE_STEP * FINITE_DIFFERENCE_STEP;
}

#endif