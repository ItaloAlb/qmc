#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants {
    const double PI = 3.14159265359;

    //QMC
    const int DEFAULT_N_PARTICLE = 2;
    const int DEFAULT_N_DIM = 2;


    //DMC
    const int MAX_N_WALKERS = 200000;
    const int N_WALKERS_TARGET = 5000;
    const int MAX_BRANCH_FACTOR = 3;
    const double MIN_POPULATION_RATIO = 1e-4;

    const int N_BLOCK_STEPS = 3000;
    const int N_STEPS_PER_BLOCK = 1000;
    const int RUNNING_AVERAGE_WINDOW = 1000;


    //VMC

    

    //UTILS
    const double EQUILIBRATION_STEPS = 1e4;
    const double MIN_METROPOLIS_STEP = 1e-4;
    const double MIN_DISTANCE = 1e-8;
    const double FINITE_DIFFERENCE_STEP = 1e-4;
    const double FINITE_DIFFERENCE_STEP_2 = FINITE_DIFFERENCE_STEP * FINITE_DIFFERENCE_STEP;

    const double RYDBERG_FOR_HARTREE = 2.0;
    const double RYDBERG = 13605.7; // in meV
    const double HARTREE = RYDBERG * RYDBERG_FOR_HARTREE; // in meV
    const double a0 = 0.5292; // in Angs
}

#endif