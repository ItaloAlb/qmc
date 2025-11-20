// src/main.cpp
#include <iostream>
#include "qmc.h" // Apenas para testar se o include funciona

int main() {
    std::cout << "Ola, Mundo! O projeto QMC compilou com sucesso." << std::endl;

    // Você pode até criar um objeto para testar a classe
    QMC qmc_solver;
    qmc_solver.run();

    return 0;
}