// src/main.cpp
#include "dmc.h"
#include "vmc.h"
#include "optimizer.h"
#include "constants.h"
#include "utils.h"

#include "wavefunctions/hydrogen_wf.h"
#include "wavefunctions/helium_wf.h"
#include "wavefunctions/monolayer_exciton_wf.h"
#include "wavefunctions/monolayer_trion_wf.h"
#include "wavefunctions/monolayer_biexciton_wf.h"
#include "wavefunctions/twisted_bilayer_exciton_wf.h"
#include "wavefunctions/bilayer_exciton_wf.h"
#include "wavefunctions/twisted_bilayer_exciton_gaussian_wf.h"

#include "hamiltonians/coulomb_hamiltonian.h"
#include "hamiltonians/efficient_rk_hamiltonian.h"
#include "hamiltonians/twisted_heterobilayer_hamiltonian.h"
#include "hamiltonians/heterobilayer_hamiltonian.h"

// int main() {
//     std::cout << "=======================\n";
//     std::cout << "   Moiré exciton (X)   \n";
//     std::cout << "=======================\n";

//     double thickness = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;
//     double theta = 0.5;
//     double eField = -100.0;

//     TwistedBilayerSystem moire(theta, eField, thickness);

//     double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     std::vector<double> initParams = {35.8, 37.8, 33.5, 36.2};

//     int nParticles = 2;
//     int nDim = 2;

//     TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
//     TwistedBilayerExcitonGaussianWF wf(initParams, nParticles, nDim, 
//         324.8537752460106, 
//         275.7926272419918,
//         324.84973392640273,
//         277.110522144819);


//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);

//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << ", "
//               << params[2] << ", "
//               << params[3] << "]\n\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.1;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

int main() {
    std::cout << "=======================\n";
    std::cout << "   Moiré exciton (X)   \n";
    std::cout << "=======================\n";

    double thickness = 6.15;
    double alpha = 1.5;
    double eps = 14.0;
    double eps1 = 4.5;
    double eps2 = 4.5;
    double theta = 0.5;
    double eField = -50.0;

    TwistedBilayerSystem moire(theta, eField, thickness);

    double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

    double me = 0.43;
    double mh = 0.35;

    std::vector<double> masses = {me,  mh};
    std::vector<double> charges = {-1.0, +1.0};

    std::vector<double> initParams = {40.2, 40.2, 37.5, 37.5};

    int nParticles = 2;
    int nDim = 2;

    TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
    TwistedBilayerExcitonGaussianWF wf(initParams, nParticles, nDim, 
        243.70652695413165, 
        140.90486103351043,
        243.6684947401863,
        140.77958422241366);


    std::random_device rd;
    unsigned int randomSeed = rd();
    
    Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

    JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
    optVariance.optimize(wf, hamiltonian, optimizerSampler);

    std::vector<double> params = wf.getParameters();
    std::cout << "Parametros Otimizados: [" 
              << params[0] << ", "
              << params[1] << ", "
              << params[2] << ", "
              << params[3] << "]\n\n";
    
    VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
    vmc.run();

    std::cout << "Energy: "             << vmc.result.energy             << "\n";
    std::cout << "Variance: "           << vmc.result.variance           << "\n";
    std::cout << "StdError: "           << vmc.result.stdError           << "\n";
    std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
    std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


    std::cout << "--- Rodando DMC ---\n";
    double deltaTau = 0.1;
    bool useFixedNode = false;
    bool useMaxBranch = true;

    DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
    dmc.run();

    return 0;
}


// int main() {
//     std::cout << "============================\n";
//     std::cout << "   Interlayer exciton (X)   \n";
//     std::cout << "============================\n";

//     double d = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;

//     double rho0 = alpha * 2 * d * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;

//     std::vector<double> initParams = {c1, 0.1, 0.1};

//     std::vector<double> optParams = {-2.30259, -3.175};

//     int nParticles = 2;
//     int nDim = 2;

//     HeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, d, rho0, eps1, eps2);
//     BilayerExcitonWF wf(initParams, nParticles, nDim, d);

//     wf.setParameters(optParams);

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.2, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << "]\n\n";

//     std::cout << "--- Rodando VMC ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// void computeMoireGridPotential(const TwistedHeterobilayerHamiltonian& hamiltonian,
//                                const TwistedBilayerSystem& moire,
//                                int nx, int ny,
//                                double system_scale_x, double system_scale_y,
//                                const std::string& filename) {
    
//     int N = nx * ny;
    
//     double size_x = system_scale_x * moire.moireLength;
//     double size_y = system_scale_y * moire.moireLength;
//     double delta_x = size_x / nx;
//     double delta_y = size_y / ny;
    
//     std::vector<double> POTE(N);
//     std::vector<double> POTH(N);
//     std::vector<double> x_coords(N);
//     std::vector<double> y_coords(N);
    
//     std::cout << "Calculando potencial de moiré na grade...\n";
//     std::cout << "nx = " << nx << ", ny = " << ny << ", N = " << N << "\n";
//     std::cout << "size_x = " << size_x << " a0, size_y = " << size_y << " a0\n";
//     std::cout << "delta_x = " << delta_x << " a0, delta_y = " << delta_y << " a0\n\n";
    
//     int l = 0;
//     for (int j = 0; j < ny; ++j) {
//         for (int i = 0; i < nx; ++i) {
//             double x = i * delta_x;
//             double y = j * delta_y;

//             double K1_dot_r = moire.k1x * x + moire.k1y * y;
//             double K2_dot_r = moire.k2x * x + moire.k2y * y;
//             double K3_dot_r = moire.k3x * x + moire.k3y * y;
            
//             std::complex<double> f1 = (std::exp(-1i * K1_dot_r) + 
//                                        std::exp(-1i * K2_dot_r) + 
//                                        std::exp(-1i * K3_dot_r)) / 3.0;
            
//             std::complex<double> f2 = (std::exp(-1i * K1_dot_r) + 
//                                        std::exp(-1i * (K2_dot_r + moire.HALF_PHASE)) + 
//                                        std::exp(-1i * (K3_dot_r + moire.PHASE))) / 3.0;
            
//             double f1Squared = std::norm(f1);
//             double f2Squared = std::norm(f2);
            
//             double d = moire.d0 + moire.d1 * f1Squared + moire.d2 * f2Squared;
            
//             double Ve = moire.Ve1 * f1Squared + moire.Ve2 * f2Squared + moire.eField * d * 0.5;
//             double Vh = moire.Vh1 * f1Squared + moire.Vh2 * f2Squared + moire.eField * d * 0.5;
            
//             POTE[l] = Ve;
//             POTH[l] = Vh;
//             x_coords[l] = x;
//             y_coords[l] = y;
//             l++;
//         }
        
//         if ((j + 1) % 10 == 0) {
//             std::cout << "Progresso: " << (j + 1) << "/" << ny << " linhas\n";
//         }
//     }
    
//     // Normalizar (subtrair mínimo como no Python)
//     double min_POTE = *std::min_element(POTE.begin(), POTE.end());
//     double min_POTH = *std::min_element(POTH.begin(), POTH.end());

//     double max_POTE = *std::max_element(POTE.begin(), POTE.end());
//     double max_POTH = *std::max_element(POTH.begin(), POTH.end());
    
//     std::cout << "\nNormalizando potenciais...\n";
//     std::cout << "min(POTE) = " << min_POTE << " Hartree\n";
//     std::cout << "min(POTH) = " << min_POTH << " Hartree\n";

//     std::cout << "max(POTE) = " << max_POTE << " Hartree\n";
//     std::cout << "max(POTH) = " << max_POTH << " Hartree\n";
    
//     for (int i = 0; i < N; ++i) {
//         POTE[i] -= min_POTE;
//         POTH[i] -= min_POTH;
//     }
    
//     std::ofstream outfile(filename);
//     outfile << std::setprecision(15);
//     outfile << "x,y,POTE,POTH\n";
    
//     for (int i = 0; i < N; ++i) {
//         outfile << x_coords[i] << "," 
//                 << y_coords[i] << "," 
//                 << POTE[i] << "," 
//                 << POTH[i] << "\n";
//     }
    
//     outfile.close();
//     std::cout << "\nPotencial salvo em: " << filename << "\n";
// }

// void computeRytovaKeldyshGrid(const TwistedHeterobilayerHamiltonian& hamiltonian,
//                               int nx, int ny,
//                               double system_scale_x, double system_scale_y,
//                               double grid_scaling, 
//                               const std::string& filename) {
    
//     int N = nx * ny;
    
//     // Define o tamanho da janela de visualização do potencial
//     double size_x = system_scale_x * grid_scaling;
//     double size_y = system_scale_y * grid_scaling;
//     double delta_x = size_x / nx;
//     double delta_y = size_y / ny;
    
//     std::vector<double> RK(N);
//     std::vector<double> x_coords(N);
//     std::vector<double> y_coords(N);
    
//     std::vector<double> pos(4, 0.0); 
    
//     std::cout << "Mapeando Potencial Rytova-Keldysh (espaco relativo)...\n";

//     int l = 0;
//     for (int j = 0; j < ny; ++j) {
//         for (int i = 0; i < nx; ++i) {
//             // Coordenada relativa r = (rx, ry)
//             double rx = i * delta_x; 
//             double ry = j * delta_y;

//             // PREPARAÇÃO PARA O C++:
//             // O método getHeterobilayerRytovaKeldysh calcula internamente:
//             // dist = pos[k] - pos[k + dim]
//             // Para que dist seja igual a rx, fazemos:
            
//             pos[0] = rx;   // x da partícula 1
//             pos[1] = ry;   // y da partícula 1
//             pos[2] = 0.0;  // x da partícula 2
//             pos[3] = 0.0;  // y da partícula 2
            
//             // Agora a distância interna será sqrt((rx-0)^2 + (ry-0)^2) = r
//             // Isso reproduz exatamente o cálculo do Python: r = sqrt(x^2 + y^2)

//             RK[l] = hamiltonian.getHeterobilayerRytovaKeldysh(pos.data());
            
//             x_coords[l] = rx;
//             y_coords[l] = ry;
//             l++;
//         }
//     }
    
//     // Normalização (subtraindo o mínimo para comparar formatos)
//     double min_RK = *std::min_element(RK.begin(), RK.end());
//     double max_RK = *std::max_element(RK.begin(), RK.end());
    
//     std::cout << "Normalizando RK...\n";
//     std::cout << "min(RK) = " << min_RK << " Ha\n";
//     std::cout << "max(RK) = " << max_RK << " Ha\n";
    
//     for (int i = 0; i < N; ++i) {
//         RK[i] -= min_RK;
//     }
    
//     std::ofstream outfile(filename);
//     outfile << std::setprecision(15);
//     outfile << "x,y,RK\n";
    
//     for (int i = 0; i < N; ++i) {
//         outfile << x_coords[i] << "," 
//                 << y_coords[i] << "," 
//                 << RK[i] << "\n";
//     }
    
//     outfile.close();
//     std::cout << "Dados salvos em: " << filename << "\n";
// }

// int main() {
//     std::cout << "=======================\n";
//     std::cout << "   Moiré Potential Grid\n";
//     std::cout << "=======================\n\n";

//     // Parâmetros do sistema
//     double thickness = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;
//     double theta = 0.5;  // em graus
//     double eField = -50.0;  // meV/Å (use 0.0 para comparação inicial)

//     TwistedBilayerSystem moire(theta, eField, thickness);
    
//     std::cout << "Comprimento de moiré: " << moire.moireLength << " a0\n";
//     std::cout << "Comprimento de moiré: " << moire.moireLength * Constants::a0 << " Angstrom\n\n";

//     double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;
//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me, mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     int nParticles = 2;
//     int nDim = 2;

//     // Criar hamiltoniano
//     TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, 
//                                                 moire, rho0, eps1, eps2);

//     // Parâmetros da grade (mesmos do Python)
//     int nx = int(100 * std::sqrt(3.0));
//     int ny = 100;
//     double system_scale_x = 1.0 * std::sqrt(3.0);
//     double system_scale_y = 1.0;

//     std::cout << "rho0 usado: " << rho0 << "\n";
//     std::cout << "thicknessSquared usado: " << moire.thicknessSquared << "\n";

//     // Calcular e salvar potencial na grade
//     computeMoireGridPotential(hamiltonian, moire, nx, ny, 
//                               system_scale_x, system_scale_y,
//                               "moire_potential_cpp.csv");

//     double grid_scaling = moire.moireLength; 

//     computeRytovaKeldyshGrid(hamiltonian, nx, ny, 
//                             system_scale_x, system_scale_y, 
//                             grid_scaling,
//                             "rk_potential_cpp.csv");

//     std::cout << "\nFinalizado!\n";

//     return 0;
// }

// int main() {
//     std::cout << "=======================\n";
//     std::cout << "   Moiré exciton (X)   \n";
//     std::cout << "=======================\n";

//     double thickness = 6.15;
//     double alpha = 1.5;
//     double eps = 14.0;
//     double eps1 = 4.5;
//     double eps2 = 4.5;
//     double theta = 0.5;
//     double eField = -300.0;

//     TwistedBilayerSystem moire(theta, eField, thickness);

//     double rho0 = alpha * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

//     double me = 0.43;
//     double mh = 0.35;

//     std::vector<double> masses = {me,  mh};
//     std::vector<double> charges = {-1.0, +1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;

//     std::vector<double> initParams = {c1, 0.1, 0.1, 0.1, 0.1, 0.1};

//     // std::vector<double> optParams = {-2.3026, -3.84062, -0.558838, -0.603411, 0.216823};

//     int nParticles = 2;
//     int nDim = 2;

//     TwistedHeterobilayerHamiltonian hamiltonian(nParticles, nDim, masses, charges, moire, rho0, eps1, eps2);
//     TwistedBilayerExcitonWF wf(initParams, nParticles, nDim, moire);

//     // wf.setParameters(optParams);

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.3, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> params = wf.getParameters();
//     std::cout << "Parametros Otimizados: [" 
//               << params[0] << ", "
//               << params[1] << ", "
//               << params[2] << ", "
//               << params[3] << ", "
//               << params[4] << "]\n\n";

//     std::cout << "--- Rodando VMC ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.1;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "==============\n";
//     std::cout << "   WS2 (XX)   \n";
//     std::cout << "==============\n";

//     double X2D = 6.393 / Constants::a0;
//     double rho0 = 2 * Constants::PI * X2D;

//     std::vector<double> masses = {0.32,  0.32, 0.35, 0.35};
//     std::vector<double> charges = {-1.0, -1.0, 1.0, 1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;
//     std::vector<double> alpha = {c1, 0.1, 0.1, c4, 0.1};

//     std::vector<double> params = {-0.0621872, -3.85307, -2.40632};

//     int nParticles = 4;
//     int nDim = 2;

//     EfficientRKHamiltonian hamiltonian(nParticles, nDim, masses, charges, rho0);
//     MonolayerBiexcitonWF wf(alpha, nParticles, nDim);

//     wf.setParameters(params);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.1, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << "]\n\n";



//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = true;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "========================================\n";
//     std::cout << "   DMC TEST: HYDROGEN ATOM (Ground State)\n";
//     std::cout << "========================================\n";

//     std::vector<double> masses = {10000.0, 1.0};
//     std::vector<double> charges = {1.0, -1.0};
//     std::vector<double> alpha = {1.0};


//     CoulombHamiltonian hamiltonian(2, 2, masses, charges);
//     HydrogenWF wf(alpha, 2, 2);

//     double deltaTau = 0.01;
    
//     bool useFixedNode = false;
//     bool useMaxBranch = false;

//     DMC dmc(hamiltonian, wf, deltaTau, 20000, useFixedNode, useMaxBranch);

//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "========================================\n";
//     std::cout << "   QMC TEST: HELIUM ATOM (Ground State)\n";
//     std::cout << "========================================\n";

//     std::vector<double> masses = {1e12, 1.0, 1.0};
//     std::vector<double> charges = {2.0, -1.0, -1.0};
    
//     std::vector<double> alpha = {0.5, 0.5}; 

//     int nParticles = 3;
//     int nDim = 2; 

//     CoulombHamiltonian hamiltonian(nParticles, nDim, masses, charges);
//     HeliumWF wf(alpha, nParticles, nDim);

//     std::random_device rd;
//     unsigned int randomSeed = rd();

//     Metropolis sampler(randomSeed, 1.0, nParticles, nDim);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     JastrowBFGSOptimizer opt(0.001, 100, 100000); 
    
//     opt.optimize(wf, hamiltonian, sampler);

//     std::cout << "Parametros Otimizados: [" 
//               << wf.getParameters()[0] << ", "
//               << wf.getParameters()[1] << "]\n\n";

//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, sampler, 1e5, 1e4); 
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";

//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);

//     dmc.run();

//     return 0;
// }

// int main() {
//     std::cout << "==============\n";
//     std::cout << "   WSe2 (X^-)   \n";
//     std::cout << "==============\n";

//     double X2D =  7.571 / Constants::a0;
//     double rho0 = 2 * Constants::PI * X2D;

//     std::vector<double> masses = {0.34,  0.34, 0.36};
//     std::vector<double> charges = {-1.0, -1.0, 1.0};

//     double c1 = masses[0] * masses[2] / 2 / (masses[0] + masses[2]);
//     double c4 = - masses[0] / 4;
//     std::vector<double> alpha = {c1, 0.1, 0.1, c4, 0.1};

//     std::vector<double> params = {-0.0621872, -3.85307, -2.40632};

//     int nParticles = 3;
//     int nDim = 2;

//     EfficientRKHamiltonian hamiltonian(nParticles, nDim, masses, charges, rho0);
//     MonolayerTrionWF wf(alpha, nParticles, nDim);

//     wf.setParameters(params);

//     std::cout << "\n--- Iniciando Otimizacao BFGS ---\n";

//     std::random_device rd;
//     unsigned int randomSeed = rd();
    
//     Metropolis optimizerSampler(randomSeed, 1.0, nParticles, nDim); 

//     JastrowBFGSOptimizer optVariance(0.05, 50, 1e5);
//     optVariance.optimize(wf, hamiltonian, optimizerSampler);


//     std::vector<double> optParams = wf.getParameters();
//     std::cout << "Parametros Otimizados (log): [" 
//               << optParams[0] << ", "
//               << optParams[1] << ", "
//               << optParams[2] << "]\n\n";



//     std::cout << "--- Rodando VMC de Producao ---\n";
    
//     VMC vmc(hamiltonian, wf, optimizerSampler, 1e7, 1e6);
//     vmc.run();

//     std::cout << "Energy: "             << vmc.result.energy             << "\n";
//     std::cout << "Variance: "           << vmc.result.variance           << "\n";
//     std::cout << "StdError: "           << vmc.result.stdError           << "\n";
//     std::cout << "metropolisStepSize: " << vmc.result.metropolisStepSize << "\n";
//     std::cout << "acceptanceRate: "     << vmc.result.acceptanceRate     << "\n\n";


//     std::cout << "--- Rodando DMC ---\n";
//     double deltaTau = 0.01;
//     bool useFixedNode = false;
//     bool useMaxBranch = true;

//     DMC dmc(hamiltonian, wf, deltaTau, Constants::N_WALKERS_TARGET, useFixedNode, useMaxBranch);
//     dmc.run();

//     return 0;
// }