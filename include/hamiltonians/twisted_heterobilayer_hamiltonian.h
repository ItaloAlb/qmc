#include "hamiltonian.h"
#include "utils.h"
#include <complex>

using namespace Constants;
using namespace Utils;
using namespace std::complex_literals;

class TwistedHeterobilayerHamiltonian : public Hamiltonian {
    private:
        const double Vh1 = -107.1 / HARTREE;
        const double Vh2 = -(107.1 - 16.9) / HARTREE; 
        const double Ve1 = -17.3 / HARTREE;
        const double Ve2 = -(17.3 - 3.5) / HARTREE;
        const double d0 = 6.387 / a0;
        const double d1 = 0.544 / a0;
        const double d2 = 0.042 / a0;

        const double a10 = 3.282;
        const double a20 = 3.160;
        const double delta = std::abs(a10 - a20) / a10;

        double theta;
        double moireLength;
        double absK;
        double Efield;
        double eps1, eps2;

        double K1x;
        double K1y;
        double K2x;
        double K2y;
        double K3x;
        double K3y;

        const double HALF_PHASE = 2.0 * PI / 3.0;
        const double PHASE = 4.0 * PI / 3.0;

        double rho0;
        double invrho0;
    public:
        TwistedHeterobilayerHamiltonian(int nParticles, int dim,
                        const std::vector<double>& masses,
                        const std::vector<double>& charges,
                        double theta_, double Efield_, double rho0_)
            : Hamiltonian(nParticles, dim, masses, charges),
            rho0(rho0_), theta(theta_), Efield(Efield_) {
                invrho0 = 1.0 / rho0;
            }

        double getTMDMoirePotential(const double* position) const {
                double xe = position[0];
                double ye = position[1];
                double xh = position[2];
                double yh = position[3];

                double K1_dot_re = K1x * xe + K1y * ye;
                double K2_dot_re = K2x * xe + K2y * ye;
                double K3_dot_re = K3x * xe + K3y * ye;

                std::complex<double> f1e = (std::exp(-1i * K1_dot_re) + 
                                            std::exp(-1i * K2_dot_re) + 
                                            std::exp(-1i * K3_dot_re)) / 3.0;

                std::complex<double> f2e = (std::exp(-1i * K1_dot_re) + 
                                            std::exp(-1i * (K2_dot_re + HALF_PHASE)) + 
                                            std::exp(-1i * (K3_dot_re + PHASE))) / 3.0;

                double f1eSquared = std::norm(f1e);
                double f2eSquared = std::norm(f2e);

                double de = d0 + d1 * f1eSquared + d2 * f2eSquared;

                double Ve = Ve1 * f1eSquared + Ve2 * f2eSquared + Efield * de * 0.5;

                double K1_dot_rh = K1x * xh + K1y * yh;
                double K2_dot_rh = K2x * xh + K2y * yh;
                double K3_dot_rh = K3x * xh + K3y * yh;

                    std::complex<double> f1h = (std::exp(-1i * K1_dot_rh) + 
                                                std::exp(-1i * K2_dot_rh) + 
                                                std::exp(-1i * K3_dot_rh)) / 3.0;

                std::complex<double> f2h = (std::exp(-1i * K1_dot_rh) + 
                                            std::exp(-1i * (K2_dot_rh + HALF_PHASE)) + 
                                            std::exp(-1i * (K3_dot_rh + PHASE))) / 3.0;

                double f1hSquared = std::norm(f1h);
                double f2hSquared = std::norm(f2h);

                double dh = d0 + d1 * f1hSquared + d2 * f2hSquared;

                double Vh = Vh1 * f1hSquared + Vh2 * f2hSquared + Efield * dh * 0.5;
                
                return Ve + Vh;

        }

        double getHeterobilayerRytovaKeldysh(const double* position) const {
            double r2 = 0.0;

            for(int k = 0; k < dim; k++) {
                double dist = position[k] - position[k + dim];
                r2 += dist * dist;
            }
            double r = std::sqrt(r2);
            double r_times_inv_rho0 = r * invrho0;
            
            return - PI / (eps1 + eps2) * invrho0 * (stvh0(r_times_inv_rho0) - jy0b(r_times_inv_rho0));
        }

        double getPotential(const double* position) const override {
            return getHeterobilayerRytovaKeldysh(position) + getTMDMoirePotential(position);
        }
};


// double DMC::potentialEnergy(const double* position) const {
//     double xe = position[0];
//     double ye = position[1];
//     double xh = position[2];
//     double yh = position[3];

//     double dx_eh = xe - xh;
//     double dy_eh = ye - yh;

//     double r_eh = std::sqrt(dx_eh * dx_eh + dy_eh * dy_eh + thickness * thickness);

//     double h0 = stvh0(r_eh / r0);
//     double y0 = jy0b(r_eh / r0);

//     double Vrk = - PI / ((eps1 + eps2) * r0) * (h0 - y0);

    // double K1_dot_re = K1x * xe + K1y * ye;
    // double K2_dot_re = K2x * xe + K2y * ye;
    // double K3_dot_re = K3x * xe + K3y * ye;

//     std::complex<double> f1_e = (std::exp(-1i * K1_dot_re) + 
//                                    std::exp(-1i * K2_dot_re) + 
//                                    std::exp(-1i * K3_dot_re)) / 3.0;

//     std::complex<double> f2_e = (std::exp(-1i * K1_dot_re) + 
//                                    std::exp(-1i * (K2_dot_re + theta_s_div_2)) + 
//                                    std::exp(-1i * (K3_dot_re + theta_s))) / 3.0;

//     double f1_sq_e = std::norm(f1_e);
//     double f2_sq_e = std::norm(f2_e);

//     double C_e = DIL_C0 + DIL_C1 * f1_sq_e + DIL_C2 * f2_sq_e;
//     double V_E_e = E_field * C_e * 0.5;

//     double Ve = Ve1 * f1_sq_e + Ve2 * f2_sq_e + V_E_e;
    
//     double K1_dot_rh = K1x * xh + K1y * yh;
//     double K2_dot_rh = K2x * xh + K2y * yh;
//     double K3_dot_rh = K3x * xh + K3y * yh;

//     std::complex<double> f1_h = (std::exp(-1i * K1_dot_rh) + 
//                                    std::exp(-1i * K2_dot_rh) + 
//                                    std::exp(-1i * K3_dot_rh)) / 3.0;

//     std::complex<double> f2_h = (std::exp(-1i * K1_dot_rh) + 
//                                    std::exp(-1i * (K2_dot_rh + theta_s_div_2)) + 
//                                    std::exp(-1i * (K3_dot_rh + theta_s))) / 3.0;

//     double f1_sq_h = std::norm(f1_h);
//     double f2_sq_h = std::norm(f2_h);

//     double C_h = DIL_C0 + DIL_C1 * f1_sq_h + DIL_C2 * f2_sq_h;
//     double V_E_h = E_field * C_h * 0.5;

//     double Vh = Vh1 * f1_sq_h + Vh2 * f2_sq_h + V_E_h;
    
//     return Ve + Vh;
// }