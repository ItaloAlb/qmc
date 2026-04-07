#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include "hamiltonian.h"
#include "wavefunction.h"
#include "utils.h"
#include "constants.h"

using namespace Utils;


class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void optimize(WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler) = 0;
};


// =============================================================================
// LinearMethodOptimizer
//
// Energy minimization via the zero-variance linear method of
// Umrigar, Toulouse, Filippi, Sorella & Hennig, PRL 98, 110201 (2007).
//
// At each epoch we sample R from |Ψ_0|^2 and accumulate the matrices
//
//   S_00 = 1
//   S_0j = <O_j>                              S_i0 = <O_i>
//   S_ij = <O_i O_j>
//
//   H_00 = <E_L>
//   H_0j = <∂_j E_L>  + <E_L O_j>             H_i0 = <O_i E_L>
//   H_ij = <O_i ∂_j E_L> + <O_i O_j E_L>
//
// where O_i = ∂(lnΨ)/∂p_i.  The H estimator is intentionally non-symmetric:
// it is the only choice giving the strong zero-variance principle of Eq. (4)
// of the paper.  The lowest eigenvalue solution of  H Δp = E S Δp  with the
// constraint Δp_0 = 1 gives the parameter update.  We then apply the
// non-linear rescaling of Eq. (7-8) (with ξ = 1/2 by default) so that the
// linear approximation behind the method remains valid for non-linear
// parameters, and stabilize H by adding a small constant `adiag` to all
// diagonal elements except H_00.
// =============================================================================
class LinearMethodOptimizer : public Optimizer {
private:
    int    maxEpochs;
    int    samplesPerEpoch;
    double xi;        // ξ in Eq. (8); 0 = stable/min ‖Ψ_lin-Ψ_0‖, 1/2 = balanced, 1 = Sorella SR
    double adiag;     // diagonal stabilization added to H

    std::vector<double> currentPosition;
    double currentStepSize;

    void adjustStepSize(WaveFunction& wf, Metropolis& sampler, double& currentPsi) {
        double targetAcc = 0.5;
        int adjustInterval = 100;
        int acceptedCount = 0;

        for (int i = 0; i < Constants::EQUILIBRATION_STEPS; ++i) {
            if (sampler.step(wf, currentPosition, currentPsi)) acceptedCount++;

            if ((i + 1) % adjustInterval == 0) {
                double accRate = (double)acceptedCount / adjustInterval;
                double oldStep = sampler.getStepSize();
                double newStep = oldStep * (1.0 + 0.5 * (accRate - targetAcc));
                sampler.setStepSize(newStep);
                currentStepSize = newStep;
                acceptedCount = 0;
            }
        }
    }

    // O_i = ∂(lnΨ)/∂p_i  AND  ∂E_L/∂p_i  via central differences.
    // Both quantities are computed in a single sweep over the parameters so
    // we re-use the perturbed wavefunctions/energies (halves the cost).
    // Restores the wavefunction parameters before returning.
    struct Derivatives {
        std::vector<double> O;    // log-derivatives of Ψ
        std::vector<double> dE;   // derivatives of the local energy
    };

    Derivatives computeDerivatives(WaveFunction& wf, Hamiltonian& ham, const double* position) {
        std::vector<double> p = wf.getParameters();
        int N = (int)p.size();

        Derivatives d;
        d.O.resize(N);
        d.dE.resize(N);

        const double eps = 1e-5;
        const double inv2eps = 1.0 / (2.0 * eps);

        for (int i = 0; i < N; ++i) {
            double saved = p[i];

            p[i] = saved + eps;
            wf.setParameters(p);
            double lnPsiPlus = std::log(std::abs(wf.trialWaveFunction(position)));
            double EPlus     = ham.getLocalEnergy(wf, position);

            p[i] = saved - eps;
            wf.setParameters(p);
            double lnPsiMinus = std::log(std::abs(wf.trialWaveFunction(position)));
            double EMinus     = ham.getLocalEnergy(wf, position);

            p[i] = saved;
            wf.setParameters(p);

            d.O[i]  = (lnPsiPlus - lnPsiMinus) * inv2eps;
            d.dE[i] = (EPlus     - EMinus)     * inv2eps;
        }
        return d;
    }

    // Solve  H x = λ S x  by inverse iteration with shift σ, normalising the
    // returned eigenvector so its first component is 1 (Δp_0 = 1).
    std::vector<double> solveGeneralizedEig(
        const std::vector<std::vector<double>>& H,
        const std::vector<std::vector<double>>& S,
        double sigma) const
    {
        int n = (int)H.size();

        // M = H − σ S, flattened row-major for invertMatrix
        std::vector<double> M(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                M[i * n + j] = H[i][j] - sigma * S[i][j];

        std::vector<double> Minv = invertMatrix(M);

        std::vector<double> x(n, 0.0);
        x[0] = 1.0;

        for (int iter = 0; iter < 100; ++iter) {
            // y = S x
            std::vector<double> y(n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    y[i] += S[i][j] * x[j];

            // x_new = M^{-1} y
            std::vector<double> xNew(n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    xNew[i] += Minv[i * n + j] * y[j];

            if (std::abs(xNew[0]) < 1e-15) break;
            for (auto& v : xNew) v /= xNew[0];

            double diff = 0.0;
            for (int i = 0; i < n; ++i) diff += std::abs(xNew[i] - x[i]);
            x = xNew;
            if (diff < 1e-10) break;
        }
        return x;
    }

    // Non-linear rescaling of Δp following Eq. (7) and the choice of N_i in
    // Eq. (8) of Umrigar et al. PRL 98, 110201 (2007).  All parameters are
    // treated as non-linear (the safe choice — for purely linear params N_i
    // would be zero and the rescaling reduces to the identity).
    std::vector<double> applyRescaling(
        const std::vector<double>& dp,
        const std::vector<std::vector<double>>& S) const
    {
        int n = (int)dp.size() - 1;

        double sum1 = 0.0;  // Σ_j S_0j Δp_j
        double sum2 = 0.0;  // Σ_{j,k} S_jk Δp_j Δp_k
        for (int j = 1; j <= n; ++j) {
            sum1 += S[0][j] * dp[j];
            for (int k = 1; k <= n; ++k)
                sum2 += S[j][k] * dp[j] * dp[k];
        }

        double D = std::sqrt(std::max(0.0, 1.0 + 2.0 * sum1 + sum2));
        double denom = xi * D + (1.0 - xi) * (1.0 + sum1);
        if (std::abs(denom) < 1e-15) return dp;

        double rescale = 1.0;
        for (int i = 1; i <= n; ++i) {
            double sumSij_dpj = 0.0;
            for (int j = 1; j <= n; ++j) sumSij_dpj += S[i][j] * dp[j];

            double Ni = -(xi * D * S[0][i]
                          + (1.0 - xi) * (S[0][i] + sumSij_dpj)) / denom;
            rescale -= Ni * dp[i];
        }
        if (std::abs(rescale) < 1e-15) return dp;

        std::vector<double> result(dp.size());
        for (size_t i = 0; i < dp.size(); ++i) result[i] = dp[i] / rescale;
        return result;
    }

    // Correlated-sampling energy estimate for a candidate parameter set p_new.
    // Runs a short Metropolis chain at p_old (sampling from |Ψ_old|^2) and
    // reweights each step by |Ψ_new/Ψ_old|^2 on the fly — no stored configs.
    // Restores the original parameters before returning.
    double estimateEnergyCorrelated(
        WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler,
        const std::vector<double>& p_new,
        const std::vector<double>& p_old,
        std::vector<double>& position,
        int nSteps) const
    {
        std::vector<double> original_params = wf.getParameters();

        wf.setParameters(p_old);
        double psi_old = wf.trialWaveFunction(position.data());

        double total_weighted_E = 0.0;
        double total_W = 0.0;

        for (int step = 0; step < nSteps; ++step) {
            sampler.step(wf, position, psi_old);             // chain stays at p_old
            double psi_old_sq = psi_old * psi_old;
            if (psi_old_sq < 1e-18) continue;

            wf.setParameters(p_new);
            double psi_new = wf.trialWaveFunction(position.data());
            double EL_new  = ham.getLocalEnergy(wf, position.data());
            double W = (psi_new * psi_new) / psi_old_sq;
            total_weighted_E += EL_new * W;
            total_W          += W;

            wf.setParameters(p_old);                          // restore for next step
        }

        wf.setParameters(original_params);
        return (total_W > 1e-18) ? (total_weighted_E / total_W)
                                 : std::numeric_limits<double>::max();
    }

    // Parabolic interpolation of three (x, y) points: returns the x of the
    // vertex of the fitted parabola, or the x of the lowest sample on failure.
    double parabolicInterpolation(const std::vector<double>& log_alphas,
                                  const std::vector<double>& energies) const
    {
        double x1 = log_alphas[0], y1 = energies[0];
        double x2 = log_alphas[1], y2 = energies[1];
        double x3 = log_alphas[2], y3 = energies[2];
        double denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
        if (std::abs(denom) < 1e-18) return x2;
        double a = (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) / denom;
        double b = (y1 * (x3*x3 - x2*x2) + y2 * (x1*x1 - x3*x3) + y3 * (x2*x2 - x1*x1)) / denom;
        if (a > 1e-18) return -b / (2.0 * a);
        return (y1 < y2 && y1 < y3) ? x1 : (y3 < y2 ? x3 : x2);
    }

public:
    LinearMethodOptimizer(int epochs, int samples,
                          double xi_ = 0.5, double adiag_ = 1e-3)
        : maxEpochs(epochs), samplesPerEpoch(samples),
          xi(xi_), adiag(adiag_) {}

    void optimize(WaveFunction& wf, Hamiltonian& ham, Metropolis& sampler) override {
        std::vector<double> currentParams = wf.getParameters();
        int nParams = (int)currentParams.size();

        currentPosition.resize(wf.getStride());
        std::mt19937 tempGen(1234);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& pos : currentPosition) pos = dist(tempGen);

        currentStepSize = sampler.getStepSize();
        if (currentStepSize < 0.1) currentStepSize = 1.0;
        sampler.setStepSize(currentStepSize);

        double psiTmp = wf.trialWaveFunction(currentPosition.data());
        adjustStepSize(wf, sampler, psiTmp);

        std::cout << "LINEAR METHOD (Umrigar 2007) - Adaptive adiag" << std::endl;

        for (int epoch = 0; epoch < maxEpochs; ++epoch) {
            // ---- Equilibrate at the current parameter set ----------------------
            double currentPsi = wf.trialWaveFunction(currentPosition.data());
            for (int i = 0; i < Constants::EQUILIBRATION_STEPS; ++i)
                sampler.step(wf, currentPosition, currentPsi);

            // ---- Accumulate MC averages ----------------------------------------
            double sumE = 0.0;
            std::vector<double> sumO  (nParams, 0.0);
            std::vector<double> sumdE (nParams, 0.0);
            std::vector<double> sumOE (nParams, 0.0);
            std::vector<std::vector<double>> sumOO (nParams, std::vector<double>(nParams, 0.0));
            std::vector<std::vector<double>> sumOdE(nParams, std::vector<double>(nParams, 0.0));
            std::vector<std::vector<double>> sumOOE(nParams, std::vector<double>(nParams, 0.0));

            int accepted = 0;
            for (int step = 0; step < samplesPerEpoch; ++step) {
                if (sampler.step(wf, currentPosition, currentPsi)) accepted++;

                double EL = ham.getLocalEnergy(wf, currentPosition.data());
                Derivatives d = computeDerivatives(wf, ham, currentPosition.data());

                sumE += EL;
                for (int i = 0; i < nParams; ++i) {
                    sumO [i] += d.O[i];
                    sumdE[i] += d.dE[i];
                    sumOE[i] += d.O[i] * EL;
                    for (int j = 0; j < nParams; ++j) {
                        sumOO [i][j] += d.O[i] * d.O[j];
                        sumOdE[i][j] += d.O[i] * d.dE[j];
                        sumOOE[i][j] += d.O[i] * d.O[j] * EL;
                    }
                }
            }

            double accRate = (double)accepted / samplesPerEpoch;
            if (accRate < 0.2 || accRate > 0.8) adjustStepSize(wf, sampler, currentPsi);

            double inv = 1.0 / samplesPerEpoch;
            double avgE = sumE * inv;

            std::vector<double> avgO(nParams), avgdE(nParams), avgOE(nParams);
            for (int i = 0; i < nParams; ++i) {
                avgO [i] = sumO [i] * inv;
                avgdE[i] = sumdE[i] * inv;
                avgOE[i] = sumOE[i] * inv;
            }
            std::vector<std::vector<double>> avgOO (nParams, std::vector<double>(nParams));
            std::vector<std::vector<double>> avgOdE(nParams, std::vector<double>(nParams));
            std::vector<std::vector<double>> avgOOE(nParams, std::vector<double>(nParams));
            for (int i = 0; i < nParams; ++i)
                for (int j = 0; j < nParams; ++j) {
                    avgOO [i][j] = sumOO [i][j] * inv;
                    avgOdE[i][j] = sumOdE[i][j] * inv;
                    avgOOE[i][j] = sumOOE[i][j] * inv;
                }

            // ---- Build (n+1)x(n+1) S and (asymmetric) H matrices ---------------
            int N = nParams + 1;
            std::vector<std::vector<double>> H(N, std::vector<double>(N, 0.0));
            std::vector<std::vector<double>> S(N, std::vector<double>(N, 0.0));

            S[0][0] = 1.0;
            for (int i = 0; i < nParams; ++i) {
                S[0][i + 1] = avgO[i];
                S[i + 1][0] = avgO[i];
                for (int j = 0; j < nParams; ++j)
                    S[i + 1][j + 1] = avgOO[i][j];
            }

            H[0][0] = avgE;
            for (int j = 0; j < nParams; ++j) {
                H[0][j + 1] = avgdE[j] + avgOE[j];   // <∂_j E_L> + <O_j E_L>
                H[j + 1][0] = avgOE[j];              // <O_i E_L>
            }
            for (int i = 0; i < nParams; ++i)
                for (int j = 0; j < nParams; ++j)
                    H[i + 1][j + 1] = avgOdE[i][j] + avgOOE[i][j];

            // ---- Adaptive adiag selection (Umrigar 2007 stabilization) ---------
            // Try three values of adiag differing by factors of 10, estimate the
            // resulting energies via correlated sampling on the stored MC configs,
            // and pick the optimum by parabolic interpolation in log-space.
            std::vector<double> candidates  = { adiag / 10.0, adiag, adiag * 10.0 };
            std::vector<double> log_alphas  = { std::log10(candidates[0]),
                                                std::log10(candidates[1]),
                                                std::log10(candidates[2]) };
            std::vector<double> candidate_Es;
            candidate_Es.reserve(3);

            const int correlatedSteps = std::max(100, samplesPerEpoch / 10);
            for (double alpha : candidates) {
                std::vector<std::vector<double>> H_c = H;
                for (int i = 1; i < N; ++i) H_c[i][i] += alpha;
                std::vector<double> dp  = solveGeneralizedEig(H_c, S, avgE - 0.1);
                std::vector<double> dpR = applyRescaling(dp, S);
                std::vector<double> p_new = currentParams;
                for (int i = 0; i < nParams; ++i) p_new[i] += dpR[i + 1];
                candidate_Es.push_back(
                    estimateEnergyCorrelated(wf, ham, sampler, p_new, currentParams,
                                             currentPosition, correlatedSteps));
            }

            double opt_log_alpha = parabolicInterpolation(log_alphas, candidate_Es);
            opt_log_alpha = std::max(-6.0, std::min(2.0, opt_log_alpha));
            adiag = std::pow(10.0, opt_log_alpha);

            // ---- Final solve with chosen adiag ---------------------------------
            std::vector<std::vector<double>> H_f = H;
            for (int i = 1; i < N; ++i) H_f[i][i] += adiag;
            std::vector<double> dp_f  = solveGeneralizedEig(H_f, S, avgE - 0.1);
            std::vector<double> dpR_f = applyRescaling(dp_f, S);

            // ---- Update parameters ---------------------------------------------
            for (int i = 0; i < nParams; ++i)
                currentParams[i] += dpR_f[i + 1];
            wf.setParameters(currentParams);

            std::cout << "Epoch " << (epoch + 1)
                      << " | E = "      << avgE
                      << " | adiag = "  << adiag
                      << " | accRate = " << accRate
                      << std::endl;
        }
    }
};

#endif