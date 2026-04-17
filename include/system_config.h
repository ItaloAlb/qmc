#ifndef SYSTEM_CONFIG_H
#define SYSTEM_CONFIG_H

#include "config.h"
#include "constants.h"

// Read an optional "pbc" block from the params JSON and build a
// PeriodicBoundary. Returns nullptr if no "pbc" key is present.
//
// Expected JSON shape (2D example):
//     "pbc": {
//         "a1": [L, 0],
//         "a2": [0, L]
//     }
// Vectors must be in the same units the rest of the system uses (Bohr
// internally — lengths are NOT converted from Angstrom here).
inline std::unique_ptr<PeriodicBoundary> buildPBC(const json& p) {
    if (!p.contains("pbc")) return nullptr;

    const json& pbcJson = p.at("pbc");
    std::vector<std::vector<double>> lattice;

    if (pbcJson.contains("a1")) lattice.push_back(pbcJson.at("a1").get<std::vector<double>>());
    if (pbcJson.contains("a2")) lattice.push_back(pbcJson.at("a2").get<std::vector<double>>());
    if (pbcJson.contains("a3")) lattice.push_back(pbcJson.at("a3").get<std::vector<double>>());

    if (lattice.empty()) {
        throw std::runtime_error("pbc block is present but contains no a1/a2/a3");
    }

    // Config vectors are in Angstrom; convert to Bohr (internal units).
    for (auto& vec : lattice) {
        for (auto& component : vec) {
            component /= Constants::a0;
        }
    }

    return std::make_unique<PeriodicBoundary>(lattice);
}


System Helium(const json& p) {
    std::vector<double> masses  = p.at("masses").get<std::vector<double>>();
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> alpha   = p.at("wf_alpha").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    auto ham = std::make_unique<CoulombHamiltonian>(nP, nD, masses, charges);
    auto wf  = std::make_unique<HeliumWF>(alpha, nP, nD);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}

System MonolayerBiexciton(const json& p) {
    double X2D  = p.at("X2D");
    double rho0 = 2 * Constants::PI * (X2D / Constants::a0);

    std::vector<double> masses  = p.at("masses").get<std::vector<double>>();
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> alpha   = p.at("wf_alpha").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    auto ham = std::make_unique<EfficientRKHamiltonian>(nP, nD, masses, charges, rho0);
    auto wf  = std::make_unique<MonolayerBiexcitonWF>(alpha, nP, nD);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}

System TwistedHeterobilayerExciton(const json& p) {
    double me        = p.at("me");
    double mh        = p.at("mh");
    double thickness = p.at("thickness");
    double alpha_rk  = p.at("alpha");
    double eps       = p.at("eps");
    double eps1      = p.at("eps1");
    double eps2      = p.at("eps2");
    double theta     = p.at("theta");
    double eField    = p.at("eField");
    double a10       = p.at("a10");
    double a20       = p.at("a20");
    double Vh1       = p.at("Vh1");
    double Vh2       = p.at("Vh2");
    double Ve1       = p.at("Ve1");
    double Ve2       = p.at("Ve2");
    double d0        = p.at("d0");
    double d1        = p.at("d1");
    double d2        = p.at("d2");

    double rho0 = alpha_rk * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

    TwistedBilayerSystem moire(a10, a20, theta, eField,
                               Vh1, Vh2, Ve1, Ve2,
                               d0, d1, d2);

    bool interacting = p.value("interacting", true);

    std::vector<double> masses  = { me, mh };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> optP    = p.at("wf_params_init").get<std::vector<double>>();

    std::vector<double> initP(7, 0.0);
    if (interacting) {
        // c1 is fixed by the Kato cusp condition
        double c1 = (me * mh) / ((eps1 + eps2) * rho0 * (me + mh));
        initP[0] = c1;
        initP[1] = std::exp(optP[0]);  // c2 (config gives log-space)
        initP[2] = std::exp(optP[1]);  // c3 (config gives log-space)
        initP[3] = optP[2];            // Ve1_var
        initP[4] = optP[3];            // Ve2_var
        initP[5] = optP[4];            // Vh1_var
        initP[6] = optP[5];            // Vh2_var
    } else {
        // jastrow disabled: c1,c2,c3 are unused, config holds only the 4 moiré vars
        initP[3] = optP[0];            // Ve1_var
        initP[4] = optP[1];            // Ve2_var
        initP[5] = optP[2];            // Vh1_var
        initP[6] = optP[3];            // Vh2_var
    }

    int nP = 2, nD = 2;
    auto ham = std::make_unique<TwistedHeterobilayerHamiltonian>(
        nP, nD, masses, charges, moire, rho0, thickness, eps1, eps2, interacting);
    auto wf  = std::make_unique<TwistedBilayerExcitonWF>(
        initP, nP, nD, moire, thickness, interacting);
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}

System ExcitonInASquarePotential(const json& p) {
    double me        = p.at("me");
    double mh        = p.at("mh");
    double thickness = p.at("thickness");
    double alpha_rk  = p.at("alpha");
    double eps       = p.at("eps");
    double eps1      = p.at("eps1");
    double eps2      = p.at("eps2");
    double V0        = p.at("V0");
    double side      = p.at("side");

    double rho0 = alpha_rk * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

    bool interacting = p.value("interacting", true);

    std::vector<double> masses  = { me, mh };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> optP    = p.at("wf_params_init").get<std::vector<double>>();

    std::vector<double> initP(5, 0.0);
    if (interacting) {
        // c1 is fixed by the Kato cusp condition
        double c1 = (me * mh) / ((eps1 + eps2) * rho0 * (me + mh));
        initP[0] = c1;
        initP[1] = std::exp(optP[0]);  // c2 (config gives log-space)
        initP[2] = std::exp(optP[1]);  // c3 (config gives log-space)
        initP[3] = optP[2];            // lambda_e
        initP[4] = optP[3];            // lambda_h
    } else {
        initP[3] = optP[0];            // lambda_e
        initP[4] = optP[1];            // lambda_h
    }

    int nP = 2, nD = 2;
    auto ham = std::make_unique<SquareHamiltonian>(
        nP, nD, masses, charges, V0, side, eps1, eps2, rho0, thickness, interacting);
    auto wf  = std::make_unique<ExcitonInASquarePotentialWF>(
        initP, nP, nD, side, thickness, interacting);
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}

System ExcitonInATrianglePotential(const json& p) {
    double me        = p.at("me");
    double mh        = p.at("mh");
    double thickness = p.at("thickness");
    double alpha_rk  = p.at("alpha");
    double eps       = p.at("eps");
    double eps1      = p.at("eps1");
    double eps2      = p.at("eps2");
    double V0        = p.at("V0");
    double side      = p.at("side");

    double rho0 = alpha_rk * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;

    bool interacting = p.value("interacting", true);

    std::vector<double> masses  = { me, mh };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> optP    = p.at("wf_params_init").get<std::vector<double>>();

    std::vector<double> initP(5, 0.0);
    if (interacting) {
        double c1 = (me * mh) / ((eps1 + eps2) * rho0 * (me + mh));
        initP[0] = c1;
        initP[1] = std::exp(optP[0]);  // c2 (config gives log-space)
        initP[2] = std::exp(optP[1]);  // c3 (config gives log-space)
        initP[3] = optP[2];            // lambda_e
        initP[4] = optP[3];            // lambda_h
    } else {
        initP[3] = optP[0];            // lambda_e
        initP[4] = optP[1];            // lambda_h
    }

    int nP = 2, nD = 2;
    auto ham = std::make_unique<TriangleHamiltonian>(
        nP, nD, masses, charges, V0, side, eps1, eps2, rho0, thickness, interacting);
    auto wf  = std::make_unique<ExcitonInATrianglePotentialWF>(
        initP, nP, nD, side, thickness, interacting);
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}

System ExcitonExciton(const json& p) {
    double me = p.at("me"), mh = p.at("mh");
    double mu = (me * mh) / (me + mh);
    double d  = p.value("d", 0.2 ) * me / mu;
    double R  = p.value("R", 1.0) * me / mu;

    bool interacting = p.value("interacting", true);

    std::vector<double> masses  = { mu, mu };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    int Nee = p.value("Nee", 0);
    int Nhh = p.value("Nhh", 0);
    int Neh = p.value("Neh", 0);

    int eeCount = (Nee > 0) ? 1 + Nee : 0;
    int hhCount = (Nhh > 0) ? 1 + Nhh : 0;
    int ehCount = (Neh > 0) ? 1 + Neh : 0;
    std::vector<double> initP(9 + eeCount + hhCount + ehCount, 0.0);

    auto ham = std::make_unique<ExcitonExcitonCoulombHamiltonian>(
        nP, nD, masses, charges, me, mh, d, R, interacting);
    auto wf  = std::make_unique<ExcitonExcitonWF>(
        initP, nP, nD, me, mh, d, R, interacting, Nee, Nhh, Neh);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), buildPBC(p), nP, nD };
}


System buildSystem(const QMCConfig& cfg) {
    const std::string& name = cfg.systemName;
    const json& p           = cfg.params;

    if      (name == "helium")              return Helium(p);
    else if (name == "monolayer_biexciton") return MonolayerBiexciton(p);
    else if (name == "exciton_exciton")     return ExcitonExciton(p);
    else if (name == "twisted_heterobilayer_exciton")   return TwistedHeterobilayerExciton(p);
    else if (name == "exciton_in_a_square_potential")   return ExcitonInASquarePotential(p);
    else if (name == "exciton_in_a_triangle_potential") return ExcitonInATrianglePotential(p);
    // ...
    else throw std::runtime_error("Unknown system: " + name);
}

#endif