#ifndef SYSTEM_CONFIG_H
#define SYSTEM_CONFIG_H

#include "config.h"
#include "constants.h"


System Helium(const json& p) {
    std::vector<double> masses  = p.at("masses").get<std::vector<double>>();
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> alpha   = p.at("wf_alpha").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    auto ham = std::make_unique<CoulombHamiltonian>(nP, nD, masses, charges);
    auto wf  = std::make_unique<HeliumWF>(alpha, nP, nD);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), nP, nD };
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
    return { std::move(ham), std::move(wf), nP, nD };
}

System MoireExciton(const json& p) {
    double thickness = p.at("thickness");
    double alpha_rk  = p.at("alpha");
    double eps       = p.at("eps");
    double eps1      = p.at("eps1");
    double eps2      = p.at("eps2");
    double theta     = p.at("theta");
    double eField    = p.at("eField");
    double me        = p.at("me");
    double mh        = p.at("mh");

    double rho0 = alpha_rk * 2 * thickness * eps / (eps1 + eps2) / Constants::a0;
    TwistedBilayerSystem moire(theta, eField, thickness);

    std::vector<double> masses  = { me, mh };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> initP   = p.at("wf_params_init").get<std::vector<double>>();

    double varX1 = p.at("variance_params")[0];
    double varX2 = p.at("variance_params")[1];
    double varX3 = p.at("variance_params")[2];
    double varX4 = p.at("variance_params")[3];

    int nP = 2, nD = 2;
    auto ham = std::make_unique<TwistedHeterobilayerHamiltonian>(
        nP, nD, masses, charges, moire, rho0, eps1, eps2);
    auto wf  = std::make_unique<TwistedBilayerExcitonGaussianWF>(
        initP, nP, nD, varX1, varX2, varX3, varX4);
    return { std::move(ham), std::move(wf), nP, nD };
}

System ExcitonExciton(const json& p) {
    double me = p.at("me"), mh = p.at("mh");
    double mu = (me * mh) / (me + mh);
    double d  = p.value("d", 0.2 * me / mu);
    double R  = p.value("R", 4.0 * me / mu);

    std::vector<double> masses  = { mu, mu };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> alpha   = p.at("wf_alpha").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    auto ham = std::make_unique<ExcitonExcitonCoulombHamiltonian>(
        nP, nD, masses, charges, me, mh, d, R);
    auto wf  = std::make_unique<ExcitonExcitonWF>(
        alpha, nP, nD, me, mh, d, R);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), nP, nD };
}

System ExcitonExcitonNonInteract(const json& p) {
    double me = p.at("me"), mh = p.at("mh");
    double mu = (me * mh) / (me + mh);
    double d  = p.value("d", 0.2 * me / mu);

    std::vector<double> masses  = { mu, mu };
    std::vector<double> charges = p.at("charges").get<std::vector<double>>();
    std::vector<double> alpha   = p.at("wf_alpha").get<std::vector<double>>();
    int nP = p.at("nParticles"), nD = p.at("nDim");

    auto ham = std::make_unique<ExcitonExcitonNonInteractHamiltonian>(
        nP, nD, masses, charges, me, mh, d);
    auto wf  = std::make_unique<ExcitonExcitonNonInteractWF>(
        alpha, nP, nD, me, mh, d);
    wf->setParameters(p.at("wf_params_init").get<std::vector<double>>());
    return { std::move(ham), std::move(wf), nP, nD };
}


System buildSystem(const QMCConfig& cfg) {
    const std::string& name = cfg.systemName;
    const json& p           = cfg.params;

    if      (name == "helium")              return Helium(p);
    else if (name == "monolayer_biexciton") return MonolayerBiexciton(p);
    else if (name == "moire_exciton")       return MoireExciton(p);
    else if (name == "exciton_exciton")     return ExcitonExciton(p);
    else if (name == "exciton_exciton_non_interact")    return ExcitonExcitonNonInteract(p);
    // ...
    else throw std::runtime_error("Unknown system: " + name);
}

#endif