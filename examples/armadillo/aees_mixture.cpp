/*################################################################################
  ##
  ##   Copyright (C) 2011-2023 Keith O'Hara
  ##
  ##   This file is part of the MCMC C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/
 
/*
 * Sampling from a Gaussian Mixture Distribution using the Adaptive Equi-Energy Sampler (AEES)
 */

// $CXX -Wall -std=c++11 -O3 -mcpu=native -ffp-contract=fast -I$ARMA_INCLUDE_PATH -I./../../include/ aees_mixture.cpp -o aees_mixture.out -L./../.. -lmcmc

#define MCMC_ENABLE_ARMA_WRAPPERS
#include "mcmc.hpp"

struct mixture_data_t { 
    arma::mat mu;
    arma::vec sig_sq;
    arma::vec weights;
};
 
double
gaussian_mixture(const arma::vec& X_vec_inp, const arma::vec& weights, const arma::mat& mu, const arma::vec& sig_sq)
{
    const double pi = arma::datum::pi;
    
    const int n_vals = X_vec_inp.n_elem;
    const int n_mix = weights.n_elem;
    
    //

    double dens_val = 0;
    
    for (int i = 0; i < n_mix; ++i) {
        const double dist_val = arma::accu(arma::pow(X_vec_inp - mu.col(i), 2));
        
        dens_val += weights(i) * std::exp(-0.5 * dist_val / sig_sq(i)) / std::pow(2.0 * pi * sig_sq(i), static_cast<double>(n_vals) / 2.0);
    }

    //
    
    return std::log(dens_val);
}

double
target_log_kernel(const arma::vec& vals_inp, void* target_data)
{
    mixture_data_t* dta = reinterpret_cast<mixture_data_t*>(target_data);

    return gaussian_mixture(vals_inp, dta->weights, dta->mu, dta->sig_sq);
}

int main()
{
    const int n_vals = 2;
    const int n_mix  = 2;

    //

    arma::mat mu = arma::ones(n_vals, n_mix) + 1.0;
    mu.col(0) *= -1.0; // (-2, 2)

    arma::vec weights(n_mix, arma::fill::value(1.0 / n_mix));

    arma::vec sig_sq = 0.1 * arma::ones(n_mix);

    mixture_data_t dta;
    dta.mu = mu;
    dta.sig_sq = sig_sq;
    dta.weights = weights;

    //

    arma::vec T_vec(2);
    T_vec(0) = 60.0;
    T_vec(1) = 9.0;

    // settings

    mcmc::algo_settings_t settings;

    settings.aees_settings.n_initial_draws = 1000;
    settings.aees_settings.n_burnin_draws  = 1000;
    settings.aees_settings.n_keep_draws    = 20000;

    settings.aees_settings.n_rings = 11;
    settings.aees_settings.ee_prob_par = 0.05;
    settings.aees_settings.temper_vec = T_vec;

    settings.aees_settings.par_scale = 1.0;
    settings.aees_settings.cov_mat = 0.35 * arma::eye(n_vals, n_vals);

    //

    arma::mat draws_out;

    mcmc::aees(mu.col(0), target_log_kernel, draws_out, &dta, settings);

    arma::cout << "posterior mean for > 0.1:\n" << arma::mean(draws_out.elem( arma::find(draws_out > 0.1) ), 0) << arma::endl;
    arma::cout << "posterior mean for < -0.1:\n" << arma::mean(draws_out.elem( arma::find(draws_out < -0.1) ), 0) << arma::endl;

    //

    return 0;
}
