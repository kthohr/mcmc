/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
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
 * AEES with mixture model
 */

#include "mcmc.hpp"

double
gaussian_mixture(const arma::vec& X_vec_inp, const arma::vec& weights, const arma::mat& mu, const arma::vec& sig_sq)
{
    const double pi = arma::datum::pi;
    
    const int n_vals = X_vec_inp.n_elem;
    const int n_mix = weights.n_elem;
    
    //

    double dens_val = 0;
    
    for (int i=0; i < n_mix; i++) {
        double dist_val = arma::accu(arma::pow(X_vec_inp - mu.col(i),2));
        
        dens_val += weights(i) * std::exp(-0.5*dist_val/sig_sq(i)) / std::pow(2.0*pi*sig_sq(i), static_cast<double>(n_vals)/2.0);
    }

    //
    
    return log(dens_val);
}

struct mixture_data { 
    arma::mat mu;
    arma::vec sig_sq;
    arma::vec weights;
};

double
target_log_kernel(const arma::vec& vals_inp, void* target_data)
{
    mixture_data* dta = reinterpret_cast<mixture_data*>(target_data);

    return gaussian_mixture(vals_inp, dta->weights, dta->mu, dta->sig_sq);
}

int main()
{
    const int n_vals = 2;

    arma::vec T_vec(2);
    T_vec(0) = 60.0;
    T_vec(1) = 9.0;

    const int n_mix = 2;

    //

    arma::mat mu = arma::ones(n_vals,n_mix) + 1.0;
    mu.col(0) *= -1.0;

    arma::vec weights(n_mix); 
    weights(0) = 0.5;
    weights(1) = 0.5;

    arma::vec sig_sq = arma::ones(n_mix,1)*0.1;

    mixture_data dta;
    dta.mu = mu;
    dta.sig_sq = sig_sq;
    dta.weights = weights;

    //

    mcmc::algo_settings_t settings;

    settings.aees_n_draws = 20000;
    settings.aees_n_burnin = 1000;
    settings.aees_n_initial_draws = 1000;

    settings.aees_prob_par = 0.05;
    settings.aees_temper_vec = T_vec;
    settings.aees_n_rings = 11;

    settings.rwmh_par_scale = 1.0;
    settings.rwmh_cov_mat = 0.35*arma::eye(n_vals,n_vals);

    //

    arma::mat draws_out;

    mcmc::aees(mu.col(0),draws_out,target_log_kernel,&dta,settings);

    // draws_out.save("aees_res.txt",arma::raw_ascii);

    //

    return 0;
}
