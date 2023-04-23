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
 * Sampling from a Gaussian distribution using DE-MCMC
 */

// $CXX -Wall -std=c++11 -O3 -mcpu=native -ffp-contract=fast -I$ARMA_INCLUDE_PATH -I./../../include/ de_normal_mean.cpp -o de_normal_mean.out -L./../.. -lmcmc

#define MCMC_ENABLE_ARMA_WRAPPERS
#include "mcmc.hpp"

struct norm_data_t {
    double sigma;
    arma::vec x;
 
    double mu_0;
    double sigma_0;
};
 
double ll_dens(const arma::vec& vals_inp, void* ll_data)
{
    const double pi = arma::datum::pi;

    //

    const double mu = vals_inp(0);
 
    norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
    const double sigma = dta->sigma;
    const arma::vec x = dta->x;
 
    const int n_vals = x.n_rows;
 
    //
 
    const double ret = - ((double) n_vals) * (0.5*std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );
 
    //
 
    return ret;
}
 
double log_pr_dens(const arma::vec& vals_inp, void* ll_data)
{
    const double pi = arma::datum::pi;

    //

    norm_data_t* dta = reinterpret_cast< norm_data_t* >(ll_data);
 
    const double mu_0 = dta->mu_0;
    const double sigma_0 = dta->sigma_0;
 
    const double x = vals_inp(0);
 
    const double ret = - 0.5*std::log(2*pi) - std::log(sigma_0) - std::pow(x - mu_0,2) / (2*sigma_0*sigma_0);
 
    return ret;
}
 
double log_target_dens(const arma::vec& vals_inp, void* ll_data)
{
    return ll_dens(vals_inp,ll_data) + log_pr_dens(vals_inp,ll_data);
}

int main()
{
    const int n_data = 100;
    const double mu = 2.0;
 
    norm_data_t dta;
    dta.sigma = 1.0;
    dta.mu_0 = 1.0;
    dta.sigma_0 = 2.0;
 
    arma::vec x_dta = mu + arma::randn(n_data,1);
    dta.x = x_dta;
 
    arma::vec initial_val(1);
    initial_val(0) = 1.0;

    //

    mcmc::algo_settings_t settings;

    settings.de_settings.n_burnin_draws = 2000;
    settings.de_settings.n_keep_draws = 2000;

    //

    mcmc::Cube_t draws_out;
    mcmc::de(initial_val, log_target_dens, draws_out, &dta, settings);

    //
  
    std::cout << "de mean:\n" << arma::mean(draws_out.mat(settings.de_settings.n_keep_draws - 1)) << std::endl;
    std::cout << "acceptance rate: " << static_cast<double>(settings.de_settings.n_accept_draws) / (settings.de_settings.n_keep_draws * settings.de_settings.n_pop) << std::endl;
    
    //
 
    return 0;
}
