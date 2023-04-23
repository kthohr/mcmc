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
 * Sampling from a Gaussian distribution using RM-HMC
 */

// $CXX -Wall -std=c++11 -O3 -mcpu=native -ffp-contract=fast -I$ARMA_INCLUDE_PATH -I./../../include/ rmhmc_normal.cpp -o rmhmc_normal.out -L./../.. -lmcmc

#define MCMC_ENABLE_ARMA_WRAPPERS
#include "mcmc.hpp"

struct norm_data_t {
    arma::vec x;
};

double ll_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
{
    const double pi = arma::datum::pi;
    
    const double mu    = vals_inp(0);
    const double sigma = vals_inp(1);
  
    norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
    const arma::vec x = dta->x;
    const int n_vals = x.n_rows;
  
    //
  
    const double ret = - n_vals * (0.5 * std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );
  
    //

    if (grad_out) {
        grad_out->set_size(2,1);
  
        //
  
        const double m_1 = arma::accu(x - mu);
        const double m_2 = arma::accu( arma::pow(x - mu,2) );
  
        (*grad_out)(0,0) = m_1 / (sigma*sigma);
        (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - ((double) n_vals) / sigma;
    }
  
    //
  
    return ret;
}

arma::mat tensor_fn(const arma::vec& vals_inp, mcmc::Cube_t* tensor_deriv_out, void* tensor_data)
{
    // const double mu    = vals_inp(0);
    const double sigma = vals_inp(1);
  
    norm_data_t* dta = reinterpret_cast<norm_data_t*>(tensor_data);
     
    const int n_vals = dta->x.n_rows;
  
    //
 
    const double sigma_sq = sigma*sigma;
  
    arma::mat tensor_out = arma::zeros(2,2);
 
    tensor_out(0,0) = ((double) n_vals) / sigma_sq;
    tensor_out(1,1) = 2.0 * ((double) n_vals) / sigma_sq;
     
    //
  
    if (tensor_deriv_out) {
        tensor_deriv_out->setZero(2,2,2);
  
        //
  
        // tensor_deriv_out->mat(0).setZero();
  
        tensor_deriv_out->mat(1) = - 2.0 * tensor_out / sigma;
    }
  
    //
  
    return tensor_out;
}

double log_target_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
{
    return ll_dens(vals_inp,grad_out,ll_data);
}

int main()
{
    const int n_data = 1000;

    const double mu = 2.0;
    const double sigma = 2.0;
  
    norm_data_t dta;
  
    arma::vec x_dta = mu + sigma * arma::randn(n_data,1);
    dta.x = x_dta;
  
    arma::vec initial_val(2);
    initial_val(0) = mu + 1; // mu
    initial_val(1) = sigma + 1; // sigma

    //
  
    mcmc::algo_settings_t settings;
  
    settings.rmhmc_settings.step_size = 0.2;
    settings.rmhmc_settings.n_burnin_draws = 2000;
    settings.rmhmc_settings.n_keep_draws = 2000;

    //
  
    arma::mat draws_out;
    mcmc::rmhmc(initial_val, log_target_dens, tensor_fn, draws_out, &dta, &dta, settings);

    //
  
    std::cout << "rmhmc mean:\n" << arma::mean(draws_out) << std::endl;
    std::cout << "acceptance rate: " << static_cast<double>(settings.rmhmc_settings.n_accept_draws) / settings.rmhmc_settings.n_keep_draws << std::endl;

    //
 
    return 0;
}
