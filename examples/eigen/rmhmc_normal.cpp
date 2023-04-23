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
 * Sampling from a Gaussian distribution using HMC
 */

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ rmhmc_normal.cpp -o rmhmc_normal.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"

inline
Eigen::VectorXd
eigen_randn_colvec(size_t nr)
{
    static std::mt19937 gen{ std::random_device{}() };
    static std::normal_distribution<> dist;

    return Eigen::VectorXd{ nr }.unaryExpr([&](double x) { (void)(x); return dist(gen); });
}

struct norm_data_t {
    Eigen::VectorXd x;
};
  
double ll_dens(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* ll_data)
{
    const double pi = 3.14159265358979;

    const double mu    = vals_inp(0);
    const double sigma = vals_inp(1);
  
    norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
    const Eigen::VectorXd x = dta->x;
    const int n_vals = x.size();
  
    //
  
    const double ret = - n_vals * (0.5 * std::log(2*pi) + std::log(sigma)) - (x.array() - mu).pow(2).sum() / (2*sigma*sigma);
  
    //

    if (grad_out) {
        grad_out->resize(2,1);
  
        //
  
        const double m_1 = (x.array() - mu).sum();
        const double m_2 = (x.array() - mu).pow(2).sum();
  
        (*grad_out)(0,0) = m_1 / (sigma*sigma);
        (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - ((double) n_vals) / sigma;
    }
  
    //
  
    return ret;
}

Eigen::MatrixXd tensor_fn(const Eigen::VectorXd& vals_inp, mcmc::Cube_t* tensor_deriv_out, void* tensor_data)
{
    // const double mu    = vals_inp(0);
    const double sigma = vals_inp(1);
  
    norm_data_t* dta = reinterpret_cast<norm_data_t*>(tensor_data);
     
    const int n_vals = dta->x.size();
  
    //
 
    const double sigma_sq = sigma*sigma;
  
    Eigen::MatrixXd tensor_out = Eigen::MatrixXd::Zero(2,2);
 
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

double log_target_dens(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* ll_data)
{
    return ll_dens(vals_inp,grad_out,ll_data);
}

int main()
{
    const int n_data = 1000;

    const double mu = 2.0;
    const double sigma = 2.0;
  
    norm_data_t dta;
  
    Eigen::VectorXd x_dta = mu + sigma * eigen_randn_colvec(n_data).array();
    dta.x = x_dta;
  
    Eigen::VectorXd initial_val(2);
    initial_val(0) = mu + 1; // mu
    initial_val(1) = sigma + 1; // sigma
  
    mcmc::algo_settings_t settings;
  
    settings.rmhmc_settings.step_size = 0.2;
    settings.rmhmc_settings.n_burnin_draws = 2000;
    settings.rmhmc_settings.n_keep_draws = 2000;

    //
  
    Eigen::MatrixXd draws_out;
    mcmc::rmhmc(initial_val, log_target_dens, tensor_fn, draws_out, &dta, &dta, settings);

    //
  
    std::cout << "rmhmc mean:\n" << draws_out.colwise().mean() << std::endl;
    std::cout << "acceptance rate: " << static_cast<double>(settings.rmhmc_settings.n_accept_draws) / settings.rmhmc_settings.n_keep_draws << std::endl;

    //
 
    return 0;
}
