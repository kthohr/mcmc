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

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ aees_mixture.cpp -o aees_mixture.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"

struct mixture_data_t { 
    Eigen::MatrixXd mu;
    Eigen::VectorXd sig_sq;
    Eigen::VectorXd weights;
};
 
double
gaussian_mixture(const Eigen::VectorXd& X_vec_inp, const Eigen::VectorXd& weights, const Eigen::MatrixXd& mu, const Eigen::VectorXd& sig_sq)
{
    const double pi = 3.14159265358979;
    
    const int n_vals = X_vec_inp.size();
    const int n_mix = weights.size();
    
    //

    double dens_val = 0;
    
    for (int i = 0; i < n_mix; ++i) {
        const double dist_val = (X_vec_inp - mu.col(i)).array().pow(2).sum();
        
        dens_val += weights(i) * std::exp(-0.5 * dist_val / sig_sq(i)) / std::pow(2.0 * pi * sig_sq(i), static_cast<double>(n_vals) / 2.0);
    }

    //
    
    return std::log(dens_val);
}

double
target_log_kernel(const Eigen::VectorXd& vals_inp, void* target_data)
{
    mixture_data_t* dta = reinterpret_cast<mixture_data_t*>(target_data);

    return gaussian_mixture(vals_inp, dta->weights, dta->mu, dta->sig_sq);
}

int main()
{
    const int n_vals = 2;
    const int n_mix  = 2;

    //

    Eigen::MatrixXd mu = Eigen::MatrixXd::Ones(n_vals, n_mix).array() + 1.0;
    mu.col(0) *= -1.0; // (-2, 2)

    Eigen::VectorXd weights = Eigen::VectorXd::Constant(n_mix, 1.0 / n_mix);

    Eigen::VectorXd sig_sq = 0.1 * Eigen::VectorXd::Ones(n_mix);

    mixture_data_t dta;
    dta.mu = mu;
    dta.sig_sq = sig_sq;
    dta.weights = weights;

    //

    Eigen::VectorXd T_vec(2);
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
    settings.aees_settings.cov_mat = 0.35 * Eigen::MatrixXd::Identity(n_vals, n_vals);

    //

    Eigen::MatrixXd draws_out;

    mcmc::aees(mu.col(0), target_log_kernel, draws_out, &dta, settings);

    //

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> pos_inds = (draws_out.array() > 0.1);

    Eigen::VectorXd mean_vec = Eigen::VectorXd::Zero(2);

    for (int i = 0; i < n_vals; ++i) {
        for (size_t draw_ind = 0; draw_ind < settings.aees_settings.n_keep_draws; ++draw_ind) {
            if (pos_inds(draw_ind, i)) {
                mean_vec(i) += draws_out(draw_ind, i);
            }
        }
        mean_vec(i) /= pos_inds.col(i).count();
    }
    
    std::cout << "posterior mean for > 0.1:\n" << mean_vec << std::endl;

    //

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> neg_inds = (draws_out.array() < - 0.1);

    mean_vec = Eigen::VectorXd::Zero(2);

    for (int i = 0; i < n_vals; ++i) {
        for (size_t draw_ind = 0; draw_ind < settings.aees_settings.n_keep_draws; ++draw_ind) {
            if (neg_inds(draw_ind, i)) {
                mean_vec(i) += draws_out(draw_ind, i);
            }
        }
        mean_vec(i) /= neg_inds.col(i).count();
    }
    
    std::cout << "posterior mean for < - 0.1:\n" << mean_vec << std::endl;

    //

    return 0;
}
