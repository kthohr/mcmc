/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
  ##
  ##   This file is part of the MCMC C++ library.
  ##
  ##   MCMC is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   MCMC is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/
 
/*
 * Metropolis-adjusted Langevin algorithm
 */

#ifndef _mcmc_mala_HPP
#define _mcmc_mala_HPP

bool mala_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings* settings_inp);

bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data);
bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings& settings);

//

inline
double
mala_prop_adjustment(const arma::vec& prop_vals, const arma::vec& prev_vals, const double step_size, const bool vals_bound, const arma::mat& precond_mat, std::function<arma::vec (const arma::vec& vals_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out)> mala_mean_fn, void* target_data)
{

    double ret_val = 0;
    const double step_size_sq = step_size*step_size;

    //

    if (vals_bound) {

        arma::mat prop_inv_jacob, prev_inv_jacob;

        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, &prop_inv_jacob);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, &prev_inv_jacob);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*prop_inv_jacob*precond_mat, true) - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*prop_inv_jacob*precond_mat, true);

    } else {
        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, nullptr);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, nullptr);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*precond_mat, true) - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*precond_mat, true);
    }

    //

    return ret_val;
}

#endif
