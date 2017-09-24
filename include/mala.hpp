/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
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

bool mala_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, mcmc_settings* settings_inp);

bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data);
bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, mcmc_settings& settings);

//

inline
double
mala_prop_adjustment(const arma::vec& prop_vals, const arma::vec& prev_vals, const double step_size, const bool vals_bound, std::function<arma::vec (const arma::vec& vals_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out)> mala_mean_fn, void* target_data)
{
    const int n_vals = prop_vals.n_elem;

    double ret_val = 0;

    //

    if (vals_bound) {

        arma::mat prop_inv_jacob, prev_inv_jacob;

        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, &prop_inv_jacob);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, &prev_inv_jacob);

        for (int i=0; i < n_vals; i++) {
            ret_val += stats_mcmc::dnorm(prev_vals(i),prop_mean(i),step_size * std::sqrt(prop_inv_jacob(i,i)),true) - stats_mcmc::dnorm(prop_vals(i),prev_mean(i),step_size * std::sqrt(prev_inv_jacob(i,i)),true);
        }

    } else {
        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, nullptr);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, nullptr);

        for (int i=0; i < n_vals; i++) {
            ret_val += stats_mcmc::dnorm(prev_vals(i),prop_mean(i),step_size,true) - stats_mcmc::dnorm(prop_vals(i),prev_mean(i),step_size,true);
        }
    }

    //

    return ret_val;
}

#endif
