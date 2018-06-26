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
 * Metropolis-adjusted Langevin algorithm
 */

#ifndef _mcmc_mala_HPP
#define _mcmc_mala_HPP

bool mala_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp);

bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data);
bool mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings);

//

inline
double
mala_prop_adjustment(const arma::vec& prop_vals, const arma::vec& prev_vals, const double step_size, const bool vals_bound, const arma::mat& precond_mat, 
                     std::function<arma::vec (const arma::vec& vals_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out)> mala_mean_fn, void* target_data)
{

    double ret_val = 0;
    const double step_size_sq = step_size*step_size;

    //

    if (vals_bound) 
    {
        arma::mat prop_inv_jacob, prev_inv_jacob;

        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, &prop_inv_jacob);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, &prev_inv_jacob);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*prop_inv_jacob*precond_mat, true) \
                  - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*prop_inv_jacob*precond_mat, true);
    }
    else
    {
        arma::vec prop_mean = mala_mean_fn(prop_vals, target_data, step_size, nullptr);
        arma::vec prev_mean = mala_mean_fn(prev_vals, target_data, step_size, nullptr);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*precond_mat, true) \
                  - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*precond_mat, true);
    }

    //

    return ret_val;
}

#endif
