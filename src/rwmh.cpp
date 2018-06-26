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
 * Random Walk Metropolis-Hastings (RWMH) MCMC
 */

#include "mcmc.hpp" 

bool
mcmc::rwmh_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = initial_vals.n_elem;

    //
    // RWMH settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_draws_keep   = settings.rwmh_n_draws;
    const size_t n_draws_burnin = settings.rwmh_n_burnin;

    const double par_scale = settings.rwmh_par_scale;

    const arma::mat cov_mcmc = (settings.rwmh_cov_mat.n_elem == n_vals*n_vals) ? settings.rwmh_cov_mat : arma::eye(n_vals,n_vals);

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, void* box_data)> box_log_kernel \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, void* target_data) \
    -> double 
    {
        if (vals_bound)
        {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        }
        else
        {
            return target_log_kernel(vals_inp, target_data);
        }
    };

    //
    // setup
    
    arma::vec first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    draws_out.set_size(n_draws_keep, n_vals);

    double prev_LP = box_log_kernel(first_draw, target_data);
    double prop_LP = prev_LP;
    
    arma::vec prev_draw = first_draw;
    arma::vec new_draw  = first_draw;
    
    arma::mat cov_mcmc_sc   = par_scale * par_scale * cov_mcmc;
    arma::mat cov_mcmc_chol = arma::chol(cov_mcmc_sc,"lower");

    //

    int n_accept = 0;
    arma::vec krand(n_vals);
    
    for (size_t jj = 0; jj < n_draws_keep + n_draws_burnin; jj++)
    {

        new_draw = prev_draw + cov_mcmc_chol * krand.randn();
        
        prop_LP = box_log_kernel(new_draw, target_data);
        
        if (!std::isfinite(prop_LP)) {
            prop_LP = neginf;
        }

        //

        double comp_val = std::min(0.0,prop_LP - prev_LP);
        double z = arma::as_scalar(arma::randu(1));

        if (z < std::exp(comp_val))
        {
            prev_draw = new_draw;
            prev_LP = prop_LP;

            if (jj >= n_draws_burnin)
            {
                draws_out.row(jj - n_draws_burnin) = new_draw.t();
                n_accept++;
            }
        }
        else
        {
            if (jj >= n_draws_burnin) {
                draws_out.row(jj - n_draws_burnin) = prev_draw.t();
            }
        }
    }

    success = true;

    //

    if (vals_bound) {
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t jj = 0; jj < n_draws_keep; jj++) {
            draws_out.row(jj) = arma::trans(inv_transform(draws_out.row(jj).t(), bounds_type, lower_bounds, upper_bounds));
        }
    }

    if (settings_inp) {
        settings_inp->rwmh_accept_rate = static_cast<double>(n_accept) / static_cast<double>(n_draws_keep);
    }

    //

    return success;
}

// wrappers

bool
mcmc::rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data)
{
    return rwmh_int(initial_vals,draws_out,target_log_kernel,target_data,nullptr);
}

bool
mcmc::rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings)
{
    return rwmh_int(initial_vals,draws_out,target_log_kernel,target_data,&settings);
}
