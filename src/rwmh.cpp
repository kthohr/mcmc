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
 * Random Walk Metropolis-Hastings (RWMH) MCMC
 *
 * Keith O'Hara
 * 05/01/2012
 *
 * This version:
 * 08/12/2017
 */

#include "mcmc.hpp" 

bool
mcmc::rwmh_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, mcmc_settings* settings_inp)
{
    bool success = false;

    const double BIG_NEG_VAL = MCMC_BIG_NEG_VAL;
    const int n_vals = initial_vals.n_elem;

    //
    // RWMH settings

    mcmc_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int n_draws_keep   = settings.rwmh_n_draws_keep;
    const int n_draws_burnin = settings.rwmh_n_draws_burnin;

    const double par_scale = settings.rwmh_par_scale;

    const arma::mat cov_mcmc = (settings.rwmh_cov_mat.n_elem == n_vals*n_vals) ? settings.rwmh_cov_mat : arma::eye(n_vals,n_vals);

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, void* box_data)> box_log_kernel = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, void* target_data) -> double {
        //
        if (vals_bound) {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        } else {
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
    arma::mat cov_mcmc_chol = arma::trans(arma::chol(cov_mcmc_sc));
    //
    int n_accept = 0;    
    double comp_val, rand_val;
    arma::vec krand(n_vals);
    
    for (int jj = 0; jj < n_draws_keep + n_draws_burnin; jj++) {

        new_draw = prev_draw + cov_mcmc_chol * krand.randn();
        
        prop_LP = box_log_kernel(new_draw, target_data);
        
        if (!std::isfinite(prop_LP)) {
            prop_LP = BIG_NEG_VAL;
        }
        //
        comp_val = prop_LP - prev_LP;
        
        if (comp_val > 0.0) { // the '> exp(0)' case; works around taking exp of big values and receiving an error
            prev_draw = new_draw;
            prev_LP = prop_LP;

            if (jj >= n_draws_burnin) {
                draws_out.row(jj - n_draws_burnin) = new_draw.t();
                n_accept++;
            }
        } else {
            rand_val = arma::as_scalar(arma::randu(1));

            if (rand_val < std::exp(comp_val)) {
                prev_draw = new_draw;
                prev_LP = prop_LP;

                if (jj >= n_draws_burnin) {
                    draws_out.row(jj - n_draws_burnin) = new_draw.t();
                    n_accept++;
                }
            } else {
                if (jj >= n_draws_burnin) {
                    draws_out.row(jj - n_draws_burnin) = prev_draw.t();
                }
            }
        }
    }
    //
    if (vals_bound) {
        for (int jj = 0; jj < n_draws_keep; jj++) {
            draws_out.row(jj) = arma::trans(inv_transform(draws_out.row(jj).t(), bounds_type, lower_bounds, upper_bounds));
        }
    }
    //
    if (settings_inp) {
        settings_inp->rwmh_accept_rate = (double) n_accept / (double) n_draws_keep;
    }
    //
    success = true;
    return success;
}

// wrappers

bool
mcmc::rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data)
{
    return rwmh_int(initial_vals,draws_out,target_log_kernel,target_data,nullptr);
}

bool
mcmc::rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, mcmc_settings& settings)
{
    return rwmh_int(initial_vals,draws_out,target_log_kernel,target_data,&settings);
}
