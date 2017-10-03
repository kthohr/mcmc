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
 * Hamiltonian Monte Carlo
 */

#include "mcmc.hpp" 

bool
mcmc::hmc_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings* settings_inp)
{
    bool success = false;

    const double BIG_NEG_VAL = MCMC_BIG_NEG_VAL;
    const int n_vals = initial_vals.n_elem;

    //
    // MALA settings

    algo_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int n_draws_keep   = settings.hmc_n_draws;
    const int n_draws_burnin = settings.hmc_n_burnin;

    const double step_size = settings.hmc_step_size;

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);
    
    //
    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* box_data)> box_log_kernel = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data) -> double {
        //
        if (vals_bound) {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, nullptr, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        } else {
            return target_log_kernel(vals_inp, nullptr, target_data);
        }
    };

    // momentum update
    std::function<arma::vec (const arma::vec& pos_inp, const arma::vec& mntm_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out)> mntm_update_fn = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& pos_inp, const arma::vec& mntm_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out) -> arma::vec {

        const int n_vals = pos_inp.n_elem;
        arma::vec grad_obj(n_vals);

        if (vals_bound) {

            arma::vec pos_inv_trans = inv_transform(pos_inp, bounds_type, lower_bounds, upper_bounds);

            target_log_kernel(pos_inv_trans,&grad_obj,target_data);

            //

            arma::mat jacob_matrix = inv_jacobian_adjust(pos_inp,bounds_type,lower_bounds,upper_bounds);

            if (jacob_matrix_out) {
                *jacob_matrix_out = jacob_matrix;
            }

            //

            arma::vec mntm_out = mntm_inp + step_size * jacob_matrix * grad_obj / 2.0;

            return mntm_out;
        } else {
            target_log_kernel(pos_inp,&grad_obj,target_data);

            arma::vec mntm_out = mntm_inp + step_size * grad_obj / 2.0;

            return mntm_out;
        }
    };

    //
    // setup
    
    arma::vec first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    draws_out.set_size(n_draws_keep, n_vals);

    double prev_U = - box_log_kernel(first_draw, nullptr, target_data);
    double prop_U = prev_U;

    double prop_K, prev_K;
    
    arma::vec prev_draw = first_draw;
    arma::vec new_draw  = first_draw;

    arma::vec new_mntm  = arma::randn(n_vals,1);

    //

    int n_accept = 0;
    arma::vec krand(n_vals);
    
    for (int jj = 0; jj < n_draws_keep + n_draws_burnin; jj++) {

        krand.randn();

        new_mntm = mntm_update_fn(prev_draw,krand,target_data,step_size,nullptr); // half-step
        prev_K = arma::dot(krand,krand) / 2.0;

        //

        new_draw = prev_draw + step_size*new_mntm;
        
        prop_U = - box_log_kernel(new_draw, nullptr, target_data);
        
        if (!std::isfinite(prop_U)) {
            prop_U = -BIG_NEG_VAL;
        }

        //

        new_mntm = mntm_update_fn(new_draw,new_mntm,target_data,step_size,nullptr); // half-step

        prop_K = arma::dot(new_mntm,new_mntm) / 2.0;

        //

        double comp_val = std::min(0.0,- prop_U - prop_K + prev_U + prev_K);
        double z = arma::as_scalar(arma::randu(1));

        if (z < std::exp(comp_val)) {
            prev_draw = new_draw;
            prev_U = prop_U;
            prev_K = prop_K;

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

    success = true;

    //

    if (vals_bound) {
#ifdef MCMC_USE_OMP
        #pragma omp parallel for
#endif
        for (int jj = 0; jj < n_draws_keep; jj++) {
            draws_out.row(jj) = arma::trans(inv_transform(draws_out.row(jj).t(), bounds_type, lower_bounds, upper_bounds));
        }
    }

    if (settings_inp) {
        settings_inp->hmc_accept_rate = (double) n_accept / (double) n_draws_keep;
    }

    //

    return success;
}

// wrappers

bool
mcmc::hmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data)
{
    return hmc_int(initial_vals,draws_out,target_log_kernel,target_data,nullptr);
}

bool
mcmc::hmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings& settings)
{
    return hmc_int(initial_vals,draws_out,target_log_kernel,target_data,&settings);
}
