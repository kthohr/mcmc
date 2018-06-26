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
 * Metropolis-adjusted Langevin algorithm (MALA)
 */

#include "mcmc.hpp" 

bool
mcmc::mala_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = initial_vals.n_elem;

    //
    // MALA settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_draws_keep   = settings.mala_n_draws;
    const size_t n_draws_burnin = settings.mala_n_burnin;

    const double step_size = settings.mala_step_size;

    const arma::mat precond_matrix = (settings.mala_precond_mat.n_elem == n_vals*n_vals) ? settings.mala_precond_mat : arma::eye(n_vals,n_vals);
    const arma::mat sqrt_precond_matrix = arma::chol(precond_matrix,"lower");

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    //
    // lambda functions for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* box_data)> box_log_kernel \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data) \
    -> double 
    {
        if (vals_bound)
        {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, nullptr, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        }
        else
        {
            return target_log_kernel(vals_inp, nullptr, target_data);
        }
    };

    std::function<arma::vec (const arma::vec& vals_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out)> mala_mean_fn \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds, precond_matrix] (const arma::vec& vals_inp, void* target_data, const double step_size, arma::mat* jacob_matrix_out) \
    -> arma::vec
    {
        const size_t n_vals = vals_inp.n_elem;
        arma::vec grad_obj(n_vals);

        if (vals_bound)
        {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            target_log_kernel(vals_inv_trans,&grad_obj,target_data);

            //

            arma::mat jacob_matrix = inv_jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);

            if (jacob_matrix_out) {
                *jacob_matrix_out = jacob_matrix;
            }

            //

            return vals_inp + step_size * step_size * jacob_matrix * precond_matrix * grad_obj / 2.0;
        }
        else
        {
            target_log_kernel(vals_inp,&grad_obj,target_data);

            return vals_inp + step_size * step_size * precond_matrix * grad_obj / 2.0;
        }
    };

    //
    // setup
    
    arma::vec first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    draws_out.set_size(n_draws_keep, n_vals);

    double prev_LP = box_log_kernel(first_draw, nullptr, target_data);
    double prop_LP = prev_LP;
    
    arma::vec prev_draw = first_draw;
    arma::vec new_draw  = first_draw;

    //

    int n_accept = 0;    
    arma::vec krand(n_vals);
    
    for (size_t jj = 0; jj < n_draws_keep + n_draws_burnin; jj++)
    {
        if (vals_bound) 
        {
            arma::mat jacob_matrix;
            arma::vec mean_vec = mala_mean_fn(prev_draw, target_data, step_size, &jacob_matrix);
            
            new_draw = mean_vec + step_size * arma::chol(jacob_matrix,"lower") * sqrt_precond_matrix * krand.randn();
        }
        else
        {
            new_draw = mala_mean_fn(prev_draw, target_data, step_size, nullptr) + step_size * sqrt_precond_matrix * krand.randn();
        }
        
        prop_LP = box_log_kernel(new_draw, nullptr, target_data);
        
        if (!std::isfinite(prop_LP)) {
            prop_LP = neginf;
        }

        //

        double comp_val = std::min(0.0, prop_LP - prev_LP + mala_prop_adjustment(new_draw, prev_draw, step_size, vals_bound, precond_matrix, mala_mean_fn, target_data));
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
            if (jj >= n_draws_burnin) 
            {
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
        settings_inp->mala_accept_rate = static_cast<double>(n_accept) / static_cast<double>(n_draws_keep);
    }

    //

    return success;
}

// wrappers

bool
mcmc::mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data)
{
    return mala_int(initial_vals,draws_out,target_log_kernel,target_data,nullptr);
}

bool
mcmc::mala(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings)
{
    return mala_int(initial_vals,draws_out,target_log_kernel,target_data,&settings);
}
