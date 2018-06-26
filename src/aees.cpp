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
 * Adaptive Equi-Energy Sampler
 */

#include "mcmc.hpp"

bool
mcmc::aees_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = initial_vals.n_elem;

    //
    // AEES settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_draws         = settings.aees_n_draws;
    const size_t n_initial_draws = settings.aees_n_initial_draws;
    const size_t n_burnin        = settings.aees_n_burnin;

    const double ee_prob_par = settings.aees_prob_par;

    // temperature vector: add T = 1 and sort

    arma::vec temper_vec = settings.aees_temper_vec;
    const uint_t K = temper_vec.n_elem + 1;
    
    temper_vec.resize(K);
    temper_vec(K-1) = 1.0;

    temper_vec = arma::sort(temper_vec,"descend"); // largest to smallest

    //

    const size_t n_rings = settings.aees_n_rings;

    const size_t total_draws = n_draws + K*(n_initial_draws + n_burnin);

    //

    const double par_scale = settings.rwmh_par_scale;
    const arma::mat cov_mcmc = (settings.rwmh_cov_mat.n_elem == n_vals*n_vals) ? settings.rwmh_cov_mat : arma::eye(n_vals,n_vals);

    const arma::mat sqrt_cov_mcmc = par_scale*arma::chol(cov_mcmc,"lower");

    const bool vals_bound = settings.vals_bound;

    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, void* target_data)> box_log_kernel \
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

    arma::vec first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }
    
    arma::cube X_out(n_vals,K,total_draws);

    arma::mat X_new(n_vals,K);
    X_new.col(0) = first_draw;
    
    arma::mat ring_vals(K,n_rings-1);

    arma::mat kernel_vals(K,total_draws); // target kernel values at temperature = 1
    arma::mat kernel_vals_prev(2,K);      // store kernel values from run (n - 1) at two different temperatures
    arma::mat kernel_vals_new(2,K);       // store kernel values from run (n)     at two different temperatures

    //
    // begin loop

    for (size_t n=0; n < total_draws; n++)
    {
        arma::mat X_prev = X_new;
        kernel_vals_prev = kernel_vals_new;

        //

        double val_out_hot; // holds kernel value

        X_new.col(0) = single_step_mh(X_prev.col(0), temper_vec(0), sqrt_cov_mcmc, box_log_kernel, target_data, &val_out_hot);
        
        kernel_vals_new.col(0).fill(val_out_hot);
        
        // loop down temperature vector
        
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t j=1; j < K; j++) 
        {
            if (n > j*(n_initial_draws + n_burnin))
            {
                double val_out_j;
                double z_eps = arma::as_scalar(arma::randu(1,1));

                if (z_eps > ee_prob_par) 
                {
                    X_new.col(j) = single_step_mh(X_prev.col(j), temper_vec(j), sqrt_cov_mcmc, box_log_kernel, target_data, &val_out_j);

                    kernel_vals_new(0,j) = val_out_j / temper_vec(j-1);
                    kernel_vals_new(1,j) = val_out_j / temper_vec(j);
                }
                else
                {
                    size_t draws_j_begin_ind = (j-1)*(n_initial_draws + n_burnin);

                    size_t ring_ind_spacing = std::floor( (double)(n - draws_j_begin_ind + 1) / n_rings);

                    if (ring_ind_spacing == 0)
                    {
                        X_new.col(j) = X_prev.col(j);
                        kernel_vals_new.col(j) = kernel_vals_prev.col(j);
                    }
                    else
                    {
                        arma::vec past_kernel_vals = arma::trans(kernel_vals(arma::span(j-1,j-1),arma::span(draws_j_begin_ind,n)));

                        arma::uvec sort_ind = arma::sort_index(past_kernel_vals);
                        past_kernel_vals = past_kernel_vals(sort_ind);

                        // construct rings

                        for (size_t i=0; i < (n_rings-1); i++)
                        {
                            int ring_i_ind = (i+1)*ring_ind_spacing;
                            ring_vals(j-1,i) = (past_kernel_vals(ring_i_ind) + past_kernel_vals(ring_i_ind-1)) / 2.0;
                        }

                        size_t which_ring = 0;
                        while ( which_ring < (n_rings-1) && kernel_vals(j,n-1) > ring_vals(j-1,which_ring) )
                        {
                            which_ring++;
                        }

                        //

                        double z_tmp = arma::as_scalar(arma::randu());

                        size_t ind_mix = sort_ind( static_cast<size_t>(ring_ind_spacing*which_ring + std::floor(z_tmp * ring_ind_spacing)) );

                        //

                        X_new.col(j) = X_out.slice(ind_mix).col(j-1);

                        val_out_j = box_log_kernel(X_new.col(j), target_data);

                        kernel_vals_new(0,j) = val_out_j / temper_vec(j-1);
                        kernel_vals_new(1,j) = val_out_j / temper_vec(j);

                        //
                        
                        double comp_val_1 = kernel_vals_new(1,j) - kernel_vals_prev(1,j);
                        double comp_val_2 = kernel_vals_prev(0,j) - kernel_vals_new(0,j);

                        double comp_val = std::min(0.0, comp_val_1 + comp_val_2);
                        double z = arma::as_scalar(arma::randu(1,1));

                        if (z > std::exp(comp_val)) 
                        {
                            X_new.col(j) = X_prev.col(j);
                            kernel_vals_new.col(j) = kernel_vals_prev.col(j);
                        }
                    }
                }

                // store target kernel values
                kernel_vals(j,n) = box_log_kernel(X_new.col(j), target_data); // temperature = 1
            }
        }
        
        // store draws
        X_out.slice(n) = X_new;
    }

    //

    draws_out.set_size(total_draws,n_vals);

    for (size_t i=0; i < total_draws; i++) 
    {
        arma::vec tmp_vec = X_out(arma::span(),arma::span(K-1,K-1),arma::span(i,i));

        if (vals_bound) {
            tmp_vec = inv_transform(tmp_vec, bounds_type, lower_bounds, upper_bounds);
        }

        draws_out.row(i) = tmp_vec.t();
    }

    draws_out.shed_rows(0,K*(n_initial_draws + n_burnin)-1);
    
    //

    success = true;

    return success;
}

// wrappers

bool
mcmc::aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data)
{
    return aees_int(initial_vals,draws_out,target_log_kernel,target_data,nullptr);
}

bool
mcmc::aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings)
{
    return aees_int(initial_vals,draws_out,target_log_kernel,target_data,&settings);
}

