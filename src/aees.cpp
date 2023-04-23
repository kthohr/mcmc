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
 * Adaptive Equi-Energy Sampler
 */

#include "mcmc.hpp"

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::aees_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(initial_vals);

    //
    // AEES settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_initial_draws = settings.aees_settings.n_initial_draws;
    const size_t n_burnin_draws  = settings.aees_settings.n_burnin_draws;
    const size_t n_keep_draws    = settings.aees_settings.n_keep_draws;

    const fp_t ee_prob_par = settings.aees_settings.ee_prob_par;
    const size_t n_rings = settings.aees_settings.n_rings;

    // temperature vector: add T = 1 and sort

    const size_t K = BMO_MATOPS_SIZE(settings.aees_settings.temper_vec) + 1;

    ColVec_t temper_vec(K + 1);

    if (K > 1) {
        for (size_t k = 0; k < K; ++k) {
            temper_vec(k) = settings.aees_settings.temper_vec(k);
        }
    }
    
    temper_vec(K - 1) = fp_t(1.0);

    bmo::sort(temper_vec, false);

    //

    const size_t n_total_draws = n_keep_draws + K * (n_initial_draws + n_burnin_draws);

    //

    const fp_t par_scale = settings.aees_settings.par_scale;
    const Mat_t cov_mcmc = (BMO_MATOPS_SIZE(settings.aees_settings.cov_mat) == n_vals*n_vals) ? settings.aees_settings.cov_mat : BMO_MATOPS_EYE(n_vals);

    const Mat_t sqrt_cov_mcmc = BMO_MATOPS_CHOL_LOWER(cov_mcmc);
    const Mat_t prop_scaling_mat = par_scale * sqrt_cov_mcmc;

    const bool vals_bound = settings.vals_bound;

    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // parallelization setup

    int omp_n_threads = 1;

#ifdef MCMC_USE_OPENMP
    if (settings.aees_settings.omp_n_threads > 0) {
        omp_n_threads = settings.aees_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif

    // random sampling setup

    rand_engine_t rand_engine(settings.rng_seed_value);
    std::vector<rand_engine_t> rand_engines_vec;

    for (int i = 0; i < omp_n_threads; ++i) {
        size_t seed_val = generate_seed_value(i, omp_n_threads, rand_engine);
        rand_engines_vec.push_back(rand_engine_t(seed_val));
    }

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> box_log_kernel \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, void* target_data) \
    -> fp_t 
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        } else {
            return target_log_kernel(vals_inp, target_data);
        }
    };

    //

    ColVec_t first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    Mat_t X_new(n_vals,K);
    X_new.col(0) = first_draw;
    
    Mat_t ring_vals(K, n_rings - 1);

    Mat_t kernel_vals(K, n_total_draws); // target kernel values at temperature = 1
    Mat_t kernel_vals_prev(2, K);      // store kernel values from run (n - 1) at two different temperatures
    Mat_t kernel_vals_new(2, K);       // store kernel values from run (n)     at two different temperatures

    Cube_t draw_storage(n_vals, K, n_total_draws);

    //
    // begin loop

    for (size_t draw_ind = 0; draw_ind < n_total_draws; ++draw_ind) {
        Mat_t X_prev = X_new;
        kernel_vals_prev = kernel_vals_new;

        //

        fp_t val_out_hot; // holds kernel value

        X_new.col(0) = single_step_mh(X_prev.col(0), temper_vec(0), prop_scaling_mat, box_log_kernel, target_data, rand_engine, &val_out_hot);
        
        BMO_MATOPS_SET_VALUES_SCALAR(kernel_vals_new.col(0), val_out_hot);
        
        // loop down temperature vector
        
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
        for (size_t temper_ind = 1; temper_ind < K; ++temper_ind) {
            size_t thread_num = 0;

#ifdef MCMC_USE_OPENMP
            thread_num = omp_get_thread_num();
#endif

            if ( draw_ind > temper_ind * (n_initial_draws + n_burnin_draws) ) {
                fp_t val_out_j;
                
                const fp_t z_eps = bmo::stats::runif<fp_t>(rand_engines_vec[thread_num]);

                if (z_eps > ee_prob_par) {
                    X_new.col(temper_ind) = single_step_mh(X_prev.col(temper_ind), temper_vec(temper_ind), prop_scaling_mat, box_log_kernel, target_data, rand_engines_vec[thread_num], &val_out_j);

                    kernel_vals_new(0, temper_ind) = val_out_j / temper_vec(temper_ind - 1);
                    kernel_vals_new(1, temper_ind) = val_out_j / temper_vec(temper_ind);
                } else {
                    size_t draws_j_begin_ind = (temper_ind - 1) * (n_initial_draws + n_burnin_draws);

                    size_t ring_ind_spacing = std::floor( static_cast<fp_t>(draw_ind - draws_j_begin_ind + 1) / n_rings);

                    if (ring_ind_spacing == 0) {
                        X_new.col(temper_ind) = X_prev.col(temper_ind);
                        kernel_vals_new.col(temper_ind) = kernel_vals_prev.col(temper_ind);
                    } else {
                        // ColVec_t past_kernel_vals = arma::trans(kernel_vals(arma::span(temper_ind-1,temper_ind-1),arma::span(draws_j_begin_ind, draw_ind)));
                        ColVec_t past_kernel_vals = BMO_MATOPS_TRANSPOSE(BMO_MATOPS_MIDDLE_COLS(kernel_vals.row(temper_ind - 1), draws_j_begin_ind, draw_ind));

                        ColVecUInt_t sort_vec = bmo::get_sort_index(past_kernel_vals);
                        past_kernel_vals = past_kernel_vals(sort_vec);

                        // construct rings

                        for (size_t i = 0; i < (n_rings - 1); ++i) {
                            size_t ring_i_ind = (i + 1) * ring_ind_spacing;
                            ring_vals(temper_ind-1, i) = (past_kernel_vals(ring_i_ind) + past_kernel_vals(ring_i_ind - 1)) / fp_t(2);
                        }

                        size_t which_ring = 0;

                        while ( which_ring < (n_rings - 1) && kernel_vals(temper_ind, draw_ind - 1) > ring_vals(temper_ind - 1, which_ring) ) {
                            which_ring++;
                        }

                        //

                        const fp_t z_tmp = bmo::stats::runif<fp_t>(rand_engines_vec[thread_num]);

                        const size_t ind_mix = sort_vec( static_cast<size_t>(ring_ind_spacing * which_ring + std::floor(z_tmp * ring_ind_spacing)) );

                        //

                        X_new.col(temper_ind) = draw_storage.mat(ind_mix).col(temper_ind-1);

                        val_out_j = box_log_kernel(X_new.col(temper_ind), target_data);

                        kernel_vals_new(0,temper_ind) = val_out_j / temper_vec(temper_ind - 1);
                        kernel_vals_new(1,temper_ind) = val_out_j / temper_vec(temper_ind);

                        //
                        
                        const fp_t comp_val_1 = kernel_vals_new(1, temper_ind) - kernel_vals_prev(1, temper_ind);
                        const fp_t comp_val_2 = kernel_vals_prev(0, temper_ind) - kernel_vals_new(0, temper_ind);

                        const fp_t comp_val = std::min(fp_t(0.01), comp_val_1 + comp_val_2);
                        const fp_t z = bmo::stats::runif<fp_t>(rand_engines_vec[thread_num]);

                        if (z > std::exp(comp_val)) {
                            X_new.col(temper_ind) = X_prev.col(temper_ind);
                            kernel_vals_new.col(temper_ind) = kernel_vals_prev.col(temper_ind);
                        }
                    }
                }

                // store target kernel values
                kernel_vals(temper_ind, draw_ind) = box_log_kernel(X_new.col(temper_ind), target_data); // temperature = 1
            }
        }
        
        // store draws
        draw_storage.mat(draw_ind) = X_new;
    }

    //

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);

    size_t draw_ind_begin = K * (n_initial_draws + n_burnin_draws);
    size_t draw_save_ind = 0;

    for (size_t draw_ind = draw_ind_begin; draw_ind < n_total_draws; ++draw_ind) {
        // ColVec_t tmp_vec = draw_storage(arma::span(),arma::span(K-1,K-1),arma::span(i,i));
        ColVec_t tmp_vec = draw_storage.mat(draw_ind).col(K-1);

        if (vals_bound) {
            tmp_vec = inv_transform(tmp_vec, bounds_type, lower_bounds, upper_bounds);
        }

        draws_out.row(draw_save_ind) = BMO_MATOPS_TRANSPOSE(tmp_vec);
        draw_save_ind++;
    }
    
    //

    success = true;

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::aees(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
)
{
    return internal::aees_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::aees(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel,
    Mat_t& draws_out, 
    void* target_data,
    algo_settings_t& settings
)
{
    return internal::aees_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}

