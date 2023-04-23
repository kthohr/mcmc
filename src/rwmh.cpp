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
 * Random Walk Metropolis-Hastings (RWMH) MCMC
 */

#include "mcmc.hpp" 

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::rwmh_impl(
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
    // RWMH settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_keep_draws   = settings.rwmh_settings.n_keep_draws;
    const size_t n_burnin_draws = settings.rwmh_settings.n_burnin_draws;
    const size_t n_total_draws  = n_burnin_draws + n_keep_draws;

    const fp_t par_scale = settings.rwmh_settings.par_scale;

    const Mat_t cov_mcmc = (BMO_MATOPS_SIZE(settings.rwmh_settings.cov_mat) == n_vals*n_vals) ? settings.rwmh_settings.cov_mat : BMO_MATOPS_EYE(n_vals);

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    rand_engine_t rand_engine(settings.rng_seed_value);

    // parallelization setup

#ifdef MCMC_USE_OPENMP
    int omp_n_threads = 1;

    if (settings.rwmh_settings.omp_n_threads > 0) {
        omp_n_threads = settings.rwmh_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, void* box_data)> box_log_kernel \
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
    // setup
    
    ColVec_t rand_vec(n_vals);
    ColVec_t first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);

    fp_t prev_LP = box_log_kernel(first_draw, target_data);
    fp_t prop_LP = prev_LP;
    
    ColVec_t prev_draw = first_draw;
    ColVec_t new_draw  = first_draw;
    
    Mat_t cov_mcmc_chol = par_scale * BMO_MATOPS_CHOL_LOWER(cov_mcmc);

    //

    size_t n_accept_draws = 0;
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; ++draw_ind) {

        // new_draw = prev_draw + cov_mcmc_chol * bmo::stats::rnorm_vec<fp_t>(n_vals, rand_engine);
        bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);
        new_draw = prev_draw + cov_mcmc_chol * rand_vec;
        
        prop_LP = box_log_kernel(new_draw, target_data);
        
        if (!std::isfinite(prop_LP)) {
            prop_LP = neginf;
        }

        //

        fp_t comp_val = std::min(fp_t(0.0), prop_LP - prev_LP);
        fp_t z = bmo::stats::runif<fp_t>(rand_engine);

        if (z < std::exp(comp_val)) {
            prev_draw = new_draw;
            prev_LP = prop_LP;

            if (draw_ind >= n_burnin_draws) {
                ++n_accept_draws;
            }
        }

        //

        if (draw_ind >= n_burnin_draws) {
            draws_out.row(draw_ind - n_burnin_draws) = BMO_MATOPS_TRANSPOSE(prev_draw);
        }
    }

    success = true;

    //

    if (vals_bound) {
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
        for (size_t draw_ind = 0; draw_ind < n_keep_draws; draw_ind++) {
            draws_out.row(draw_ind) = inv_transform<RowVec_t>(draws_out.row(draw_ind), bounds_type, lower_bounds, upper_bounds);
        }
    }

    if (settings_inp) {
        settings_inp->rwmh_settings.n_accept_draws = n_accept_draws;
    }

    //

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::rwmh(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
)
{
    return internal::rwmh_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::rwmh(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
)
{
    return internal::rwmh_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}
