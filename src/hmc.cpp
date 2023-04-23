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
 * Hamiltonian Monte Carlo (HMC)
 */

#include "mcmc.hpp"

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::hmc_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(initial_vals);

    // settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_burnin_draws = settings.hmc_settings.n_burnin_draws;
    const size_t n_keep_draws   = settings.hmc_settings.n_keep_draws;
    const size_t n_total_draws  = n_burnin_draws + n_keep_draws;

    const fp_t step_size = settings.hmc_settings.step_size;
    const uint_t n_leap_steps = settings.hmc_settings.n_leap_steps;

    const Mat_t precond_matrix = (BMO_MATOPS_SIZE(settings.hmc_settings.precond_mat) == n_vals*n_vals) ? settings.hmc_settings.precond_mat : BMO_MATOPS_EYE(n_vals);
    const Mat_t inv_precond_matrix = BMO_MATOPS_INV(precond_matrix);
    const Mat_t sqrt_precond_matrix = BMO_MATOPS_CHOL_LOWER(precond_matrix);

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    rand_engine_t rand_engine(settings.rng_seed_value);

    // parallelization setup

#ifdef MCMC_USE_OPENMP
    int omp_n_threads = 1;

    if (settings.hmc_settings.omp_n_threads > 0) {
        omp_n_threads = settings.hmc_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif
    
    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_log_kernel \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data) \
    -> fp_t 
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return target_log_kernel(vals_inv_trans, nullptr, target_data) + log_jacobian(vals_inp, bounds_type, lower_bounds, upper_bounds);
        } else {
            return target_log_kernel(vals_inp, nullptr, target_data);
        }
    };

    // momentum update function
    
    std::function<ColVec_t (const ColVec_t& pos_inp, const ColVec_t& mntm_inp, void* target_data, const fp_t step_size, Mat_t* jacob_matrix_out)> mntm_update_fn \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& pos_inp, const ColVec_t& mntm_inp, void* target_data, const fp_t step_size, Mat_t* jacob_matrix_out) \
    -> ColVec_t 
    {
        const size_t n_vals = BMO_MATOPS_SIZE(pos_inp);

        ColVec_t grad_obj(n_vals);

        if (vals_bound) {
            ColVec_t pos_inv_trans = inv_transform(pos_inp, bounds_type, lower_bounds, upper_bounds);

            target_log_kernel(pos_inv_trans, &grad_obj, target_data);

            //

            Mat_t jacob_matrix = inv_jacobian_adjust(pos_inp, bounds_type, lower_bounds, upper_bounds);

            if (jacob_matrix_out) {
                *jacob_matrix_out = jacob_matrix;
            }

            //

            return mntm_inp + step_size * jacob_matrix * grad_obj / fp_t(2);
        } else {
            target_log_kernel(pos_inp,&grad_obj,target_data);

            return mntm_inp + step_size * grad_obj / fp_t(2);
        }
    };

    // setup
    
    ColVec_t first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);

    fp_t prev_U = - box_log_kernel(first_draw, nullptr, target_data);
    fp_t prop_U = prev_U;

    fp_t prop_K, prev_K;
    
    ColVec_t prev_draw = first_draw;
    ColVec_t new_draw  = first_draw;

    ColVec_t new_mntm(n_vals);

    //

    size_t n_accept = 0;
    ColVec_t rand_vec(n_vals);
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; ++draw_ind) {
        bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);

        new_mntm = sqrt_precond_matrix * rand_vec;

        prev_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

        new_draw = prev_draw;

        for (size_t k = 0; k < n_leap_steps; ++k) {
            // begin leap-frog steps

            new_mntm = mntm_update_fn(new_draw, new_mntm, target_data, step_size, nullptr); // first half-step

            //

            new_draw += step_size * inv_precond_matrix * new_mntm;

            //

            new_mntm = mntm_update_fn(new_draw, new_mntm, target_data, step_size, nullptr); // second half-step
        }

        prop_U = - box_log_kernel(new_draw, nullptr, target_data);
        
        if (!std::isfinite(prop_U)) {
            prop_U = posinf;
        }

        prop_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

        //

        const fp_t comp_val = std::min(fp_t(0.01), - (prop_U + prop_K) + (prev_U + prev_K));
        const fp_t z = bmo::stats::runif<fp_t>(rand_engine);

        if (z < std::exp(comp_val)) {
            prev_draw = new_draw;
            prev_U = prop_U;
            prev_K = prop_K;

            if (draw_ind >= n_burnin_draws) {
                draws_out.row(draw_ind - n_burnin_draws) = BMO_MATOPS_TRANSPOSE(new_draw);
                n_accept++;
            }
        } else {
            if (draw_ind >= n_burnin_draws) {
                draws_out.row(draw_ind - n_burnin_draws) = BMO_MATOPS_TRANSPOSE(prev_draw);
            }
        }
    }

    success = true;

    //

    if (vals_bound) {
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
        for (size_t draw_ind = 0; draw_ind < n_keep_draws; ++draw_ind) {
            draws_out.row(draw_ind) = inv_transform<RowVec_t>(draws_out.row(draw_ind), bounds_type, lower_bounds, upper_bounds);
        }
    }

    if (settings_inp) {
        settings_inp->hmc_settings.n_accept_draws = n_accept;
    }

    //

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::hmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
)
{
    return internal::hmc_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::hmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
)
{
    return internal::hmc_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}
