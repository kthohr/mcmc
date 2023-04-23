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
 * Metropolis-adjusted Langevin algorithm (MALA)
 */

#include "mcmc.hpp" 

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::mala_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(initial_vals);

    //
    // MALA settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_burnin_draws = settings.mala_settings.n_burnin_draws;
    const size_t n_keep_draws   = settings.mala_settings.n_keep_draws;
    const size_t n_total_draws  = n_burnin_draws + n_keep_draws;

    const fp_t step_size = settings.mala_settings.step_size;

    const Mat_t precond_matrix = (BMO_MATOPS_SIZE(settings.mala_settings.precond_mat) == n_vals*n_vals) ? settings.mala_settings.precond_mat : BMO_MATOPS_EYE(n_vals);
    const Mat_t sqrt_precond_matrix = BMO_MATOPS_CHOL_LOWER(precond_matrix);

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    rand_engine_t rand_engine(settings.rng_seed_value);

    // parallelization setup

#ifdef MCMC_USE_OPENMP
    int omp_n_threads = 1;
    
    if (settings.mala_settings.omp_n_threads > 0) {
        omp_n_threads = settings.mala_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif

    //
    // lambda functions for box constraints

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

    std::function<ColVec_t (const ColVec_t& vals_inp, void* target_data, const fp_t step_size, Mat_t* jacob_matrix_out)> mala_mean_fn \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds, precond_matrix] (const ColVec_t& vals_inp, void* target_data, const fp_t step_size, Mat_t* jacob_matrix_out) \
    -> ColVec_t
    {
        const size_t n_vals = BMO_MATOPS_SIZE(vals_inp);
        ColVec_t grad_obj(n_vals);

        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            target_log_kernel(vals_inv_trans,&grad_obj,target_data);

            //

            Mat_t jacob_matrix = inv_jacobian_adjust(vals_inp, bounds_type, lower_bounds, upper_bounds);

            if (jacob_matrix_out) {
                *jacob_matrix_out = jacob_matrix;
            }

            //

            return vals_inp + step_size * step_size * jacob_matrix * precond_matrix * grad_obj / fp_t(2);
        } else {
            target_log_kernel(vals_inp, &grad_obj, target_data);

            return vals_inp + step_size * step_size * precond_matrix * grad_obj / fp_t(2);
        }
    };

    //
    // setup
    
    ColVec_t first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);

    fp_t prev_LP = box_log_kernel(first_draw, nullptr, target_data);
    fp_t prop_LP = prev_LP;
    
    ColVec_t prev_draw = first_draw;
    ColVec_t new_draw  = first_draw;

    //

    size_t n_accept = 0;
    ColVec_t rand_vec(n_vals);
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; ++draw_ind) {
        bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);

        if (vals_bound) {
            Mat_t jacob_matrix;

            ColVec_t mean_vec = mala_mean_fn(prev_draw, target_data, step_size, &jacob_matrix);
            
            new_draw = mean_vec + step_size * BMO_MATOPS_CHOL_LOWER(jacob_matrix) * sqrt_precond_matrix * rand_vec;
        } else {
            new_draw = mala_mean_fn(prev_draw, target_data, step_size, nullptr) + step_size * sqrt_precond_matrix * rand_vec;
        }
        
        prop_LP = box_log_kernel(new_draw, nullptr, target_data);
        
        if (!std::isfinite(prop_LP)) {
            prop_LP = neginf;
        }

        //

        const fp_t comp_val = std::min(fp_t(0.01), prop_LP - prev_LP + mala_prop_adjustment(new_draw, prev_draw, step_size, vals_bound, precond_matrix, mala_mean_fn, target_data));
        const fp_t z = bmo::stats::runif<fp_t>(rand_engine);

        if (z < std::exp(comp_val)) {
            prev_draw = new_draw;
            prev_LP = prop_LP;

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
        settings_inp->mala_settings.n_accept_draws = n_accept;
    }

    //

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::mala(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
)
{
    return internal::mala_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::mala(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel,
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
)
{
    return internal::mala_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}
