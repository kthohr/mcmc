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
 * No-U-Turn Sampler (NUTS) (with Dual Averaging)
 */

#include "mcmc.hpp"

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::nuts_impl(
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

    const size_t n_burnin_draws = settings.nuts_settings.n_burnin_draws;
    const size_t n_keep_draws   = settings.nuts_settings.n_keep_draws;
    const size_t n_total_draws  = n_burnin_draws + n_keep_draws;

    const size_t n_adapt_draws = (settings.nuts_settings.n_adapt_draws <= n_total_draws) ? settings.nuts_settings.n_adapt_draws : n_total_draws;
    const fp_t target_accept_rate = settings.nuts_settings.target_accept_rate;

    const size_t max_tree_depth = settings.nuts_settings.max_tree_depth;

    fp_t epsilon_bar     = settings.nuts_settings.step_size; // \bar{\epsilon}_0
    const fp_t gamma_val = settings.nuts_settings.gamma_val;
    const fp_t t0_val    = settings.nuts_settings.t0_val;
    const fp_t kappa_val = settings.nuts_settings.kappa_val;

    const Mat_t precond_matrix = (BMO_MATOPS_SIZE(settings.nuts_settings.precond_mat) == n_vals*n_vals) ? settings.nuts_settings.precond_mat : BMO_MATOPS_EYE(n_vals);
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

    if (settings.nuts_settings.omp_n_threads > 0) {
        omp_n_threads = settings.nuts_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif
    
    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_log_kernel_fn \
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

    // leapfrog function

    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, void* target_data)> leap_frog_fn \
    = [mntm_update_fn, inv_precond_matrix] (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, void* target_data) \
    -> void 
    {
        for (size_t k = 0; k < n_leap_steps; ++k) {
            new_mntm = mntm_update_fn(new_draw, new_mntm, target_data, step_size, nullptr); // first half-step

            //

            new_draw += step_size * inv_precond_matrix * new_mntm;

            //

            new_mntm = mntm_update_fn(new_draw, new_mntm, target_data, step_size, nullptr); // second half-step
        }
    };

    // setup
    
    ColVec_t first_draw = initial_vals;

    if (vals_bound) { // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    ColVec_t rand_vec(n_vals);

    bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);

    ColVec_t mntm_vec = sqrt_precond_matrix * rand_vec;

    //

    fp_t step_size = nuts_find_initial_step_size(first_draw, mntm_vec, inv_precond_matrix, box_log_kernel_fn, leap_frog_fn, target_data);

    const fp_t mu_val = std::log(10 * step_size);
    fp_t h_val = 0;

    //

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);

    fp_t prev_U = - box_log_kernel_fn(first_draw, nullptr, target_data);
    fp_t prop_U = prev_U;

    fp_t prev_K;
    fp_t log_rand_val;
    
    ColVec_t prev_draw = first_draw;
    ColVec_t new_draw  = first_draw;

    ColVec_t draw_pos  = first_draw;
    ColVec_t draw_neg  = first_draw;
    ColVec_t mntm_pos  = mntm_vec;
    ColVec_t mntm_neg  = mntm_vec;

    // 

    size_t n_accept = 0;
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; ++draw_ind) {
        bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);

        mntm_vec = sqrt_precond_matrix * rand_vec;

        prev_K = BMO_MATOPS_DOT_PROD(mntm_vec, inv_precond_matrix * mntm_vec) / fp_t(2);

        log_rand_val = std::log(bmo::stats::runif<fp_t>(rand_engine)) - prev_U - prev_K ;

        //

        new_draw = prev_draw;

        draw_pos = prev_draw; 
        draw_neg = prev_draw; 
        mntm_pos = mntm_vec;
        mntm_neg = mntm_vec;

        size_t tree_depth = 0;
        size_t n_val = 1;
        size_t s_val = 1;

        fp_t alpha_val = 0;
        size_t n_alpha_val = 0;
        int good_round = 0;

        //

        while (s_val == size_t(1) && tree_depth < max_tree_depth) {
            size_t n_p_val;
            size_t s_p_val;

            //

            fp_t z = bmo::stats::runif<fp_t>(rand_engine);

            int direction_val = (z <= fp_t(0.5)) ? -1 : 1;

            if (direction_val == -1) {
                ColVec_t dummy_draw = draw_pos;
                ColVec_t dummy_mntm = mntm_pos;

                nuts_build_tree(
                    direction_val, step_size, log_rand_val, prev_U, prev_K,
                    prev_draw, mntm_vec, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth,
                    new_draw, dummy_draw, draw_neg, dummy_mntm, mntm_neg,
                    n_p_val, s_p_val, alpha_val, n_alpha_val, rand_engine, target_data);
            } else {
                ColVec_t dummy_draw = draw_neg;
                ColVec_t dummy_mntm = mntm_neg;

                nuts_build_tree(direction_val, step_size, log_rand_val, prev_U, prev_K,
                    prev_draw, mntm_vec, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth,
                    new_draw, draw_pos, dummy_draw, mntm_pos, dummy_mntm,
                    n_p_val, s_p_val, alpha_val, n_alpha_val, rand_engine, target_data);
            }

            //

            if (s_p_val == size_t(1)) {
                z = bmo::stats::runif<fp_t>(rand_engine);

                if (z < fp_t(n_p_val) / fp_t(n_val)) {
                    prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);
            
                    if (!std::isfinite(prop_U)) {
                        prop_U = posinf;
                    }

                    //

                    prev_draw = new_draw;
                    prev_U = prop_U;

                    //

                    good_round = 1;
                }
            }

            //

            n_val += n_p_val;
            tree_depth += 1;

            int check_val_1 = BMO_MATOPS_DOT_PROD(draw_pos - draw_neg, mntm_neg) >= fp_t(0);
            int check_val_2 = BMO_MATOPS_DOT_PROD(draw_pos - draw_neg, mntm_pos) >= fp_t(0);

            s_val = s_p_val * check_val_1 * check_val_2;
        } // while s == 1

        //

        if (draw_ind < n_adapt_draws) {
            h_val += (1/fp_t(draw_ind + 1 + t0_val)) * ( target_accept_rate - (fp_t(alpha_val) / fp_t(n_alpha_val)) - h_val);

            step_size = std::exp(mu_val - h_val * std::sqrt(draw_ind + 1) / gamma_val);

            epsilon_bar *= std::exp( std::pow(draw_ind + 1, - kappa_val) * (std::log(step_size) - std::log(epsilon_bar)) );
        } else {
            step_size = epsilon_bar;
        }

        //

        if (draw_ind >= n_burnin_draws) {
            draws_out.row(draw_ind - n_burnin_draws) = BMO_MATOPS_TRANSPOSE(prev_draw);
            n_accept += good_round;
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
        settings_inp->nuts_settings.n_accept_draws = n_accept;
    }

    //

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::nuts(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
)
{
    return internal::nuts_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::nuts(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
)
{
    return internal::nuts_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}
