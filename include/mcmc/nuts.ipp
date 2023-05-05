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

#ifndef _mcmc_nuts_IPP
#define _mcmc_nuts_IPP

// 

inline
fp_t
nuts_find_initial_step_size(
    const ColVec_t& draw_vec,
    const ColVec_t& mntm_vec,
    const Mat_t& inv_precond_matrix,
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> box_log_kernel_fn, 
    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, void* target_data)> leap_frog_fn,
    void* target_data
)
{
    fp_t step_size = fp_t(1); // initial value

    //

    fp_t prev_U = - box_log_kernel_fn(draw_vec, nullptr, target_data);
        
    if (!std::isfinite(prev_U)) {
        prev_U = posinf;
    }

    fp_t prev_K = BMO_MATOPS_DOT_PROD(mntm_vec, inv_precond_matrix * mntm_vec) / fp_t(2);

    //

    ColVec_t new_draw = draw_vec;
    ColVec_t new_mntm = mntm_vec;

    leap_frog_fn(step_size, 1, new_draw, new_mntm, target_data);

    fp_t prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);
    
    if (!std::isfinite(prop_U)) {
        prop_U = posinf;
    }

    fp_t prop_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

    //

    int a_val = 2 * ( - (prop_U + prop_K) + (prev_U + prev_K) > std::log(0.5) ) - 1;
    bool check_cond = (- (prop_U + prop_K) + (prev_U + prev_K)) > - std::log(2);

    while (check_cond) {
        step_size *= std::pow(2, a_val);

        leap_frog_fn(step_size, 1, new_draw, new_mntm, target_data);

        prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);
    
        if (!std::isfinite(prop_U)) {
            prop_U = posinf;
        }

        prop_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

        //

        a_val = 2 * ( (- (prop_U + prop_K) + (prev_U + prev_K)) > std::log(0.5)) - 1; // not obvious from the paper, but I think we should update a...
        check_cond = (- (prop_U + prop_K) + (prev_U + prev_K)) > - std::log(2);
    }

    return step_size;
}

//

inline
void
nuts_build_tree(
    const int direction_val,         // v
    const fp_t step_size,            // epsilon
    const fp_t log_rand_val,             // u
    const fp_t prev_U,               // - L(\theta^0)
    const fp_t prev_K,               // 0.5 * [r^0]^T M^{-1} [r^0]
    const ColVec_t& draw_vec,        // \theta
    const ColVec_t& mntm_vec,        // r
    const Mat_t& inv_precond_matrix, // not in the original paper
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> box_log_kernel_fn, 
    std::function<void (const fp_t step_size, const size_t n_leap_steps, ColVec_t& new_draw, ColVec_t& new_mntm, void* target_data)> leap_frog_fn, 
    const size_t tree_depth,         // j
    ColVec_t& new_draw,              // \theta'
    ColVec_t& new_draw_pos,          // \theta^+
    ColVec_t& new_draw_neg,          // \theta^-
    ColVec_t& new_mntm_pos,          // r^+
    ColVec_t& new_mntm_neg,          // r^-
    size_t& n_val,                   // n and n'
    size_t& s_val,                   // s and s'
    fp_t& alpha_val,                 // \alpha and \alpha'
    size_t& n_alpha_val,             // n_\alpha and n_\alpha'
    rand_engine_t& rand_engine,
    void* target_data
)
{
    const fp_t max_tuning_par = 1000; // from the paper: We recommend setting \delta_max to a large value like 1000 so that it does not interfere with the algorithm so long as the simulation is even moderately accurate.

    if (tree_depth == size_t(0)) {
        new_draw = draw_vec;
        ColVec_t new_mntm = mntm_vec;

        //

        leap_frog_fn(direction_val * step_size, 1, new_draw, new_mntm, target_data);

        fp_t prop_U = - box_log_kernel_fn(new_draw, nullptr, target_data);
        
        if (!std::isfinite(prop_U)) {
            prop_U = posinf;
        }

        fp_t prop_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_precond_matrix * new_mntm) / fp_t(2);

        //

        // fp_t log_log_rand_val = std::log(rand_val);

        n_val = (log_rand_val <= - prop_U - prop_K);
        s_val = (log_rand_val < max_tuning_par - prop_U - prop_K);

        //

        new_draw_pos = new_draw;
        new_draw_neg = new_draw;

        new_mntm_pos = new_mntm;
        new_mntm_neg = new_mntm;

        alpha_val = std::exp( std::min( fp_t(0), - (prop_U + prop_K) + (prev_U + prev_K)) );
        n_alpha_val = 1;
    } else {
        size_t n_p_val;
        size_t s_p_val;
        fp_t alpha_p_val;
        size_t n_alpha_p_val;
        ColVec_t new_draw_p;

        nuts_build_tree(
            direction_val, step_size, log_rand_val, prev_U, prev_K,
            draw_vec, mntm_vec, inv_precond_matrix,
            box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
            new_draw_p, new_draw_pos, new_draw_neg, new_mntm_pos, new_mntm_neg,
            n_p_val, s_p_val, alpha_p_val, n_alpha_p_val, rand_engine, target_data);
        
        //
        
        if (s_p_val == size_t(1)) {
            size_t n_pp_val;
            size_t s_pp_val;
            fp_t alpha_pp_val;
            size_t n_alpha_pp_val;

            ColVec_t new_draw_pp;

            //

            if (direction_val == -1) {
                ColVec_t dummy_draw = new_draw_pos;
                ColVec_t dummy_mntm = new_mntm_pos;
                ColVec_t draw_neg = new_draw_neg;
                ColVec_t mntm_neg = new_mntm_neg;

                nuts_build_tree(
                    direction_val, step_size, log_rand_val, prev_U, prev_K,
                    draw_neg, mntm_neg, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, new_draw_neg, dummy_draw, new_mntm_neg, dummy_mntm,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data);
            } else {
                ColVec_t dummy_draw = new_draw_neg;
                ColVec_t dummy_mntm = new_mntm_neg;
                ColVec_t draw_pos = new_draw_pos;
                ColVec_t mntm_pos = new_mntm_pos;

                nuts_build_tree(
                    direction_val, step_size, log_rand_val, prev_U, prev_K,
                    draw_pos, mntm_pos, inv_precond_matrix,
                    box_log_kernel_fn, leap_frog_fn, tree_depth - 1,
                    new_draw_pp, dummy_draw, new_draw_pos, dummy_mntm, new_mntm_pos,
                    n_pp_val, s_pp_val, alpha_pp_val, n_alpha_pp_val, rand_engine, target_data);
            }

            //

            const fp_t prob_val = fp_t(n_pp_val) / fp_t( n_p_val + n_pp_val );
            const fp_t z = bmo::stats::runif<fp_t>(rand_engine);

            if (z < prob_val) {
                new_draw_p = new_draw_pp;
            }

            n_p_val += n_pp_val;
            alpha_p_val += alpha_pp_val;
            n_alpha_p_val += n_alpha_pp_val;

            //

            int check_val_1 = BMO_MATOPS_DOT_PROD(new_draw_pos - new_draw_neg, new_mntm_neg) >= fp_t(0);
            int check_val_2 = BMO_MATOPS_DOT_PROD(new_draw_pos - new_draw_neg, new_mntm_pos) >= fp_t(0);

            s_p_val = s_pp_val * check_val_1 * check_val_2;
        }

        //

        n_val = n_p_val;
        s_val = s_p_val;
        alpha_val = alpha_p_val;
        n_alpha_val = n_alpha_p_val;

        new_draw = new_draw_p;
    }
}

#endif
