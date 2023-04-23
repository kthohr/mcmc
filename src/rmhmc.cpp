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
 * Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC)
 */

#include "mcmc.hpp" 

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::rmhmc_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel,  
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(initial_vals);

    //
    // settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_burnin_draws = settings.rmhmc_settings.n_burnin_draws;
    const size_t n_keep_draws   = settings.rmhmc_settings.n_keep_draws;
    const size_t n_total_draws  = n_burnin_draws + n_keep_draws;

    const fp_t step_size = settings.rmhmc_settings.step_size;
    const uint_t n_leap_steps = settings.rmhmc_settings.n_leap_steps;
    const uint_t n_fp_steps = settings.rmhmc_settings.n_fp_steps;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    rand_engine_t rand_engine(settings.rng_seed_value);

    // parallelization setup

#ifdef MCMC_USE_OPENMP
    int omp_n_threads = 1;

    if (settings.rmhmc_settings.omp_n_threads > 0) {
        omp_n_threads = settings.rmhmc_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif
    
    //
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

    // momentum update

    std::function<ColVec_t (const ColVec_t& pos_inp, const ColVec_t& mntm_inp, void* target_data, const fp_t step_size, const Mat_t& inv_tensor_mat, const Cube_t& tensor_deriv, Mat_t* jacob_matrix_out)> mntm_update_fn \
    = [target_log_kernel, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& pos_inp, const ColVec_t& mntm_inp, void* target_data, const fp_t step_size, const Mat_t& inv_tensor_mat, const Cube_t& tensor_deriv, Mat_t* jacob_matrix_out) \
    -> ColVec_t 
    {
        const size_t n_vals = BMO_MATOPS_SIZE(pos_inp);

        ColVec_t grad_obj(n_vals);

        if (vals_bound) {
            ColVec_t pos_inv_trans = inv_transform(pos_inp, bounds_type, lower_bounds, upper_bounds);
            target_log_kernel(pos_inv_trans, &grad_obj, target_data);

            //

            for (size_t i = 0; i < n_vals; ++i) {
                Mat_t tmp_mat = inv_tensor_mat * tensor_deriv.mat(i);

                grad_obj(i) = - grad_obj(i) + fp_t(0.5) * ( BMO_MATOPS_TRACE(tmp_mat) - BMO_MATOPS_DOT_PROD(BMO_MATOPS_TRANSPOSE(tmp_mat) * mntm_inp, inv_tensor_mat * mntm_inp) );
            }

            //

            Mat_t jacob_matrix = inv_jacobian_adjust(pos_inp, bounds_type, lower_bounds, upper_bounds);

            if (jacob_matrix_out) {
                *jacob_matrix_out = jacob_matrix;
            }

            //

            return step_size * jacob_matrix * grad_obj / fp_t(2);
        } else {
            target_log_kernel(pos_inp,&grad_obj,target_data);

            //

            for (size_t i = 0; i < n_vals; ++i) {
                Mat_t tmp_mat = inv_tensor_mat * tensor_deriv.mat(i);

                grad_obj(i) = - grad_obj(i) + fp_t(0.5) * ( BMO_MATOPS_TRACE(tmp_mat) - BMO_MATOPS_DOT_PROD(BMO_MATOPS_TRANSPOSE(tmp_mat) * mntm_inp, inv_tensor_mat * mntm_inp) );
            }

            //

            ColVec_t mntm_out = step_size * grad_obj / fp_t(2);

            return step_size * grad_obj / fp_t(2);
        }
    };

    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> box_tensor_fn \
    = [tensor_fn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data) \
    -> Mat_t
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return tensor_fn(vals_inv_trans, tensor_deriv_out, tensor_data);
        } else {
            return tensor_fn(vals_inp, tensor_deriv_out, tensor_data);
        }
    };

    // setup
    
    ColVec_t first_draw = initial_vals;

    if (vals_bound){ // should we transform the parameters?
        first_draw = transform(initial_vals, bounds_type, lower_bounds, upper_bounds);
    }

    BMO_MATOPS_SET_SIZE(draws_out, n_keep_draws, n_vals);
    
    ColVec_t prev_draw = first_draw;
    ColVec_t new_draw  = first_draw;

    ColVec_t rand_vec  = bmo::stats::rnorm_vec<fp_t>(n_vals, rand_engine);
    ColVec_t new_mntm  = rand_vec;

    Cube_t new_deriv_cube;
    Mat_t new_tensor = box_tensor_fn(new_draw, &new_deriv_cube, tensor_data);

    Mat_t prev_tensor = new_tensor;
    Mat_t inv_new_tensor = BMO_MATOPS_INV(new_tensor);
    Mat_t inv_prev_tensor = inv_new_tensor;

    Cube_t prev_deriv_cube = new_deriv_cube;

    const fp_t cons_term = fp_t(0.5) * n_vals * MCMC_LOG_2PI;

    fp_t prev_U = cons_term - box_log_kernel(first_draw, nullptr, target_data) + fp_t(0.5) * BMO_MATOPS_LOG_DET(new_tensor);
    fp_t prop_U = prev_U;

    fp_t prop_K, prev_K;

    // loop

    size_t n_accept = 0;
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; draw_ind++) {
        bmo::stats::internal::rnorm_vec_inplace<fp_t>(n_vals, rand_engine, rand_vec);

        new_mntm = BMO_MATOPS_CHOL_LOWER(prev_tensor) * rand_vec;

        prev_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_prev_tensor * new_mntm) / fp_t(2);

        new_draw = prev_draw;

        for (size_t k = 0; k < n_leap_steps; ++k) {   
            // begin leap frog steps

            ColVec_t prop_mntm = new_mntm;

            for (size_t kk = 0; kk < n_fp_steps; ++kk) {
                prop_mntm = new_mntm + mntm_update_fn(new_draw,prop_mntm,target_data,step_size,inv_prev_tensor,prev_deriv_cube,nullptr); // half-step
            }

            new_mntm = prop_mntm;

            //

            ColVec_t prop_draw = new_draw;
            // new_draw += step_size * inv_prev_tensor * new_mntm;

            for (size_t kk = 0; kk < n_fp_steps; ++kk) {
                inv_new_tensor = BMO_MATOPS_INV( box_tensor_fn(prop_draw,nullptr,tensor_data) );

                prop_draw = new_draw + fp_t(0.5) * step_size * ( inv_prev_tensor + inv_new_tensor ) * new_mntm;
            }

            new_draw = prop_draw;

            new_tensor = box_tensor_fn(new_draw,&new_deriv_cube,tensor_data);
            inv_new_tensor = BMO_MATOPS_INV(new_tensor);

            //

            new_mntm += mntm_update_fn(new_draw,new_mntm,target_data,step_size,inv_new_tensor,new_deriv_cube,nullptr); // half-step
        }

        prop_U = cons_term - box_log_kernel(new_draw, nullptr, target_data) + fp_t(0.5) * BMO_MATOPS_LOG_DET(new_tensor);
        
        if (!std::isfinite(prop_U)) {
            prop_U = posinf;
        }

        prop_K = BMO_MATOPS_DOT_PROD(new_mntm, inv_new_tensor * new_mntm) / fp_t(2);

        //

        const fp_t comp_val = std::min(fp_t(0.01), - (prop_U + prop_K) + (prev_U + prev_K));
        const fp_t z = bmo::stats::runif<fp_t>(rand_engine);

        if (z < std::exp(comp_val)) {
            prev_draw = new_draw;

            prev_U = prop_U;
            prev_K = prop_K;

            prev_tensor     = new_tensor;
            inv_prev_tensor = inv_new_tensor;
            prev_deriv_cube = new_deriv_cube;

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
        for (size_t draw_ind = 0; draw_ind < n_keep_draws; draw_ind++) {
            draws_out.row(draw_ind) = inv_transform<RowVec_t>(draws_out.row(draw_ind), bounds_type, lower_bounds, upper_bounds);
        }
    }

    if (settings_inp) {
        settings_inp->rmhmc_settings.n_accept_draws = n_accept;
    }

    //

    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::rmhmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data
)
{
    return internal::rmhmc_impl(initial_vals,target_log_kernel,tensor_fn,draws_out,target_data,tensor_data,nullptr);
}

mcmclib_inline
bool
mcmc::rmhmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel,  
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data, 
    algo_settings_t& settings
)
{
    return internal::rmhmc_impl(initial_vals,target_log_kernel,tensor_fn,draws_out,target_data,tensor_data,&settings);
}
