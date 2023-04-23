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
 * Differential Evolution (DE) MCMC
 */

#include "mcmc.hpp"

// [MCMC_BEGIN]
mcmclib_inline
bool
mcmc::internal::de_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Cube_t& draws_out, 
    void* target_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(initial_vals);
    
    //
    // DE settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_pop          = settings.de_settings.n_pop;
    const size_t n_keep_draws   = settings.de_settings.n_keep_draws;
    const size_t n_burnin_draws = settings.de_settings.n_burnin_draws;

    const size_t n_total_draws = n_keep_draws + n_burnin_draws;

    const bool jumps = settings.de_settings.jumps;
    const fp_t par_b = settings.de_settings.par_b;
    // const fp_t par_gamma = settings.de_par_gamma;
    const fp_t par_gamma = 2.38 / std::sqrt(fp_t(2) * n_vals);
    const fp_t par_gamma_jump = settings.de_settings.par_gamma_jump;

    const bool vals_bound = settings.vals_bound;

    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    ColVec_t par_initial_lb = ( BMO_MATOPS_SIZE(settings.de_settings.initial_lb) == n_vals ) ? settings.de_settings.initial_lb : BMO_MATOPS_ARRAY_ADD_SCALAR(initial_vals, -0.5);
    ColVec_t par_initial_ub = ( BMO_MATOPS_SIZE(settings.de_settings.initial_ub) == n_vals ) ? settings.de_settings.initial_ub : BMO_MATOPS_ARRAY_ADD_SCALAR(initial_vals,  0.5);

    sampling_bounds_check(vals_bound, n_vals, bounds_type, lower_bounds, upper_bounds, par_initial_lb, par_initial_ub);

    // parallelization setup

    int omp_n_threads = 1;

#ifdef MCMC_USE_OPENMP
    if (settings.de_settings.omp_n_threads > 0) {
        omp_n_threads = settings.de_settings.omp_n_threads;
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

    // setup

    ColVec_t rand_vec(n_vals);
    ColVec_t target_vals(n_pop);
    Mat_t X(n_pop, n_vals);

#ifdef MCMC_USE_OPENMP
    #pragma omp parallel for num_threads(omp_n_threads) firstprivate(rand_vec)
#endif
    for (size_t i = 0; i < n_pop; ++i) {
        size_t thread_num = 0;

#ifdef MCMC_USE_OPENMP
        thread_num = omp_get_thread_num();
#endif

        bmo::stats::internal::runif_vec_inplace<fp_t>(n_vals, rand_engines_vec[thread_num], rand_vec);

        X.row(i) = BMO_MATOPS_TRANSPOSE( par_initial_lb + BMO_MATOPS_HADAMARD_PROD( (par_initial_ub - par_initial_lb), rand_vec ) );

        fp_t prop_kernel_val = box_log_kernel(BMO_MATOPS_TRANSPOSE(X.row(i)), target_data);

        if (!std::isfinite(prop_kernel_val)) {
            prop_kernel_val = neginf;
        }
        
        target_vals(i) = prop_kernel_val;
    }

    // begin MCMC run

    size_t n_accept = 0;

    fp_t par_gamma_run = par_gamma;
    draws_out.setZero(n_pop, n_vals, n_keep_draws);
    
    for (size_t draw_ind = 0; draw_ind < n_total_draws; draw_ind++) {
        fp_t temperature_j = de_cooling_schedule(draw_ind,n_keep_draws);
        
        if (jumps && ((draw_ind+1) % 10 == 0)) {
            par_gamma_run = par_gamma_jump;
        }

        //

        ColVecUInt_t n_accept_vec(omp_n_threads);
        BMO_MATOPS_SET_ZERO(n_accept_vec);

#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads) firstprivate(rand_vec)
#endif
        for (size_t i = 0; i < n_pop; ++i) {
            size_t thread_num = 0;

#ifdef MCMC_USE_OPENMP
            thread_num = omp_get_thread_num();
#endif
            uint_t c_1, c_2;

            do {
                c_1 = bmo::stats::rind(0, n_pop - 1, rand_engines_vec[thread_num]);
            } while (c_1 == i);

            do {
                c_2 = bmo::stats::rind(0, n_pop - 1, rand_engines_vec[thread_num]);
            } while (c_2 == i || c_2 == c_1);

            //

            // arma::rowvec prop_rand = arma::randu(1,n_vals)*(2*par_b) - par_b; // generate a vector of U[-b,b] RVs
            bmo::stats::internal::runif_vec_inplace(n_vals, -par_b, par_b, rand_engines_vec[thread_num], rand_vec);

            RowVec_t X_prop = X.row(i) + BMO_MATOPS_ARRAY_MULT_SCALAR(X.row(c_1) - X.row(c_2), par_gamma_run) + BMO_MATOPS_TRANSPOSE(rand_vec);
                
            fp_t prop_kernel_val = box_log_kernel(BMO_MATOPS_TRANSPOSE(X_prop),target_data);
                
            if (!std::isfinite(prop_kernel_val)) {
                prop_kernel_val = neginf;
            }
                
            //
                
            fp_t comp_val = prop_kernel_val - target_vals(i);
            fp_t z = bmo::stats::runif<fp_t>(rand_engines_vec[thread_num]);

            if (comp_val > temperature_j * std::log(z)) {
                X.row(i) = X_prop;
                    
                target_vals(i) = prop_kernel_val;
                    
                if (draw_ind >= n_burnin_draws) {
                    // n_accept++; // this would lead to a race condition if parallel samlping were used
                    n_accept_vec(thread_num) += 1;
                }
            }
        }

        //

        n_accept += BMO_MATOPS_SUM(n_accept_vec);
        
        //

        if (draw_ind >= n_burnin_draws) {
            draws_out.mat(draw_ind - n_burnin_draws) = X;
        }
        
        if (jumps && ((draw_ind + 1) % 10 == 0)) {
            par_gamma_run = par_gamma;
        }
    }

    success = true;

    //
    
    if (vals_bound) {
#ifdef MCMC_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
        for (size_t draw_ind = 0; draw_ind < n_keep_draws; draw_ind++) {
            for (size_t jj = 0; jj < n_pop; jj++) {
                draws_out.mat(draw_ind).row(jj) = inv_transform<RowVec_t>(draws_out.mat(draw_ind).row(jj), bounds_type, lower_bounds, upper_bounds);
            }
        }
    }
    
    if (settings_inp) {
        settings_inp->de_settings.n_accept_draws = n_accept;
    }

    //
    
    return success;
}

// wrappers

mcmclib_inline
bool
mcmc::de(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Cube_t& draws_out, 
    void* target_data
)
{
    return internal::de_impl(initial_vals,target_log_kernel,draws_out,target_data,nullptr);
}

mcmclib_inline
bool
mcmc::de(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Cube_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
)
{
    return internal::de_impl(initial_vals,target_log_kernel,draws_out,target_data,&settings);
}
