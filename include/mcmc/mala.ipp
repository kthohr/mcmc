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
 * Metropolis-adjusted Langevin algorithm
 */

#ifndef _mcmc_mala_IPP
#define _mcmc_mala_IPP

// 

inline
fp_t
mala_prop_adjustment(
    const ColVec_t& prop_vals, 
    const ColVec_t& prev_vals, 
    const fp_t step_size, 
    const bool vals_bound, 
    const Mat_t& precond_mat, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* target_data, const fp_t step_size, Mat_t* jacob_matrix_out)> mala_mean_fn, 
    void* target_data
)
{

    fp_t ret_val = 0;
    const fp_t step_size_sq = step_size * step_size;

    //

    if (vals_bound) 
    {
        Mat_t prop_inv_jacob, prev_inv_jacob;

        ColVec_t prop_mean = mala_mean_fn(prop_vals, target_data, step_size, &prop_inv_jacob);
        ColVec_t prev_mean = mala_mean_fn(prev_vals, target_data, step_size, &prev_inv_jacob);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*prop_inv_jacob*precond_mat, true) \
                  - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*prop_inv_jacob*precond_mat, true);
    }
    else
    {
        ColVec_t prop_mean = mala_mean_fn(prop_vals, target_data, step_size, nullptr);
        ColVec_t prev_mean = mala_mean_fn(prev_vals, target_data, step_size, nullptr);

        ret_val = stats_mcmc::dmvnorm(prev_vals, prop_mean, step_size_sq*precond_mat, true) \
                  - stats_mcmc::dmvnorm(prop_vals, prev_mean, step_size_sq*precond_mat, true);
    }

    //

    return ret_val;
}

#endif
