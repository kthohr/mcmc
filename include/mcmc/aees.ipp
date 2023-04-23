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

#ifndef _mcmc_aees_IPP
#define _mcmc_aees_IPP

// single-step Metropolis-Hastings for tempered distributions

inline
ColVec_t
single_step_mh(
    const ColVec_t& X_prev, 
    const fp_t temper_val, 
    const Mat_t& prop_scaling_mat,
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    void* target_data, 
    rand_engine_t& rand_engine,
    fp_t* val_out
)
{
    const size_t n_vals = BMO_MATOPS_SIZE(X_prev);

    const ColVec_t rand_vec = bmo::stats::rnorm_vec<fp_t>(n_vals, rand_engine);

    const ColVec_t X_new = X_prev + std::sqrt(temper_val) * prop_scaling_mat * rand_vec;

    //

    const fp_t val_new  = target_log_kernel(X_new, target_data);
    const fp_t val_prev = target_log_kernel(X_prev, target_data);

    const fp_t comp_val = std::min(fp_t(0.01), (val_new - val_prev) / temper_val );

    const fp_t z = bmo::stats::runif<fp_t>(rand_engine);
    
    if (z < std::exp(comp_val)) {
        if (val_out) {
            *val_out = val_new;
        }

        return X_new;
    } else {
        if (val_out) {
            *val_out = val_prev;
        }

        return X_prev;
    }
}

#endif
