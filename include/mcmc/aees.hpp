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

#ifndef _mcmc_aees_HPP
#define _mcmc_aees_HPP

/**
 * @brief The Adaptive Equi-Energy MCMC Algorithm
 *
 * @param initial_vals a column vector of initial values.
 * @param target_log_kernel the log posterior kernel function of the target distribution, taking two arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c target_data additional data passed to the user-provided function.
 * @param draws_out a matrix of posterior draws, where each row represents one draw.
 * @param target_data additional data passed to the user-provided function.
 *
 * @return a boolean value indicating successful completion of the sampling algorithm.
 */

bool 
aees(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data
);

/**
 * @brief The Adaptive Equi-Energy MCMC Algorithm
 *
 * @param initial_vals a column vector of initial values.
 * @param target_log_kernel the log posterior kernel function of the target distribution, taking two arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c target_data additional data passed to the user-provided function.
 * @param draws_out a matrix of posterior draws, where each row represents one draw.
 * @param target_data additional data passed to the user-provided function.
 * @param settings parameters controlling the sampling algorithm.
 *
 * @return a boolean value indicating successful completion of the sampling algorithm.
 */

bool
aees(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t& settings
);

//
// internal

namespace internal
{

bool 
aees_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, 
    Mat_t& draws_out, 
    void* target_data, 
    algo_settings_t* settings_inp
);

#include "aees.ipp"

}

#endif
