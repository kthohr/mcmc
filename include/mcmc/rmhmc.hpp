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

 #ifndef _mcmc_rmhmc_HPP
 #define _mcmc_rmhmc_HPP
 
 /**
 * @brief The Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC) MCMC Algorithm
 *
 * @param initial_vals a column vector of initial values.
 * @param target_log_kernel the log posterior kernel function of the target distribution, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c grad_out a vector to store the gradient; and
 *   - \c target_data additional data passed to the user-provided function.
 * @param tensor_fn the manifold tensor function, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c tensor_deriv_out a 3-dimensional array to store the tensor derivatives; and
 *   - \c tensor_data additional data passed to the user-provided function.
 * @param draws_out a matrix of posterior draws, where each row represents one draw.
 * @param target_data additional data passed to the user-provided log kernel function.
 * @param tensor_data additional data passed to the user-provided tensor function.
 *
 * @return a boolean value indicating successful completion of the algorithm.
 */ 

bool
rmhmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data
);

/**
 * @brief The Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC) MCMC Algorithm
 *
 * @param initial_vals a column vector of initial values.
 * @param target_log_kernel the log posterior kernel function of the target distribution, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c grad_out a vector to store the gradient; and
 *   - \c target_data additional data passed to the user-provided function.
 * @param tensor_fn the manifold tensor function, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c tensor_deriv_out a 3-dimensional array to store the tensor derivatives; and
 *   - \c tensor_data additional data passed to the user-provided function.
 * @param draws_out a matrix of posterior draws, where each row represents one draw.
 * @param target_data additional data passed to the user-provided log kernel function.
 * @param tensor_data additional data passed to the user-provided tensor function.
 * @param settings parameters controlling the MCMC routine.
 *
 * @return a boolean value indicating successful completion of the algorithm.
 */ 

bool
rmhmc(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data,
    algo_settings_t& settings
);


namespace internal
{

bool
rmhmc_impl(
    const ColVec_t& initial_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, 
    std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, 
    Mat_t& draws_out, 
    void* target_data,
    void* tensor_data,
    algo_settings_t* settings_inp
);

}
 
 //
 
 #endif
 