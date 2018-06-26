/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
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
 
 bool rmhmc_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data, algo_settings_t* settings_inp);
 
 bool rmhmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data);
 bool rmhmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data, algo_settings_t& settings);
 
 //
 
 #endif
 