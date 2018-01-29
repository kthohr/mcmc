/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
  ##
  ##   This file is part of the MCMC C++ library.
  ##
  ##   MCMC is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   MCMC is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/
 
/*
 * Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC)
 */

 #ifndef _mcmc_rmhmc_HPP
 #define _mcmc_rmhmc_HPP
 
 bool rmhmc_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data, algo_settings* settings_inp);
 
 bool rmhmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data);
 bool rmhmc(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* target_data)> target_log_kernel, void* target_data, std::function<arma::mat (const arma::vec& vals_inp, arma::cube* tensor_deriv_out, void* tensor_data)> tensor_fn, void* tensor_data, algo_settings& settings);
 
 //
 
 #endif
 