/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
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
 * Random Walk Metropolis-Hastings (RWMH) MCMC
 *
 * Keith O'Hara
 * 05/01/2012
 *
 * This version:
 * 08/12/2017
 */

#ifndef _mcmc_rwmh_HPP
#define _mcmc_rwmh_HPP

bool rwmh_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, mcmc_settings* settings_inp);

bool rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data);
bool rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, mcmc_settings& settings);

#endif
