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
 * Adaptive Equi-Energy Sampler
 */

#ifndef _mcmc_aees_HPP
#define _mcmc_aees_HPP

bool aees_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings* settings_inp);

bool aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data);
bool aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings& settings);

// single-step Metropolis-Hastings for tempered distributions
inline
arma::vec
single_step_mh(const arma::vec& X_prev, const double temper_val, const arma::mat& sqrt_cov_mcmc, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, double* val_out)
{
    const int n_vals = X_prev.n_elem;

    arma::vec X_new = X_prev + std::sqrt(temper_val)*sqrt_cov_mcmc*arma::randn(n_vals,1);

    //

    double val_new  = target_log_kernel(X_new, target_data);
    double val_prev = target_log_kernel(X_prev, target_data);

    double comp_val = std::min(0.0, (val_new - val_prev) / temper_val );

    double z = arma::as_scalar(arma::randu(1,1));
    
    if (z < std::exp(comp_val)){
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
