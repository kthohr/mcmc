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
 * Adaptive Equi-Energy Sampler
 */

#ifndef _mcmc_aees_HPP
#define _mcmc_aees_HPP

bool aees_int(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp);

bool aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data);
bool aees(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings);

// single-step Metropolis-Hastings for tempered distributions
inline
arma::vec
single_step_mh(const arma::vec& X_prev, const double temper_val, const arma::mat& sqrt_cov_mcmc,
               std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, 
               void* target_data, double* val_out)
{
    const int n_vals = X_prev.n_elem;

    arma::vec X_new = X_prev + std::sqrt(temper_val)*sqrt_cov_mcmc*arma::randn(n_vals,1);

    //

    double val_new  = target_log_kernel(X_new, target_data);
    double val_prev = target_log_kernel(X_prev, target_data);

    double comp_val = std::min(0.0, (val_new - val_prev) / temper_val );

    double z = arma::as_scalar(arma::randu(1,1));
    
    if (z < std::exp(comp_val))
    {
        if (val_out) {
            *val_out = val_new;
        }

        return X_new;
    }
    else
    {
        if (val_out) {
            *val_out = val_prev;
        }

        return X_prev;
    }
}

#endif
