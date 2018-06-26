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
 * Differential Evolution (DE) MCMC
 */

#ifndef _mcmc_de_HPP
#define _mcmc_de_HPP

bool de_int(const arma::vec& initial_vals, arma::cube& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t* settings_inp);

bool de(const arma::vec& initial_vals, arma::cube& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data);
bool de(const arma::vec& initial_vals, arma::cube& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data, algo_settings_t& settings);

inline
double
de_cooling_schedule(const int s, const int N_gen)
{
    return 1.0;
}

#endif
