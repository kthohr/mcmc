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
 * pdf of the univariate normal distribution
 */

#ifndef _statsmcmc_dnorm_HPP
#define _statsmcmc_dnorm_HPP

#ifndef GCEM_LOG_2PI
    #define GCEM_LOG_2PI 1.83787706640934548356L
#endif

namespace stats_mcmc {

// single input
template<typename T>
T dnorm(const T x, const T mu_par, const T sigma_par, const bool log_form);

double dnorm(const double x);
double dnorm(const double x, const bool log_form);
double dnorm(const double x, const double mu_par, const double sigma_par);

// matrix/vector input
arma::mat dnorm_int(const arma::mat& x, const double* mu_par_inp, const double* sigma_par_inp, const bool log_form);

arma::mat dnorm(const arma::mat& x);
arma::mat dnorm(const arma::mat& x, const bool log_form);
arma::mat dnorm(const arma::mat& x, const double mu_par, const double sigma_par, const bool log_form = false);

#include "dnorm.ipp"

}

#endif
