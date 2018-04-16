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
 * pdf of the Multivariate Normal distribution
 */

#ifndef _statslib_dmvnorm_HPP
#define _statslib_dmvnorm_HPP

#ifndef GCEM_LOG_2PI
    #define GCEM_LOG_2PI 1.83787706640934548356L
#endif

namespace stats_mcmc {

double dmvnorm_int(const arma::vec& x, const arma::vec* mu_par_inp, const arma::mat* Sigma_par_inp, const bool log_form);

double dmvnorm(const arma::vec& x);
double dmvnorm(const arma::vec& x, const bool log_form);
double dmvnorm(const arma::vec& x, const arma::vec& mu_par, const arma::mat& Sigma_par, const bool log_form = false);

#include "dmvnorm.ipp"

}

#endif
