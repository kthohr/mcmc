/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
  ##
  ##   This file is part of the StatsLib C++ library.
  ##
  ##   StatsLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   StatsLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
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
arma::mat dnorm(const arma::mat& x, const double mu_par, const double sigma_par);
arma::mat dnorm(const arma::mat& x, const double mu_par, const double sigma_par, const bool log_form);

#include "dnorm.ipp"

}

#endif
