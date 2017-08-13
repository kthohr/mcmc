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
 * simple normal model
 *
 * Keith O'Hara
 * 08/12/2017
 *
 * This version:
 * 08/12/2017
 */

// g++-mp-7 -std=c++11 -Wall -O3 -I/opt/local/include -I./../include rwmh_normal.cpp -o rwmh_normal.test -L./../ -lmcmc -framework Accelerate

#include "mcmc.hpp"

struct norm_data {
    double sigma;
    arma::vec x;

    double mu_0;
    double sigma_0;
};

double ll_dens(const arma::vec& vals_inp, void* ll_data)
{
    const double mu = vals_inp(0);
    const double pi = arma::datum::pi;

    norm_data* dta = reinterpret_cast<norm_data*>(ll_data);
    const double sigma = dta->sigma;
    const arma::vec x = dta->x;

    const double ret = - 0.5*std::log(2*pi) - std::log(sigma) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );

    return ret;
}

double log_pr_dens(const arma::vec& vals_inp, void* ll_data)
{
    norm_data* dta = reinterpret_cast<norm_data*>(ll_data);

    const double mu_0 = dta->mu_0;
    const double sigma_0 = dta->sigma_0;
    const double pi = arma::datum::pi;

    const double x = vals_inp(0);

    const double ret = - 0.5*std::log(2*pi) - std::log(sigma_0) - std::pow(x - mu_0,2) / (2*sigma_0*sigma_0);

    return ret;
}

double log_target_dens(const arma::vec& vals_inp, void* ll_data)
{
    return ll_dens(vals_inp,ll_data) + log_pr_dens(vals_inp,ll_data);
}

int main()
{
    const int n_data = 100;
    const double mu = 2.0;

    norm_data dta;
    dta.sigma = 1.0;
    dta.mu_0 = 1.0;
    dta.sigma_0 = 2.0;

    arma::vec x_dta = mu + arma::randn(n_data,1);
    dta.x = x_dta;

    arma::vec initial_val(1);
    initial_val(0) = 1.0;

    arma::mat draws_out;
    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta);

    //
    //

    mcmc::mcmc_settings settings;

    arma::vec lb(1);
    lb(0) = 0.0;

    arma::vec ub(1);
    ub(0) = 4.0;

    settings.vals_bound = true;
    settings.lower_bounds = lb;
    settings.upper_bounds = ub;

    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta,settings);

    //

    lb(0) = -arma::datum::inf;
    settings.lower_bounds = lb;

    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta,settings);

    //

    ub(0) = arma::datum::inf;
    settings.upper_bounds = ub;
    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta,settings);

    return 0;
}
