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
 * MALA with normal model
 */

#include "mcmc.hpp"

struct norm_data {
    arma::vec x;
};

double ll_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
{
    const double mu    = vals_inp(0);
    const double sigma = vals_inp(1);

    const double pi = arma::datum::pi;

    norm_data* dta = reinterpret_cast<norm_data*>(ll_data);
    const arma::vec x = dta->x;
    const int n_vals = x.n_rows;

    //

    const double ret = - ((double) n_vals) * (0.5*std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );

    //

    if (grad_out) {
        grad_out->set_size(2,1);

        //

        double m_1 = arma::accu(x - mu);
        double m_2 = arma::accu( arma::pow(x - mu,2) );

        (*grad_out)(0,0) = m_1 / (sigma*sigma);
        (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - ((double) n_vals) / sigma;
    }

    //

    return ret;
}

double log_target_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
{
    return ll_dens(vals_inp,grad_out,ll_data);
}

int main()
{
    const int n_data = 100;
    const double mu = 1.0;
    const double sigma = 10.0;

    norm_data dta;

    arma::vec x_dta = mu + sigma*arma::randn(n_data,1);
    dta.x = x_dta;

    arma::vec initial_val(2);
    initial_val(0) = mu + 1; // mu
    initial_val(1) = sigma + 1; // sigma

    mcmc::mcmc_settings settings;

    settings.mala_step_size = 0.75;
    settings.mala_n_burnin = 1000;
    settings.mala_n_draws = 1000;

    printf("begin run\n");
    arma::mat draws_out;
    mcmc::mala(initial_val,draws_out,log_target_dens,&dta,settings);

    arma::cout << "draws:\n" << draws_out.rows(0,9) << arma::endl;

    std::cout << "acceptance rate = " << settings.mala_accept_rate << arma::endl;

    std::cout << "mcmc mean = " << arma::mean(draws_out) << std::endl;

    //
    //

    // mcmc::mcmc_settings settings;

    // arma::vec lb(1);
    // lb(0) = 0.0;

    // arma::vec ub(1);
    // ub(0) = 4.0;

    // settings.vals_bound = true;
    // settings.lower_bounds = lb;
    // settings.upper_bounds = ub;

    // mcmc::mala(initial_val,draws_out,log_target_dens,&dta,settings);

    // //

    // lb(0) = -arma::datum::inf;
    // settings.lower_bounds = lb;

    // mcmc::mala(initial_val,draws_out,log_target_dens,&dta,settings);

    // //

    // ub(0) = arma::datum::inf;
    // settings.upper_bounds = ub;
    
    // mcmc::mala(initial_val,draws_out,log_target_dens,&dta,settings);

    return 0;
}
