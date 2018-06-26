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
 * RWMH with simple normal model
 */

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

    const int n_vals = x.n_rows;

    //

    const double ret = - static_cast<double>(n_vals) * (0.5*std::log(2*pi) + std::log(sigma)) \
                       - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );

    //

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

    mcmc::algo_settings_t settings;
    settings.rwmh_par_scale = 0.5;

    arma::vec initial_val(1);
    initial_val(0) = 1.0;

    arma::vec lb(1);
    // lb(0) = -arma::datum::inf;
    lb(0) = 0.0;

    arma::vec ub(1);
    ub(0) = 4.0;
    // ub(0) = arma::datum::inf;

    settings.vals_bound = true;
    settings.lower_bounds = lb;
    settings.upper_bounds = ub;

    arma::mat draws_out;
    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta,settings);

    arma::cout << "draws:\n" << draws_out.rows(0,9) << arma::endl;

    std::cout << "acceptance rate = " << settings.rwmh_accept_rate << arma::endl;

    std::cout << "mcmc mean = " << arma::mean(draws_out) << std::endl;

    return 0;
}
