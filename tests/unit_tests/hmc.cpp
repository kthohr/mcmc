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
 * HMC with normal model
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
 
     const double ret = - static_cast<double>(n_vals) * (0.5*std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );
 
     //
 
     if (grad_out) {
         grad_out->set_size(2,1);
 
         //
 
         double m_1 = arma::accu(x - mu);
         double m_2 = arma::accu( arma::pow(x - mu,2) );
 
         (*grad_out)(0,0) = m_1 / (sigma*sigma);
         (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - static_cast<double>(n_vals) / sigma;
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
     const int n_data = 1000;
     const double mu = 2.0;
     const double sigma = 2.0;
 
     norm_data dta;
 
     arma::vec x_dta = mu + sigma*arma::randn(n_data,1);
     dta.x = x_dta;
 
     arma::vec initial_val(2);
     initial_val(0) = mu + 1; // mu
     initial_val(1) = sigma + 1; // sigma
 
     mcmc::algo_settings_t settings;
 
     settings.hmc_step_size = 0.10;
     settings.hmc_n_burnin = 1000;
     settings.hmc_n_draws = 1000;
 
     arma::mat draws_out;
     mcmc::hmc(initial_val,draws_out,log_target_dens,&dta,settings);
 
     std::cout << "hmc: mcmc mean = " << arma::mean(draws_out) << std::endl;
 
     //
     //
 
     arma::vec lb(2);
     lb(0) = 0.0;
     lb(1) = 0.0;
 
     arma::vec ub(2);
     ub(0) = 4.0;
     ub(1) = arma::datum::inf;
 
     settings.vals_bound = true;
     settings.lower_bounds = lb;
     settings.upper_bounds = ub;
 
     settings.hmc_n_burnin = 5000;
     settings.hmc_n_draws = 5000;
 
     mcmc::hmc(initial_val,draws_out,log_target_dens,&dta,settings);
 
     std::cout << "hmc: mcmc mean = " << arma::mean(draws_out) << std::endl;
 
     //
 
     lb(0) = -arma::datum::inf;
     settings.lower_bounds = lb;
 
     mcmc::hmc(initial_val,draws_out,log_target_dens,&dta,settings);
 
     std::cout << "hmc: mcmc mean = " << arma::mean(draws_out) << std::endl;
 
     //
 
     ub(0) = arma::datum::inf;
     settings.upper_bounds = ub;
     
     mcmc::hmc(initial_val,draws_out,log_target_dens,&dta,settings);
 
     std::cout << "hmc: mcmc mean = " << arma::mean(draws_out) << std::endl;
 
     return 0;
 }
 