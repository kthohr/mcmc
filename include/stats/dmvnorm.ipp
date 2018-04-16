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

inline
double
dmvnorm_int(const arma::vec& x, const arma::vec* mu_par_inp, const arma::mat* Sigma_par_inp, const bool log_form)
{
    const int K = x.n_rows;

    const arma::vec mu_par = (mu_par_inp) ? *mu_par_inp : arma::zeros(K,1);
    const arma::mat Sigma_par = (Sigma_par_inp) ? *Sigma_par_inp : arma::eye(K,K);

    //

    const double cons_term = -0.5*K*GCEM_LOG_2PI;

    double ret = cons_term - 0.5 * ( std::log(arma::det(Sigma_par)) + arma::as_scalar((x - mu_par).t() * arma::inv(Sigma_par) * (x - mu_par)) );

    if (!log_form) {
        ret = std::exp(ret);
    }

    //
    
    return ret;
}

inline
double
dmvnorm(const arma::vec& x)
{
    return dmvnorm_int(x,nullptr,nullptr,false);
}

inline
double
dmvnorm(const arma::vec& x, const bool log_form)
{
    return dmvnorm_int(x,nullptr,nullptr,log_form);
}

inline
double
dmvnorm(const arma::vec& x, const arma::vec& mu_par, const arma::mat& Sigma_par, const bool log_form)
{
    return dmvnorm_int(x,&mu_par,&Sigma_par,log_form);
}
