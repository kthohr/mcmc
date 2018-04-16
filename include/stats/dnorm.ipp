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

//
// single input

template<typename T>
T
dnorm_int(const T z, const T sigma_par)
{
    return ( - 0.5*GCEM_LOG_2PI - std::log(sigma_par) - z*z/2.0 );
}

template<typename T>
T
dnorm(const T x, const T mu_par, const T sigma_par, const bool log_form)
{
    return ( log_form == true ? dnorm_int((x-mu_par)/sigma_par,sigma_par) : std::exp(dnorm_int((x-mu_par)/sigma_par,sigma_par)) );
}

inline
double
dnorm(const double x)
{
    return dnorm(x,0.0,1.0,false);
}

inline
double
dnorm(const double x, const bool log_form)
{
    return dnorm(x,0.0,1.0,log_form);
}

inline
double
dnorm(const double x, const double mu_par, const double sigma_par)
{
    return dnorm(x,mu_par,sigma_par,false);
}

//
// matrix/vector input

inline
arma::mat
dnorm_int(const arma::mat& x, const double* mu_par_inp, const double* sigma_par_inp, const bool log_form)
{
    const double mu_par = (mu_par_inp) ? *mu_par_inp : 0.0;
    const double sigma_par = (sigma_par_inp) ? *sigma_par_inp : 1.0;

    //

    const double norm_term = - 0.5*GCEM_LOG_2PI - std::log(sigma_par);
    arma::mat ret = norm_term - (x - mu_par)%(x - mu_par)  / (2 * sigma_par*sigma_par);

    if (!log_form) {
        ret = arma::exp(ret);
    }

    //
    
    return ret;
}

inline
arma::mat
dnorm(const arma::mat& x)
{
    return dnorm_int(x,nullptr,nullptr,false);
}

inline
arma::mat
dnorm(const arma::mat& x, const bool log_form)
{
    return dnorm_int(x,nullptr,nullptr,log_form);
}

inline
arma::mat
dnorm(const arma::mat& x, const double mu_par, const double sigma_par, const bool log_form)
{
    return dnorm_int(x,&mu_par,&sigma_par,log_form);
}
