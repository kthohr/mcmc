/*################################################################################
  ##
  ##   Copyright (C) 2011-2023 Keith O'Hara
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

//
// scalar input

namespace internal
{

template<typename T>
constexpr
bool
is_posinf(const T x)
noexcept
{
    return x == std::numeric_limits<T>::infinity();
}

template<typename T>
constexpr
bool
is_neginf(const T x)
noexcept
{
    return x == - std::numeric_limits<T>::infinity();
}

template<typename T1, typename T2>
constexpr
bool
all_posinf(const T1 x, const T2 y)
noexcept
{
    return( is_posinf(x) && is_posinf(y) );
}

template<typename T1, typename T2>
constexpr
bool
all_neginf(const T1 x, const T2 y)
noexcept
{
    return( is_neginf(x) && is_neginf(y) );
}

template<typename T>
constexpr
bool
is_inf(const T x)
noexcept
{
    return( is_neginf(x) || is_posinf(x) );
}

template<typename T1, typename T2>
constexpr
bool
any_inf(const T1 x, const T2 y)
noexcept
{
    return( is_inf(x) || is_inf(y) );
}

//

template<typename T>
constexpr
T
dnorm_log_compute(const T z, const T sigma_par)
noexcept
{
    return( - T(0.5)*T(MCMC_LOG_2PI) - std::log(sigma_par) - z*z/T(2) );
}

template<typename T>
constexpr
T
dnorm_limit_vals(const T x, const T mu_par, const T sigma_par)
noexcept
{
    return( // sigma == Inf
            is_posinf(sigma_par) ? \
                T(0) :
            // sigma finite; x == mu == Inf or -Inf 
            all_posinf(x,mu_par) || all_neginf(x,mu_par) ? \
                std::numeric_limits<T>::quiet_NaN() :
            // sigma == 0 and x-mu == 0
            sigma_par == T(0) && x == mu_par ? \
                std::numeric_limits<T>::infinity() :
            //
                T(0) );
}

template<typename T>
constexpr
T
dnorm_vals_check(const T x, const T mu_par, const T sigma_par, const bool log_form)
noexcept
{
    return( !norm_sanity_check(x,mu_par,sigma_par) ? \
                std::numeric_limits<T>::quiet_NaN() :
            //
            any_inf(x,mu_par,sigma_par) || sigma_par == T(0) ? \
                log_if(dnorm_limit_vals(x,mu_par,sigma_par),log_form) :
            //
            exp_if(dnorm_log_compute((x-mu_par)/sigma_par,sigma_par), !log_form) );
}

template<typename T1, typename T2, typename T3, typename TC = common_return_t<T1,T2,T3>>
constexpr
TC
dnorm_type_check(const T1 x, const T2 mu_par, const T3 sigma_par, const bool log_form)
noexcept
{
    return dnorm_vals_check(static_cast<TC>(x),static_cast<TC>(mu_par),
                            static_cast<TC>(sigma_par),log_form);
}

}

template<typename T1, typename T2, typename T3>
constexpr
common_return_t<T1,T2,T3>
dnorm(const T1 x, const T2 mu_par, const T3 sigma_par, const bool log_form)
noexcept
{
    return internal::dnorm_type_check(x,mu_par,sigma_par,log_form);
}

template<typename T>
constexpr
return_t<T>
dnorm(const T x, const bool log_form)
noexcept
{
    return dnorm(x,T(0),T(1),log_form);
}

//
// matrix/vector input

namespace internal
{

inline
Mat_t
dnorm_compute(const Mat_t& x, const fp_t sigma_par, const bool log_form)
{
    const fp_t norm_term = - fp_t(0.5) * fp_t(MCMC_LOG_2PI) - std::log(sigma_par);
    Mat_t ret = BMO_MATOPS_ARRAY_ADD_SCALAR(- BMO_MATOPS_ARRAY_DIV_SCALAR( BMO_MATOPS_HADAMARD_PROD(x,x), fp_t(2) ), norm_term);

    if (!log_form) {
        return BMO_MATOPS_EXP(ret);
    }

    //
    
    return ret;
}

inline
Mat_t
dnorm_int(const Mat_t& x, const fp_t mu_par, const fp_t sigma_par, const bool log_form)
{
    return dnorm_compute( BMO_MATOPS_ARRAY_ADD_DIV_SCALARS(x,-mu_par,sigma_par), sigma_par, log_form );
}

}

inline
Mat_t
dnorm(const Mat_t& x, const bool log_form = false)
{
    return internal::dnorm_int(x,fp_t(0),fp_t(1),log_form);
}

inline
Mat_t
dnorm(const Mat_t& x, const fp_t mu_par, const fp_t sigma_par, const bool log_form)
{
    return internal::dnorm_int(x,mu_par,sigma_par,log_form);
}


#endif
