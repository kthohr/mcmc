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
 * PDF of univariate and multivariate normal distributions
 */

#ifndef MCMC_STATS_INCLUDE
#define MCMC_STATS_INCLUDE

#ifndef MCMC_LOG_2PI
    #define MCMC_LOG_2PI 1.83787706640934548356L
#endif

namespace stats_mcmc
{

template<typename T>
using return_t = typename std::conditional<std::is_integral<T>::value,double,T>::type;

template<typename ...T>
using common_t = typename std::common_type<T...>::type;

template<typename ...T>
using common_return_t = return_t<common_t<T...>>;

#include "dnorm.hpp"
#include "dmvnorm.hpp"

}

#include "seed_values.hpp"

#endif
