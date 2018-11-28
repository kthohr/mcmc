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

#pragma once

// version

#ifndef MCMC_VERSION_MAJOR
    #define MCMC_VERSION_MAJOR 0
#endif

#ifndef MCMC_VERSION_MINOR
    #define MCMC_VERSION_MINOR 10
#endif

#ifndef MCMC_VERSION_PATCH
    #define MCMC_VERSION_PATCH 1
#endif

//

#if defined(_OPENMP) && !defined(MCMC_DONT_USE_OPENMP)
    #undef MCMC_USE_OPENMP
    #define MCMC_USE_OPENMP
#endif

#if !defined(_OPENMP) && defined(MCMC_USE_OPENMP)
    #undef MCMC_USE_OPENMP

    #undef MCMC_DONE_USE_OPENMP
    #define MCMC_DONE_USE_OPENMP
#endif

#ifdef MCMC_USE_OPENMP
    // #include "omp.h" //  OpenMP
    #ifndef ARMA_USE_OPENMP
        #define ARMA_USE_OPENMP
    #endif
#endif

#ifdef MCMC_DONT_USE_OPENMP
    #ifdef MCMC_USE_OPENMP
        #undef MCMC_USE_OPENMP
    #endif

    #ifndef ARMA_DONT_USE_OPENMP
        #define ARMA_DONT_USE_OPENMP
    #endif
#endif

//

#ifdef USE_RCPP_ARMADILLO
    #include <RcppArmadillo.h>
#else
    #ifndef ARMA_DONT_USE_WRAPPER
        #define ARMA_DONT_USE_WRAPPER
    #endif
    #include "armadillo"
#endif

// typedefs

namespace mcmc
{
    static const double eps_dbl = std::numeric_limits<double>::epsilon();
    static const double inf  = std::numeric_limits<double>::infinity();
    static const double neginf = - std::numeric_limits<double>::infinity();
    using uint_t = unsigned int;
}
