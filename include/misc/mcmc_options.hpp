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

#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

// version

#ifndef MCMC_VERSION_MAJOR
    #define MCMC_VERSION_MAJOR 2
#endif

#ifndef MCMC_VERSION_MINOR
    #define MCMC_VERSION_MINOR 1
#endif

#ifndef MCMC_VERSION_PATCH
    #define MCMC_VERSION_PATCH 0
#endif

//

#ifdef _MSC_VER
    #error MCMCLib: MSVC is not supported
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

// #ifdef MCMC_USE_OPENMP
    // #include "omp.h" //  OpenMP
// #endif

#ifdef MCMC_DONT_USE_OPENMP
    #ifdef MCMC_USE_OPENMP
        #undef MCMC_USE_OPENMP
    #endif
#endif

//

#ifndef mcmclib_inline
    #define mcmclib_inline 
#endif

// floating point number type

#ifndef MCMC_FPN_TYPE
    #define MCMC_FPN_TYPE double
#endif

#if MCMC_FPN_TYPE == float
    #undef MCMC_FPN_SMALL_NUMBER
    #define MCMC_FPN_SMALL_NUMBER fp_t(1e-05)
#elif MCMC_FPN_TYPE == double
    #undef MCMC_FPN_SMALL_NUMBER
    #define MCMC_FPN_SMALL_NUMBER fp_t(1e-08)
#else
    #error floating-point number type must be 'float' or 'double'
#endif

//

namespace mcmc
{
    using uint_t = unsigned int;
    using fp_t = MCMC_FPN_TYPE;

    using rand_engine_t = std::mt19937_64;

    static const double eps_dbl = std::numeric_limits<fp_t>::epsilon();
    static const double posinf  = std::numeric_limits<fp_t>::infinity();
    static const double neginf  = - std::numeric_limits<fp_t>::infinity();
}

//

#ifdef MCMC_ENABLE_ARMA_WRAPPERS
    #ifdef USE_RCPP_ARMADILLO
        #include <RcppArmadillo.h>
    #else
        #ifndef ARMA_DONT_USE_WRAPPER
            #define ARMA_DONT_USE_WRAPPER
        #endif
        
        #include "armadillo"
    #endif

    #ifdef MCMC_USE_OPENMP
        #ifndef ARMA_USE_OPENMP
            #define ARMA_USE_OPENMP
        #endif
    #endif

    #ifdef MCMC_DONT_USE_OPENMP
        #ifndef ARMA_DONT_USE_OPENMP
            #define ARMA_DONT_USE_OPENMP
        #endif
    #endif

    #ifndef BMO_ENABLE_ARMA_WRAPPERS
        #define BMO_ENABLE_ARMA_WRAPPERS
    #endif

    namespace mcmc
    {
        using ColVec_t = arma::Col<fp_t>;
        using RowVec_t = arma::Row<fp_t>;
        using ColVecInt_t = arma::Col<int>;
        using RowVecInt_t = arma::Row<int>;
        using ColVecUInt_t = arma::Col<unsigned long long>;

        using Mat_t = arma::Mat<fp_t>;
    }
#elif defined MCMC_ENABLE_EIGEN_WRAPPERS
    #include <iostream>
    #include <random>
    #include <Eigen/Dense>

    #ifndef BMO_ENABLE_EIGEN_WRAPPERS
        #define BMO_ENABLE_EIGEN_WRAPPERS
    #endif

    // template<typename eT, int iTr, int iTc>
    // using EigenMat = Eigen::Matrix<eT,iTr,iTc>;

    namespace mcmc
    {
        using ColVec_t = Eigen::Matrix<fp_t, Eigen::Dynamic, 1>;
        using RowVec_t = Eigen::Matrix<fp_t, 1, Eigen::Dynamic>;
        using ColVecInt_t = Eigen::Matrix<int, Eigen::Dynamic, 1>;
        using RowVecInt_t = Eigen::Matrix<int, 1, Eigen::Dynamic>;
        using ColVecUInt_t = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

        using Mat_t = Eigen::Matrix<fp_t, Eigen::Dynamic, Eigen::Dynamic>;
    }
#else
    #error MCMCLib: you must enable the Armadillo OR Eigen wrappers
#endif

//

#ifndef BMO_ENABLE_EXTRA_FEATURES
    #define BMO_ENABLE_EXTRA_FEATURES
#endif

#ifndef BMO_ENABLE_EXTRA_EXPERIMENTAL
    #define BMO_ENABLE_EXTRA_EXPERIMENTAL
#endif

#ifndef BMO_ENABLE_STATS_FEATURES
    #define BMO_ENABLE_STATS_FEATURES
#endif

#ifndef BMO_RNG_ENGINE_TYPE
    #define BMO_RNG_ENGINE_TYPE mcmc::rand_engine_t
#endif

#ifndef BMO_CORE_TYPES
    #define BMO_CORE_TYPES

    namespace bmo
    {
        using fp_t = MCMC_FPN_TYPE;

        using ColVec_t = mcmc::ColVec_t;
        using RowVec_t = mcmc::RowVec_t;
        using ColVecInt_t = mcmc::ColVecInt_t;
        using RowVecInt_t = mcmc::RowVecInt_t;
        using ColVecUInt_t = mcmc::ColVecUInt_t;

        using Mat_t = mcmc::Mat_t;
    }
#endif

#include "BaseMatrixOps/include/BaseMatrixOps.hpp"

namespace mcmc
{
    using Cube_t = bmo::Cube_t<fp_t>;
}

