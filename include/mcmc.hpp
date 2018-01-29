/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
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

#ifndef MCMC_INCLUDES
#define MCMC_INCLUDES

#ifdef USE_RCPP_ARMADILLO
    #include <RcppArmadillo.h>
#else
    #ifndef ARMA_DONT_USE_WRAPPER
        #define ARMA_DONT_USE_WRAPPER
    #endif
    #include "armadillo"
#endif

#include "mcmc_options.hpp"
#include "stats/dnorm.hpp"
#include "stats/dmvnorm.hpp"

namespace mcmc
{
    // structs
    #include "mcmc_structs.hpp"

    // utility files
    #include "determine_bounds_type.hpp"
    #include "transform_vals.hpp"
    #include "log_jacobian.hpp"
    #include "inv_jacobian_adjust.hpp"

    // MCMC routines
    #include "rwmh.hpp"
    #include "mala.hpp"
    #include "hmc.hpp"
    #include "rmhmc.hpp"

    #include "aees.hpp"
    #include "de.hpp"
}

#endif
