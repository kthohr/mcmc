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

#ifndef MCMC_INCLUDES
#define MCMC_INCLUDES

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
