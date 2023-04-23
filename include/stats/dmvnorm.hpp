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
 * pdf of the Multivariate Normal distribution
 */

#ifndef MCMC_STATS_DMVNORM
#define MCMC_STATS_DMVNORM

inline
fp_t
dmvnorm(const ColVec_t& X, const ColVec_t& mu_par, const Mat_t& Sigma_par, bool log_form)
{
    const size_t K = BMO_MATOPS_SIZE(X);

    //

    const fp_t cons_term = - fp_t(0.5) * K * fp_t(MCMC_LOG_2PI);
    const ColVec_t X_cent = X - mu_par; // avoids issues like Mat vs eGlue in templates

    const fp_t quad_term = BMO_MATOPS_QUAD_FORM_INV(X_cent, Sigma_par);
    
    fp_t ret = cons_term - fp_t(0.5) * ( BMO_MATOPS_LOG_DET(Sigma_par) + quad_term );

    if (!log_form) {
        ret = std::exp(ret);
        
        if (std::isinf(ret)) {
            ret = std::numeric_limits<fp_t>::max();
        }
    }

    //
    
    return ret;
}

#endif
