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
 * inverse Jacobian adjustment
 */

inline
Mat_t
inv_jacobian_adjust(
    const ColVec_t& vals_trans_inp, 
    const ColVecInt_t& bounds_type, 
    const ColVec_t& lower_bounds, 
    const ColVec_t& upper_bounds
)
{
    const size_t n_vals = BMO_MATOPS_SIZE(bounds_type);

    Mat_t ret_mat = BMO_MATOPS_EYE(n_vals);

    for (size_t i = 0; i < n_vals; ++i) {
        switch (bounds_type(i)) {
            case 2: // lower bound only
                // ret_mat(i,i) = 1 / std::exp(vals_trans_inp(i));
                ret_mat(i,i) = std::exp(-vals_trans_inp(i));
                break;
            case 3: // upper bound only
                // ret_mat(i,i) = 1 / std::exp(-vals_trans_inp(i));
                ret_mat(i,i) = std::exp(vals_trans_inp(i));
                break;
            case 4: // upper and lower bounds
                const fp_t exp_inp = std::exp(vals_trans_inp(i));
                ret_mat(i,i) = ( (exp_inp + 1) * (exp_inp + 1) ) / ( exp_inp * (upper_bounds(i) - lower_bounds(i)) );
                break;
        }
    }

    return ret_mat;
}
