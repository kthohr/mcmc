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
 * log Jacobian adjustment
 */

inline
double
log_jacobian(const arma::vec& vals_trans_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const int n_vals = bounds_type.n_elem;

    double ret_val = 0.0;

    for (int i=0; i < n_vals; i++) {
        switch (bounds_type(i)) {
            case 2: // lower bound only
                ret_val += vals_trans_inp(i);
                break;
            case 3: // upper bound only
                ret_val += - vals_trans_inp(i);
                break;
            case 4: // upper and lower bounds
                double exp_inp = std::exp(vals_trans_inp(i));
                if (std::isfinite(exp_inp))
                {
                    ret_val += std::log(upper_bounds(i) - lower_bounds(i)) + vals_trans_inp(i) - 2 * std::log(1 + exp_inp);
                }
                else
                {
                    ret_val += std::log(upper_bounds(i) - lower_bounds(i)) - vals_trans_inp(i);
                }
                break;
        }
    }
    //
    return ret_val;
}
