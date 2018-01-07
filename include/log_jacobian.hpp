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
                ret_val += std::log(upper_bounds(i) - lower_bounds(i)) + vals_trans_inp(i) - 2 * std::log(1 + std::exp(vals_trans_inp(i)));
                break;
        }
    }
    //
    return ret_val;
}
