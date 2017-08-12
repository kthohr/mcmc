/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
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

#ifndef mcmc_structs_HPP
#define mcmc_structs_HPP

struct mcmc_settings {
    // general
    bool vals_bound = false;

    arma::vec lower_bounds;
    arma::vec upper_bounds;

    // DE
    bool de_jumps = false;

    int de_n_pop = 100;
    int de_n_gen = 1000;
    int de_n_burnin = 1000;

    double de_par_b = 1E-04;
    double de_par_gamma = 1.0;
    double de_par_gamma_jump = 2.0;

    arma::vec de_initial_lb;
    arma::vec de_initial_ub;

    double de_accept_rate; // will be returned by the function

    // RWMH
    int rwmh_n_draws_keep = 1E04;
    int rwmh_n_draws_burnin = 1E04;

    double rwmh_par_scale = 1.0;
    arma::mat rwmh_cov_mat;

    double rwmh_accept_rate; // will be returned by the function
};

#endif