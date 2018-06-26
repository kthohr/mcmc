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

#ifndef mcmc_structs_HPP
#define mcmc_structs_HPP

struct algo_settings_t
{
    // general
    bool vals_bound = false;

    arma::vec lower_bounds;
    arma::vec upper_bounds;

    // AEES
    int aees_n_draws = 1E04;
    int aees_n_initial_draws = 1E03;
    int aees_n_burnin = 1E03;

    arma::vec aees_temper_vec;
    double aees_prob_par = 0.10;
    int aees_n_rings = 5;

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

    // HMC
    int hmc_n_draws = 1E04;
    int hmc_n_burnin = 1E04;

    double hmc_step_size = 1.0;
    int hmc_leap_steps = 1; // number of leap frog steps
    arma::mat hmc_precond_mat;

    double hmc_accept_rate; // will be returned by the function

    // RM-HMC: HMC options + below
    int rmhmc_fp_steps = 5; // number of fixed point iteration steps

    // MALA
    int mala_n_draws = 1E04;
    int mala_n_burnin = 1E04;

    double mala_step_size = 1.0;
    arma::mat mala_precond_mat;

    double mala_accept_rate; // will be returned by the function

    // RWMH
    int rwmh_n_draws = 1E04;
    int rwmh_n_burnin = 1E04;

    double rwmh_par_scale = 1.0;
    arma::mat rwmh_cov_mat;

    double rwmh_accept_rate; // will be returned by the function
};

#endif