.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

MCMC Settings
=============

.. contents:: :local:

----

Main
----

An object of type ``algo_settings_t`` can be used to control the behavior of the MCMC routines. Each algorithm page details the relevant parameters for that methods, but we list the full settings here for completeness.

.. code:: cpp

    struct algo_settings_t
    {
        // RNG seeding

        size_t rng_seed_value = std::random_device{}();

        // bounds 
        
        bool vals_bound = false;

        ColVec_t lower_bounds;
        ColVec_t upper_bounds;

        // AEES
        aees_settings_t aees_settings;

        // DE
        de_settings_t de_settings;

        // HMC
        hmc_settings_t hmc_settings;

        // RM-HMC
        rmhmc_settings_t rmhmc_settings;

        // MALA
        mala_settings_t mala_settings;

        // RWMH
        rwmh_settings_t rwmh_settings;
    };


Description:

- ``rng_seed_value`` seed value used for random number generators.

- ``vals_bound`` whether the search space of the algorithm is bounded.

- ``lower_bounds`` defines the lower bounds of the search space.

- ``upper_bounds`` defines the upper bounds of the search space.

Algorithm-specific data structures are listed in the next section.

----

By Algorithm
------------

AEES
~~~~

.. code:: cpp

    struct aees_settings_t
    {
        size_t n_initial_draws = 1E03;
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        fp_t par_scale = 1.0;
        Mat_t cov_mat;

        size_t n_rings = 5; // number of energy rings
        fp_t ee_prob_par = 0.10; // equi-energy probability parameter
        ColVec_t temper_vec; // temperature vector
    };

DE
~~

.. code:: cpp

    struct de_settings_t
    {
        bool jumps = false;

        size_t n_pop = 100;
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        fp_t par_b = 1E-04;
        fp_t par_gamma = 1.0;
        fp_t par_gamma_jump = 2.0;

        ColVec_t initial_lb; // this will default to -0.5
        ColVec_t initial_ub; // this will default to  0.5

        size_t n_accept_draws; // will be returned by the algorithm
    };


HMC
~~~

.. code:: cpp

    struct hmc_settings_t
    {
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        size_t n_leap_steps = 1; // number of leap frog steps
        fp_t step_size = 1.0;
        Mat_t precond_mat;

        size_t n_accept_draws; // will be returned by the function
    };


RM-HMC
~~~~~~

.. code:: cpp

    struct rmhmc_settings_t
    {
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        size_t n_leap_steps = 1; // number of leap frog steps
        fp_t step_size = 1.0;
        Mat_t precond_mat;

        size_t n_fp_steps = 5; // number of fixed point iteration steps

        size_t n_accept_draws; // will be returned by the function
    };


MALA
~~~~

.. code:: cpp

    struct mala_settings_t
    {
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        fp_t step_size = 1.0;
        Mat_t precond_mat;

        size_t n_accept_draws; // will be returned by the function
    };

RWMH
~~~~

.. code:: cpp

    struct rwmh_settings_t
    {
        size_t n_burnin_draws = 1E03;
        size_t n_keep_draws = 1E03;

        int omp_n_threads = -1; // numbers of threads to use

        fp_t par_scale = 1.0;
        Mat_t cov_mat;

        size_t n_accept_draws; // will be returned by the function
    };

----
