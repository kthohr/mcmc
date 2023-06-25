.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _differential-evolution:

Differential Evolution
======================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Differential Evolution (DE) sampler is a Markov Chain Monte Carlo variant of the well-known stochastic genetic search algorithm used for global optimization. See Cajo J. F. Ter Braak (2006) for details.

Let :math:`\boldsymbol{\theta}_k^{(i)}` denote a :math:`N_p \times d`-dimensional array of stored values at stage :math:`i` of the algorithm, and let :math:`\gamma = 2.38 / \sqrt{2 d}`. The DE-MCMC algorithm proceeds in two steps.

1. (**Mutation Step**) For random and unique indices :math:`j,k`, set:

  .. math::

    \theta^{(*)} = \theta^{(i)} + \gamma \times (\boldsymbol{\theta}^{(i)}(j,:) - \boldsymbol{\theta}^{(i)}(k,:))) + U

  where :math:`U \sim \text{Unif}[-b,b]` and :math:`b` is set via ``de_settings.par_b``.

2. (**Accept/Reject Step**) Define

  .. math::

    \alpha = \min \left\{ 1, K(\theta^{(*)} | X) / K(\theta^{(i)} | X) \right\}

  where :math:`K` denotes the posterior kernel. Then

  .. math::

    \theta^{(i+1)} = \begin{cases} \theta^{(*)} & \text{ with probability } \alpha \\ \theta^{(i)} & \text{ else } \end{cases}

The algorithm stops when the number of draws reaches ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws (in the form a three-dimensional array).

----

Function Declarations
---------------------

.. _de-func-ref1:
.. doxygenfunction:: de(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, void *target_data)> target_log_kernel, Cube_t& draws_out, void *target_data)
   :project: mcmclib

.. _de-func-ref2:
.. doxygenfunction:: de(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, void *target_data)> target_log_kernel, Cube_t& draws_out, void *target_data, algo_settings_t& algo_settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t de_settings.n_pop``: size of population for each set of draws.

  - Default value: ``100``.

- ``size_t de_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t de_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- Upper and lower bounds of the uniform distributions used to generate initial values:

  - ``ColVec_t de_settings.initial_lb``: defines the lower bounds of the search space.

  - ``ColVec_t de_settings.initial_ub``: defines the upper bounds of the search space.

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int de_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``fp_t de_settings.par_b``: support parameter of the uniform distribution used in the Mutation Step.

  - Default value: ``1E-04``.

----

Examples
--------

Gaussian Mean
~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define MCMC_ENABLE_ARMA_WRAPPERS
        #include "mcmc.hpp"

        struct norm_data_t {
            double sigma;
            arma::vec x;
        
            double mu_0;
            double sigma_0;
        };
        
        double ll_dens(const arma::vec& vals_inp, void* ll_data)
        {
            const double pi = arma::datum::pi;

            //

            const double mu = vals_inp(0);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
            const double sigma = dta->sigma;
            const arma::vec x = dta->x;
        
            const int n_vals = x.n_rows;
        
            //
        
            const double ret = - ((double) n_vals) * (0.5*std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );
        
            //
        
            return ret;
        }
        
        double log_pr_dens(const arma::vec& vals_inp, void* ll_data)
        {
            const double pi = arma::datum::pi;

            //

            norm_data_t* dta = reinterpret_cast< norm_data_t* >(ll_data);
        
            const double mu_0 = dta->mu_0;
            const double sigma_0 = dta->sigma_0;
        
            const double x = vals_inp(0);
        
            const double ret = - 0.5*std::log(2*pi) - std::log(sigma_0) - std::pow(x - mu_0,2) / (2*sigma_0*sigma_0);
        
            return ret;
        }
        
        double log_target_dens(const arma::vec& vals_inp, void* ll_data)
        {
            return ll_dens(vals_inp,ll_data) + log_pr_dens(vals_inp,ll_data);
        }

        int main()
        {
            const int n_data = 100;
            const double mu = 2.0;
        
            norm_data_t dta;
            dta.sigma = 1.0;
            dta.mu_0 = 1.0;
            dta.sigma_0 = 2.0;
        
            arma::vec x_dta = mu + arma::randn(n_data,1);
            dta.x = x_dta;
        
            arma::vec initial_val(1);
            initial_val(0) = 1.0;

            //

            mcmc::algo_settings_t settings;

            settings.de_settings.n_burnin_draws = 2000;
            settings.de_settings.n_keep_draws = 2000;

            //

            mcmc::Cube_t draws_out;
            mcmc::de(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "de mean:\n" << arma::mean(draws_out.mat(settings.de_settings.n_keep_draws - 1)) << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.de_settings.n_accept_draws) / (settings.de_settings.n_keep_draws * settings.de_settings.n_pop) << std::endl;
            
            //
        
            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define MCMC_ENABLE_EIGEN_WRAPPERS
        #include "mcmc.hpp"

        inline
        Eigen::VectorXd
        eigen_randn_colvec(size_t nr)
        {
            static std::mt19937 gen{ std::random_device{}() };
            static std::normal_distribution<> dist;

            return Eigen::VectorXd{ nr }.unaryExpr([&](double x) { (void)(x); return dist(gen); });
        }

        struct norm_data_t {
            double sigma;
            Eigen::VectorXd x;
        
            double mu_0;
            double sigma_0;
        };

        double ll_dens(const Eigen::VectorXd& vals_inp, void* ll_data)
        {
            const double pi = 3.14159265358979;

            //

            const double mu = vals_inp(0);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
            const double sigma = dta->sigma;
            const Eigen::VectorXd x = dta->x;
        
            const int n_vals = x.size();
        
            //
        
            const double ret = - n_vals * (0.5 * std::log(2*pi) + std::log(sigma)) - (x.array() - mu).pow(2).sum() / (2*sigma*sigma);
        
            //
        
            return ret;
        }
        
        double log_pr_dens(const Eigen::VectorXd& vals_inp, void* ll_data)
        {
            const double pi = 3.14159265358979;

            //

            norm_data_t* dta = reinterpret_cast< norm_data_t* >(ll_data);
        
            const double mu_0 = dta->mu_0;
            const double sigma_0 = dta->sigma_0;
        
            const double x = vals_inp(0);
        
            const double ret = - 0.5*std::log(2*pi) - std::log(sigma_0) - std::pow(x - mu_0,2) / (2*sigma_0*sigma_0);
        
            return ret;
        }
        
        double log_target_dens(const Eigen::VectorXd& vals_inp, void* ll_data)
        {
            return ll_dens(vals_inp,ll_data) + log_pr_dens(vals_inp,ll_data);
        }
        
        int main()
        {
            const int n_data = 100;
            const double mu = 2.0;
        
            norm_data_t dta;
            dta.sigma = 1.0;
            dta.mu_0 = 1.0;
            dta.sigma_0 = 2.0;
        
            Eigen::VectorXd x_dta = mu + eigen_randn_colvec(n_data).array();
            dta.x = x_dta;
        
            Eigen::VectorXd initial_val(1);
            initial_val(0) = 1.0;

            //

            mcmc::algo_settings_t settings;

            settings.de_settings.n_burnin_draws = 2000;
            settings.de_settings.n_keep_draws = 2000;

            //

            mcmc::Cube_t draws_out;
            mcmc::de(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "de mean:\n" << draws_out.mat(settings.de_settings.n_keep_draws - 1).colwise().mean() << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.de_settings.n_accept_draws) / (settings.de_settings.n_keep_draws * settings.de_settings.n_pop) << std::endl;
            
            //
        
            return 0;
        }

----
