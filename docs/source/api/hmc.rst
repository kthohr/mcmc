.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _hamiltonian-monte-carlo:

Hamiltonian Monte Carlo
=======================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Hamiltonian Monte Carlo (HMC) algorithm is a Markov Chain Monte Carlo method based on principles of Hamiltonian Dynamics. 

Let :math:`\theta^{(i)}` denote a :math:`d`-dimensional vector of stored values at stage :math:`i` of the algorithm. The HMC algorithm proceeds in three steps.

1. (**Initialization**) Sample :math:`p^{(i)} \sim N(0,\mathbf{M})`, and set: :math:`\theta^{(*)} = \theta^{(i)}` and :math:`p^{(*)} = p^{(i)}`.

2. (**Leapfrog Steps**) **for** :math:`k \in \{ 1, \ldots,` ``n_leap_steps`` :math:`\}` **do**:

  i. Momentum Update Half-Step.

    .. math::

        p^{(*)} = p^{(*)} + \epsilon \times \nabla_\theta \ln K(\theta^{(*)} | X) / 2

    where :math:`K` denotes the posterior kernel function and :math:`\epsilon` is a scaling value set via ``hmc_settings.step_size``.
 
  ii. Position Update Step.

    .. math::

        \theta^{(*)} = \theta^{(*)} + \epsilon \times \mathbf{M}^{-1} p^{(*)}

    where :math:`\mathbf{M}` is a pre-conditioning matrix set via ``hmc_settings.precond_mat``.

  iii. Momentum Update Half-Step.

    .. math::

        p^{(*)} = p^{(*)} + \epsilon \times \nabla_\theta \ln K(\theta^{(*)} | X) / 2

3. (**Accept/Reject Step**) Denote the Hamiltonian by

  .. math::

    H(\theta, p) := \frac{1}{2} \log \left\{ (2 \pi)^d | \mathbf{M} | \right\} + \frac{1}{2} p^\top \mathbf{M}^{-1} p - \ln K(\theta | X) 

  and define

  .. math::

    \alpha = \min \left\{ 1, \exp( H(\theta^{(i)}, p^{(i)}) - H(\theta^{(*)}, p^{(*)}) ) \right\}

  Then

  .. math::

    \theta^{(i+1)} = \begin{cases} \theta^{(*)} & \text{ with probability } \alpha \\ \theta^{(i)} & \text{ else } \end{cases}

The algorithm stops when the number of draws reaches ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws.

----

Function Declarations
---------------------

.. _hmc-func-ref1:
.. doxygenfunction:: hmc(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data)
   :project: mcmclib

.. _hmc-func-ref2:
.. doxygenfunction:: hmc(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data, algo_settings_t& settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t hmc_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t hmc_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int hmc_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``size_t hmc_settings.n_leap_steps``: the number of leapfrog steps.

  - Default value: ``1``.

- ``fp_t hmc_settings.step_size``: scaling parameter for the leapfrog step.

  - Default value: ``1.0``.

- ``Mat_t hmc_settings.precond_mat``: preconditioning matrix for the leapfrog step.

  - Default value: a diagonal matrix.

----

Examples
--------

Gaussian Distribution
~~~~~~~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define MCMC_ENABLE_ARMA_WRAPPERS
        #include "mcmc.hpp"

        struct norm_data_t {
            arma::vec x;
        };
        
        double ll_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
        {
            const double pi = arma::datum::pi;
            
            const double mu    = vals_inp(0);
            const double sigma = vals_inp(1);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
            const arma::vec x = dta->x;
            const int n_vals = x.n_rows;
        
            //
        
            const double ret = - n_vals * (0.5 * std::log(2*pi) + std::log(sigma)) - arma::accu( arma::pow(x - mu,2) / (2*sigma*sigma) );
        
            //

            if (grad_out) {
                grad_out->set_size(2,1);
        
                //
        
                const double m_1 = arma::accu(x - mu);
                const double m_2 = arma::accu( arma::pow(x - mu,2) );
        
                (*grad_out)(0,0) = m_1 / (sigma*sigma);
                (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - ((double) n_vals) / sigma;
            }
        
            //
        
            return ret;
        }
        
        double log_target_dens(const arma::vec& vals_inp, arma::vec* grad_out, void* ll_data)
        {
            return ll_dens(vals_inp,grad_out,ll_data);
        }

        int main()
        {
            const int n_data = 1000;

            const double mu = 2.0;
            const double sigma = 2.0;
        
            norm_data_t dta;
        
            arma::vec x_dta = mu + sigma * arma::randn(n_data,1);
            dta.x = x_dta;
        
            arma::vec initial_val(2);
            initial_val(0) = mu + 1; // mu
            initial_val(1) = sigma + 1; // sigma
        
            mcmc::algo_settings_t settings;
        
            settings.hmc_settings.step_size = 0.08;
            settings.hmc_settings.n_burnin_draws = 2000;
            settings.hmc_settings.n_keep_draws = 2000;
        
            arma::mat draws_out;
            mcmc::hmc(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "hmc mean:\n" << arma::mean(draws_out) << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.hmc_settings.n_accept_draws) / settings.hmc_settings.n_keep_draws << std::endl;

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
            Eigen::VectorXd x;
        };
        
        double ll_dens(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* ll_data)
        {
            const double pi = 3.14159265358979;

            const double mu    = vals_inp(0);
            const double sigma = vals_inp(1);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
            const Eigen::VectorXd x = dta->x;
            const int n_vals = x.size();
        
            //
        
            const double ret = - n_vals * (0.5 * std::log(2*pi) + std::log(sigma)) - (x.array() - mu).pow(2).sum() / (2*sigma*sigma);
        
            //

            if (grad_out) {
                grad_out->resize(2,1);
        
                //
        
                const double m_1 = (x.array() - mu).sum();
                const double m_2 = (x.array() - mu).pow(2).sum();
        
                (*grad_out)(0,0) = m_1 / (sigma*sigma);
                (*grad_out)(1,0) = (m_2 / (sigma*sigma*sigma)) - ((double) n_vals) / sigma;
            }
        
            //
        
            return ret;
        }
        
        double log_target_dens(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* ll_data)
        {
            return ll_dens(vals_inp,grad_out,ll_data);
        }

        int main()
        {
            const int n_data = 1000;

            const double mu = 2.0;
            const double sigma = 2.0;
        
            norm_data_t dta;
        
            Eigen::VectorXd x_dta = mu + sigma * eigen_randn_colvec(n_data).array();
            dta.x = x_dta;
        
            Eigen::VectorXd initial_val(2);
            initial_val(0) = mu + 1; // mu
            initial_val(1) = sigma + 1; // sigma
        
            mcmc::algo_settings_t settings;
        
            settings.hmc_settings.step_size = 0.08;
            settings.hmc_settings.n_burnin_draws = 2000;
            settings.hmc_settings.n_keep_draws = 2000;

            //
        
            Eigen::MatrixXd draws_out;
            mcmc::hmc(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "hmc mean:\n" << draws_out.colwise().mean() << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.hmc_settings.n_accept_draws) / settings.hmc_settings.n_keep_draws << std::endl;

            //
        
            return 0;
        }

----
