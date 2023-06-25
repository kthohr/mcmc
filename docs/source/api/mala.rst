.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _metropolis-adjusted-langevin-algorithm:

Metropolis-adjusted Langevin Algorithm
======================================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Metropolis-adjusted Langevin algorithm (MALA) extends the Random Walk Metropolis-Hasting algorithm by generating proposal draws via Langevin diffusions.

Let :math:`\theta^{(i)}` denote a :math:`d`-dimensional vector of stored values at stage :math:`i` of the algorithm. MALA proceeds in two steps.

1. (**Proposal Step**) Let

  .. math::

    \mu(\theta^{(i)}) := \theta^{(i)} + \frac{\epsilon^2}{2} \times \mathbf{M} \left[ \nabla_\theta \ln K(\theta^{(i)} | X) \right]

  where :math:`K` denotes the posterior kernel function; :math:`\nabla_\theta` denotes the gradient operator; :math:`\mathbf{M}` is a pre-conditioning matrix, set via ``mala_settings.precond_mat``; and :math:`\epsilon` is a scaling value, set via ``mala_settings.step_size``.

  Generate a proposal draw, :math:`\theta^{(*)}`. using:

  .. math::

    \theta^{(*)} = \mu(\theta^{(i)}) + c \times \mathbf{M}^{1/2} W

  where :math:`W \sim N(0,I_d)`.

2. (**Accept/Reject Step**) Denote the proposal density by :math:`q(\theta^{(*)} | \theta^{(i)}) := \phi(\theta^{(*)}; \mu(\theta^{(i)}), \epsilon^2 \mathbf{M})` and let

  .. math::

    \alpha = \min \left\{ 1, [ K(\theta^{(*)} | X) q(\theta^{(i)} | \theta^{(*)})] / [ K(\theta^{(i)} | X) q(\theta^{(*)} | \theta^{(i)})] \right\}

  Then

  .. math::

    \theta^{(i+1)} = \begin{cases} \theta^{(*)} & \text{ with probability } \alpha \\ \theta^{(i)} & \text{ else } \end{cases}

The algorithm stops when the number of draws reaches ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws.

----

Function Declarations
---------------------

.. _mala-func-ref1:
.. doxygenfunction:: mala(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data)
   :project: mcmclib

.. _mala-func-ref2:
.. doxygenfunction:: mala(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data, algo_settings_t& settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t mala_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t mala_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int mala_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``fp_t mala_settings.step_size``: scaling parameter for the proposal step.

  - Default value: ``1.0``.

- ``Mat_t mala_settings.precond_mat``: preconditioning matrix for the proposal step.

  - Default value: diagonal matrix.

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

            //
        
            mcmc::algo_settings_t settings;
        
            settings.mala_settings.step_size = 0.08;
            settings.mala_settings.n_burnin_draws = 2000;
            settings.mala_settings.n_keep_draws = 2000;

            //
        
            arma::mat draws_out;
            mcmc::mala(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "mala mean:\n" << arma::mean(draws_out) << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.mala_settings.n_accept_draws) / settings.mala_settings.n_keep_draws << std::endl;

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
        
            settings.mala_settings.step_size = 0.08;
            settings.mala_settings.n_burnin_draws = 2000;
            settings.mala_settings.n_keep_draws = 2000;

            //
        
            Eigen::MatrixXd draws_out;
            mcmc::mala(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "mala mean:\n" << draws_out.colwise().mean() << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.mala_settings.n_accept_draws) / settings.mala_settings.n_keep_draws << std::endl;

            //
        
            return 0;
        }

----
