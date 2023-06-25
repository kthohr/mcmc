.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _riemannian-manifold-hmc:

Riemannian Manifold HMC
=======================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC) algorithm is a Markov Chain Monte Carlo method based on principles of Hamiltonian Dynamics.

Let :math:`\theta^{(i)}` denote a :math:`d`-dimensional vector of stored values at stage :math:`i` of the algorithm, and denote the Hamiltonian by 

  .. math::

    H \left\{ \theta, p \right\} := - \ln K(\theta | X) + \frac{1}{2} \log \left\{ (2 \pi)^d | \mathbf{G}(\theta) | \right\} + \frac{1}{2} p^\top \mathbf{G}^{-1}(\theta) p

where :math:`K` denotes the posterior kernel function and :math:`\mathbf{G}` denotes the metric tensor function.

The RM-HMC algorithm proceeds in three steps.

1. (**Initialization**) Sample :math:`p^{(i)} \sim N(0,\mathbf{G}(\theta^{(i)}))`, and set: :math:`\theta^{(*)} = \theta^{(i)}` and :math:`p^{(*)} = p^{(i)}`.

2. (**Leapfrog Steps**) **for** :math:`k \in \{ 1, \ldots,` ``n_leap_steps`` :math:`\}` **do**:

  i. Initialization. Set :math:`\theta_o^{(*)} = \theta^{(i)}` and :math:`p_o^{(*)} = p^{(i)}`

  ii. Momentum Update Half-Step. **for** :math:`l \in \{ 1, \ldots,` ``n_fp_steps`` :math:`\}` **do**:

    .. math::

        p_h^{(*)} = p_o^{(*)} - \dfrac{\epsilon}{2} \times \nabla_\theta H \left\{ \theta_o^{(*)},p_h^{(*)} \right\}

    where the subscript :math:`h` denotes a half-step. (Notice that :math:`p_h^{(*)}`` appears on both sides of this expression.)

  iii. Position Update Step. **for** :math:`l \in \{ 1, \ldots,` ``n_fp_steps`` :math:`\}` **do**:

    .. math::

        \theta^{(*)} = \theta_o^{(*)} + \dfrac{\epsilon}{2} \times \left[ \nabla_p H \left\{ \theta_o^{(*)},p_h^{(*)} \right\} + \nabla_p H \left\{ \theta^{(*)},p_h^{(*)} \right\} \right]

    (Notice that :math:`\theta^{(*)}`` appears on both sides of this expression.)

  iv. Momentum Update Half-Step.

    .. math::

        p^{(*)} = p_h^{(*)} + \epsilon \times \nabla_\theta H \left\{ \theta^{(*)},p_h^{(*)} \right\}

3. (**Accept/Reject Step**) Define

  .. math::

    \alpha = \min \left\{ 1, \exp \left( H \left\{ \theta^{(i)}, p^{(i)} \right\} - H \left\{ \theta^{(*)}, p^{(*)} \right\} \right) \right\}

  Then

  .. math::

    \theta^{(i+1)} = \begin{cases} \theta^{(*)} & \text{ with probability } \alpha \\ \theta^{(i)} & \text{ else } \end{cases}

The algorithm stops when the number of draws reaches ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws.

----

Function Declarations
---------------------

.. _rmhmc-func-ref1:
.. doxygenfunction:: rmhmc(const ColVec_t& initial_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, Mat_t& draws_out, void* target_data, void* tensor_data)
   :project: mcmclib

.. _rmhmc-func-ref2:
.. doxygenfunction:: rmhmc(const ColVec_t& initial_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* target_data)> target_log_kernel, std::function<Mat_t (const ColVec_t& vals_inp, Cube_t* tensor_deriv_out, void* tensor_data)> tensor_fn, Mat_t& draws_out, void* target_data, void* tensor_data, algo_settings_t& settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t rmhmc_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t rmhmc_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int rmhmc_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``size_t rmhmc_settings.n_leap_steps``: the number of leapfrog steps.

  - Default value: ``1``.

- ``fp_t rmhmc_settings.step_size``: scaling parameter for the leapfrog step.

  - Default value: ``1.0``.

- ``Mat_t rmhmc_settings.precond_mat``: preconditioning matrix for the leapfrog step.

  - Default value: diagonal matrix.

- ``size_t rmhmc_settings.n_fp_steps``: the number of fixed-point steps.

  - Default value: ``5``.

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

        arma::mat tensor_fn(const arma::vec& vals_inp, mcmc::Cube_t* tensor_deriv_out, void* tensor_data)
        {
            // const double mu    = vals_inp(0);
            const double sigma = vals_inp(1);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(tensor_data);
            
            const int n_vals = dta->x.n_rows;
        
            //
        
            const double sigma_sq = sigma*sigma;
        
            arma::mat tensor_out = arma::zeros(2,2);
        
            tensor_out(0,0) = ((double) n_vals) / sigma_sq;
            tensor_out(1,1) = 2.0 * ((double) n_vals) / sigma_sq;
            
            //
        
            if (tensor_deriv_out) {
                tensor_deriv_out->setZero(2,2,2);
        
                //
        
                // tensor_deriv_out->mat(0).setZero();
        
                tensor_deriv_out->mat(1) = - 2.0 * tensor_out / sigma;
            }
        
            //
        
            return tensor_out;
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
        
            settings.rmhmc_settings.step_size = 0.2;
            settings.rmhmc_settings.n_burnin_draws = 2000;
            settings.rmhmc_settings.n_keep_draws = 2000;

            //
        
            arma::mat draws_out;
            mcmc::rmhmc(initial_val, log_target_dens, tensor_fn, draws_out, &dta, &dta, settings);

            //
        
            std::cout << "rmhmc mean:\n" << arma::mean(draws_out) << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.rmhmc_settings.n_accept_draws) / settings.rmhmc_settings.n_keep_draws << std::endl;

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

        Eigen::MatrixXd tensor_fn(const Eigen::VectorXd& vals_inp, mcmc::Cube_t* tensor_deriv_out, void* tensor_data)
        {
            // const double mu    = vals_inp(0);
            const double sigma = vals_inp(1);
        
            norm_data_t* dta = reinterpret_cast<norm_data_t*>(tensor_data);
            
            const int n_vals = dta->x.size();
        
            //
        
            const double sigma_sq = sigma*sigma;
        
            Eigen::MatrixXd tensor_out = Eigen::MatrixXd::Zero(2,2);
        
            tensor_out(0,0) = ((double) n_vals) / sigma_sq;
            tensor_out(1,1) = 2.0 * ((double) n_vals) / sigma_sq;
            
            //
        
            if (tensor_deriv_out) {
                tensor_deriv_out->setZero(2,2,2);
        
                //
        
                // tensor_deriv_out->mat(0).setZero();
        
                tensor_deriv_out->mat(1) = - 2.0 * tensor_out / sigma;
            }
        
            //
        
            return tensor_out;
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
        
            settings.rmhmc_settings.step_size = 0.2;
            settings.rmhmc_settings.n_burnin_draws = 2000;
            settings.rmhmc_settings.n_keep_draws = 2000;

            //
        
            Eigen::MatrixXd draws_out;
            mcmc::rmhmc(initial_val, log_target_dens, tensor_fn, draws_out, &dta, &dta, settings);

            //
        
            std::cout << "rmhmc mean:\n" << draws_out.colwise().mean() << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.rmhmc_settings.n_accept_draws) / settings.rmhmc_settings.n_keep_draws << std::endl;

            //
        
            return 0;
        }

----
