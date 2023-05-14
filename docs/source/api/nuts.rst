.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _no-u-turn-sampler:

No-U-Turn Sampler
=================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The No-U-Turn Sampler (NUTS) is a Hamiltonian Monte Carlo (HMC) method that adaptively chooses the number of leapfrog steps and step size. 
The description below is a modified and shortened version of Algorithm 6 ('No-U-Turn Sampler with Dual Averaging') in Hoffman and Gelman (2014).

Let :math:`\theta^{(i)}` denote a :math:`d`-dimensional vector of stored values at stage :math:`i` of the algorithm, and denote the Hamiltonian by

  .. math::

    H(\theta, p) := \frac{1}{2} \log \left\{ (2 \pi)^d | \mathbf{M} | \right\} + \frac{1}{2} p^\top \mathbf{M}^{-1} p - \ln K(\theta | X) 

where :math:`\mathbf{M}` is a pre-conditioning matrix. The NUTS algorithm proceeds in 3 steps.

1. (**Initialization**) Sample :math:`p^{(i)} \sim N(0,\mathbf{M})`, 

  .. math::

    u \sim U(0, \exp( H(\theta^{(i-1)}, p^{(i)}) )),

  and set: :math:`n = 1`, :math:`s = 1`,

  .. math::

    \theta^{(*)} = \theta^{(+)} = \theta^{(-)} = \theta^{(i-1)},

  and

  .. math::

    p^{(*)} = p^{(+)} = p^{(-)} = p^{(i)}

2. (**Proposal Step**) **while** :math:`s = 1` **do**:

  i. Sample a direction: :math:`v \sim R`, where :math:`R` denotes the standard Rademacher distribution (i.e., :math:`v` takes values in :math:`\{-1,1\}` with equal probability).
 
  ii. 
  
    **if** :math:`v = -1`: 
  
      update :math:`\theta^{(*)}`, :math:`\theta^{(-)}`, and :math:`p^{(-)}` by calling the proposal tree-building function (see Hoffman and Gelman (2014) for details).

    **else**: 
      
      update :math:`\theta^{(*)}`, :math:`\theta^{(+)}`, and :math:`p^{(+)}` by calling the proposal tree-building function.

    In addition to the proposal and momentum values, :math:`n'`, :math:`s'`, :math:`\alpha`, and :math:`n_\alpha` are also updated.

    (Note that, in the tree building process, each tree takes :math:`2^{\text{depth}}` leapfrog steps with step size :math:`v \epsilon`.)

  iii. 
  
    **if** :math:`s' = 1`:

    .. math::

        \theta^{(i)} = \begin{cases} \theta^{(*)} & \text{ with probability } n' / n \\ \theta^{(i-1)} & \text{ else } \end{cases}

  iv. Set: :math:`n = n + n'`, :math:`\text{depth} = \text{depth} + 1`, and

    .. math::

        s = s' \times \mathbf{1}[ (\theta^{(+)} - \theta^{(-)}) \cdot p^{(-)} \geq 0 ] \times \mathbf{1}[ (\theta^{(+)} - \theta^{(-)}) \cdot p^{(+)} \geq 0 ]

3. (**Update Step Size**) 

  **if** :math:`i \leq` ``n_adapt_draws``: 
  
    update

    .. math::

        h_i = \left( 1 - \frac{1}{i + t_0} \right) \times h_{i-1} + \frac{1}{i + t_0} \left( \delta - \frac{\alpha}{n_\alpha} \right)

    where: :math:`t_0` is set via ``nuts_settings.t0_val``; and :math:`\delta`, the target acceptance rate, is set via ``nuts_settings.target_accept_rate``.
  
    Set

    .. math::

        \ln \epsilon = \mu - \frac{\sqrt{i}}{\gamma} \times h_i 

    where :math:`\mu = \ln(10 \times \epsilon_0)` and :math:`\gamma` is set via ``nuts_settings.gamma_val``.

    .. math::

        \ln \bar{\epsilon}_i = i^{-\kappa} \times \ln \epsilon + (1 - i^{-\kappa}) \times \ln \bar{\epsilon}_{i-1}

    where :math:`\kappa` is set via ``nuts_settings.kappa_val``.

  **else**: 
    
    Set :math:`\epsilon` equal to :math:`\bar{\epsilon}` from the last adaptation round.

The algorithm stops when the number of draws reaches ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws.

----

Function Declarations
---------------------

.. _nuts-func-ref1:
.. doxygenfunction:: nuts(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data)
   :project: mcmclib

.. _nuts-func-ref2:
.. doxygenfunction:: nuts(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, ColVec_t* grad_out, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data, algo_settings_t& settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t nuts_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t nuts_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int nuts_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``size_t nuts_settings.n_adapt_draws``: the number of draws to use when adaptively setting the step size (:math:`\epsilon`).

  - Default value: ``1000``.

- ``fp_t nuts_settings.target_accept_rate``: the target acceptance rate for the MCMC chain.

  - Default value: ``0.55``.

- ``size_t nuts_settings.max_tree_depth``: maximum tree depth for build tree function.

  - Default value: ``10``.

- ``fp_t nuts_settings.gamma_val``: the tuning parameter :math:`\gamma`, used when updating the step size (:math:`\epsilon`).

  - Default value: ``0.05``.

- ``fp_t nuts_settings.t0_val``: the tuning parameter :math:`t_0`, used when updating the step size (:math:`\epsilon`).

  - Default value: ``10``.

- ``fp_t nuts_settings.kappa_val``: the tuning parameter :math:`\kappa`, used when updating the step size (:math:`\epsilon`).

  - Default value: ``0.75``.

- ``Mat_t nuts_settings.precond_mat``: preconditioning matrix for the leapfrog step.

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
        
            settings.nuts_settings.n_burnin_draws = 2000;
            settings.nuts_settings.n_keep_draws = 2000;
        
            arma::mat draws_out;
            mcmc::nuts(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "nuts mean:\n" << arma::mean(draws_out) << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.nuts_settings.n_accept_draws) / settings.nuts_settings.n_keep_draws << std::endl;

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
        
            settings.nuts_settings.n_burnin_draws = 2000;
            settings.nuts_settings.n_keep_draws = 2000;

            //
        
            Eigen::MatrixXd draws_out;
            mcmc::nuts(initial_val, log_target_dens, draws_out, &dta, settings);

            //
        
            std::cout << "nuts mean:\n" << draws_out.colwise().mean() << std::endl;
            std::cout << "acceptance rate: " << static_cast<double>(settings.nuts_settings.n_accept_draws) / settings.nuts_settings.n_keep_draws << std::endl;

            //
        
            return 0;
        }

----
