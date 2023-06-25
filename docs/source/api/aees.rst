.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _adaptive-equi-energy-sampler:

Adaptive Equi-Energy Sampler
============================

**Table of contents**

.. contents:: :local:

----

Algorithm Description
---------------------

The Adaptive Equi-Energy Sampler (AEES) algorithm is a Markov Chain Monte Carlo method designed to generate samples from multi-modal target distributions. 
See Kou, Zhou, Wong (2006) for details of the standard equi-energy sampler, and Schreck, Fort, Moulines (2013) for the adaptive version (presented here).

Let :math:`\theta_k^{(i)}` denote a :math:`d`-dimensional vector of stored values at stage :math:`i` of the algorithm, drawn from target distribution :math:`\pi_k`, 
where :math:`k \in \{ 0, 1, \ldots, K \}` and :math:`K` denotes the number of energy rings. We will use the following notation to define a tempered target distribution:

.. math::

    \pi_k (\theta) \propto \exp( - H(\theta | X) / T_k)

where :math:`T_k` denotes the temperature (with :math:`T_0 = 1`) and :math:`H` denotes the energy function (i.e., the negative of the log-posterior kernel function).

The AEES algorithm proceeds as follows.

1. Sample :math:`\theta_K^{(i+1)} \sim \pi_K` using Metropolis-Hastings.

2. **for** :math:`k \in \{ K - 1, K - 2, \ldots, 0 \}` **do**: if :math:`i > (K-k) \times (` ``n_initial_draws`` + ``n_burnin_draws`` :math:`)`:

  i. Sample :math:`z \sim U(0,1)`

  ii. (Local move) if :math:`z >` ``ee_prob_par``, sample :math:`\theta_k^{(i+1)} \sim \pi_k` using Metropolis-Hastings.

  iii. (Equi-energy move) if :math:`z \leq` ``ee_prob_par``:

    * construct ``n_rings`` number of evenly spaced energy rings using previous draws from :math:`\pi_{k+1}`: :math:`\{ \theta_{k+1}^{(0)}, \ldots, \theta_{k+1}^{(i)} \}`.

      .. math::

          \alpha = \min \left\{ 1, \dfrac{\pi_{k}(\theta_k^{(*)})}{\pi_{k+1}(\theta_k^{(*)})} \dfrac{\pi_{k+1}(\theta_k^{(i)})}{\pi_{k}(\theta_k^{(i)})} \right\}
    
      where

      .. math::

          \theta_k^{(i+1)} = \begin{cases} \theta_k^{(*)} & \text{ with probability } \alpha \\ \theta_k^{(i)} & \text{ else } \end{cases}


The algorithm stops when the number of draws reaches ``n_initial_draws`` + ``n_burnin_draws`` + ``n_keep_draws``, and returns the final ``n_keep_draws`` number of draws.

----

Function Declarations
---------------------

.. _aees-func-ref1:
.. doxygenfunction:: aees(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data)
   :project: mcmclib

.. _aees-func-ref2:
.. doxygenfunction:: aees(const ColVec_t& initial_vals, std::function< fp_t(const ColVec_t& vals_inp, void *target_data)> target_log_kernel, Mat_t& draws_out, void *target_data, algo_settings_t& algo_settings)
   :project: mcmclib

----

Control Parameters
------------------

The basic control parameters are:

- ``size_t aees_settings.n_initial_draws``: number of initial draws.

- ``size_t aees_settings.n_burnin_draws``: number of burn-in draws.

- ``size_t aees_settings.n_keep_draws``: number of draws to keep (post sample burn-in period).

- ``bool vals_bound``: whether the search space of the algorithm is bounded. If ``true``, then

  - ``ColVec_t lower_bounds``: defines the lower bounds of the search space.

  - ``ColVec_t upper_bounds``: defines the upper bounds of the search space.

Additional settings:

- ``int aees_settings.omp_n_threads``: the number of OpenMP threads to use.

  - Default value: ``-1`` (use all available threads divided by 2).

- ``fp_t aees_settings.par_scale``: scaling parameter for Metropolis-Hastings draws.

  - Default value: ``1.0``.

- ``Mat_t aees_settings.cov_mat``: covariance matrix of Metropolis-Hastings draws.

  - Default value: diagonal matrix.

- ``size_t aees_settings.n_rings``: the number of energy rings.

  - Default value: ``5``.

- ``fp_t aees_settings.ee_prob_par``: the equi-energy sampling probability.

  - Default value: ``0.10``.

- ``ColVec_t aees_settings.temper_vec``: a vector of temperature values.

----

Examples
--------

Gaussian Mixture Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code to run this example is given below.

.. toggle-header::
    :header: **Armadillo (Click to show/hide)**

    .. code:: cpp

        #define MCMC_ENABLE_ARMA_WRAPPERS
        #include "mcmc.hpp"

        struct mixture_data_t { 
            arma::mat mu;
            arma::vec sig_sq;
            arma::vec weights;
        };
        
        double
        gaussian_mixture(const arma::vec& X_vec_inp, const arma::vec& weights, const arma::mat& mu, const arma::vec& sig_sq)
        {
            const double pi = arma::datum::pi;
            
            const int n_vals = X_vec_inp.n_elem;
            const int n_mix = weights.n_elem;
            
            //

            double dens_val = 0;
            
            for (int i = 0; i < n_mix; ++i) {
                const double dist_val = arma::accu(arma::pow(X_vec_inp - mu.col(i), 2));
                
                dens_val += weights(i) * std::exp(-0.5 * dist_val / sig_sq(i)) / std::pow(2.0 * pi * sig_sq(i), static_cast<double>(n_vals) / 2.0);
            }

            //
            
            return std::log(dens_val);
        }

        double
        target_log_kernel(const arma::vec& vals_inp, void* target_data)
        {
            mixture_data_t* dta = reinterpret_cast<mixture_data_t*>(target_data);

            return gaussian_mixture(vals_inp, dta->weights, dta->mu, dta->sig_sq);
        }

        int main()
        {
            const int n_vals = 2;
            const int n_mix  = 2;

            //

            arma::mat mu = arma::ones(n_vals, n_mix) + 1.0;
            mu.col(0) *= -1.0; // (-2, 2)

            arma::vec weights(n_mix, arma::fill::value(1.0 / n_mix));

            arma::vec sig_sq = 0.1 * arma::ones(n_mix);

            mixture_data_t dta;
            dta.mu = mu;
            dta.sig_sq = sig_sq;
            dta.weights = weights;

            //

            arma::vec T_vec(2);
            T_vec(0) = 60.0;
            T_vec(1) = 9.0;

            // settings

            mcmc::algo_settings_t settings;

            settings.aees_settings.n_initial_draws = 1000;
            settings.aees_settings.n_burnin_draws  = 1000;
            settings.aees_settings.n_keep_draws    = 20000;

            settings.aees_settings.n_rings = 11;
            settings.aees_settings.ee_prob_par = 0.05;
            settings.aees_settings.temper_vec = T_vec;

            settings.aees_settings.par_scale = 1.0;
            settings.aees_settings.cov_mat = 0.35 * arma::eye(n_vals, n_vals);

            //

            arma::mat draws_out;

            mcmc::aees(mu.col(0), target_log_kernel, draws_out, &dta, settings);

            arma::cout << "posterior mean for > 0.1:\n" << arma::mean(draws_out.elem( arma::find(draws_out > 0.1) ), 0) << arma::endl;
            arma::cout << "posterior mean for < -0.1:\n" << arma::mean(draws_out.elem( arma::find(draws_out < -0.1) ), 0) << arma::endl;

            //

            return 0;
        }

.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define MCMC_ENABLE_EIGEN_WRAPPERS
        #include "mcmc.hpp"

        struct mixture_data_t { 
            Eigen::MatrixXd mu;
            Eigen::VectorXd sig_sq;
            Eigen::VectorXd weights;
        };
        
        double
        gaussian_mixture(const Eigen::VectorXd& X_vec_inp, const Eigen::VectorXd& weights, const Eigen::MatrixXd& mu, const Eigen::VectorXd& sig_sq)
        {
            const double pi = 3.14159265358979;
            
            const int n_vals = X_vec_inp.size();
            const int n_mix = weights.size();
            
            //

            double dens_val = 0;
            
            for (int i = 0; i < n_mix; ++i) {
                const double dist_val = (X_vec_inp - mu.col(i)).array().pow(2).sum();
                
                dens_val += weights(i) * std::exp(-0.5 * dist_val / sig_sq(i)) / std::pow(2.0 * pi * sig_sq(i), static_cast<double>(n_vals) / 2.0);
            }

            //
            
            return std::log(dens_val);
        }

        double
        target_log_kernel(const Eigen::VectorXd& vals_inp, void* target_data)
        {
            mixture_data_t* dta = reinterpret_cast<mixture_data_t*>(target_data);

            return gaussian_mixture(vals_inp, dta->weights, dta->mu, dta->sig_sq);
        }

        int main()
        {
            const int n_vals = 2;
            const int n_mix  = 2;

            //

            Eigen::MatrixXd mu = Eigen::MatrixXd::Ones(n_vals, n_mix).array() + 1.0;
            mu.col(0) *= -1.0; // (-2, 2)

            Eigen::VectorXd weights = Eigen::VectorXd::Constant(n_mix, 1.0 / n_mix);

            Eigen::VectorXd sig_sq = 0.1 * Eigen::VectorXd::Ones(n_mix);

            mixture_data_t dta;
            dta.mu = mu;
            dta.sig_sq = sig_sq;
            dta.weights = weights;

            //

            Eigen::VectorXd T_vec(2);
            T_vec(0) = 60.0;
            T_vec(1) = 9.0;

            // settings

            mcmc::algo_settings_t settings;

            settings.aees_settings.n_initial_draws = 1000;
            settings.aees_settings.n_burnin_draws  = 1000;
            settings.aees_settings.n_keep_draws    = 20000;

            settings.aees_settings.n_rings = 11;
            settings.aees_settings.ee_prob_par = 0.05;
            settings.aees_settings.temper_vec = T_vec;

            settings.aees_settings.par_scale = 1.0;
            settings.aees_settings.cov_mat = 0.35 * Eigen::MatrixXd::Identity(n_vals, n_vals);

            //

            Eigen::MatrixXd draws_out;

            mcmc::aees(mu.col(0), target_log_kernel, draws_out, &dta, settings);

            //

            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> pos_inds = (draws_out.array() > 0.1);

            Eigen::VectorXd mean_vec = Eigen::VectorXd::Zero(2);

            for (int i = 0; i < n_vals; ++i) {
                for (size_t draw_ind = 0; draw_ind < settings.aees_settings.n_keep_draws; ++draw_ind) {
                    if (pos_inds(draw_ind, i)) {
                        mean_vec(i) += draws_out(draw_ind, i);
                    }
                }
                mean_vec(i) /= pos_inds.col(i).count();
            }
            
            std::cout << "posterior mean for > 0.1:\n" << mean_vec << std::endl;

            //

            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> neg_inds = (draws_out.array() < - 0.1);

            mean_vec = Eigen::VectorXd::Zero(2);

            for (int i = 0; i < n_vals; ++i) {
                for (size_t draw_ind = 0; draw_ind < settings.aees_settings.n_keep_draws; ++draw_ind) {
                    if (neg_inds(draw_ind, i)) {
                        mean_vec(i) += draws_out(draw_ind, i);
                    }
                }
                mean_vec(i) /= neg_inds.col(i).count();
            }
            
            std::cout << "posterior mean for < - 0.1:\n" << mean_vec << std::endl;

            //

            return 0;
        }

----
