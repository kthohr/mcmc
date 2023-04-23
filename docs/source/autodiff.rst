.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Automatic Differentiation
=========================

Gradient-based MCMC methods in MCMCLib (such as HMC and MALA) require a user-defined function that returns a gradient vector at each function evaluation. While this is best achieved by knowing the gradient in closed form, MCMCLib also provides **experimental support** for automatic differentiation with Eigen-based builds via the `autodiff library <https://autodiff.github.io>`_. 

Requirements: an Eigen-based build of MCMCLib, a copy of the ``autodiff`` header files, and a C++17 compatible compiler.

----

Example
-------

The example below uses forward-mode automatic differentiation to compute the gradient of the Gaussian likelihood function, and the HMC algorithm to sample from the posterior distribution of the mean and variance parameters.

.. code:: cpp

    /*
     *  Sampling from a Gaussian distribution using HMC and Autodiff
     */

    #define MCMC_ENABLE_EIGEN_WRAPPERS
    #include "mcmc.hpp"

    #include <autodiff/forward/real.hpp>
    #include <autodiff/forward/real/eigen.hpp>

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
    
        norm_data_t* dta = reinterpret_cast<norm_data_t*>(ll_data);
        const Eigen::VectorXd x = dta->x;
    
        //

        autodiff::real u;
        autodiff::ArrayXreal xd = vals_inp.eval();

        std::function<autodiff::real (const autodiff::ArrayXreal& vals_inp)> normal_dens_log_form \
        = [x, pi](const autodiff::ArrayXreal& vals_inp) -> autodiff::real
        {
            autodiff::real mu    = vals_inp(0);
            autodiff::real sigma = vals_inp(1);

            return - x.size() * (0.5 * std::log(2*pi) + autodiff::detail::log(sigma)) - (x.array() - mu).pow(2).sum() / (2*sigma*sigma);
        };
    
        //

        if (grad_out) {
            Eigen::VectorXd grad_tmp = autodiff::gradient(normal_dens_log_form, autodiff::wrt(xd), autodiff::at(xd), u);

            *grad_out = grad_tmp;
        } else {
            u = normal_dens_log_form(xd);
        }
    
        //
    
        return u.val();
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


This example can be compiled using:

.. code:: bash

    g++ -Wall -std=c++17 -O3 -march=native -ffp-contract=fast -I/path/to/eigen -I/path/to/autodiff -I/path/to/mcmc/include hmc_normal_autodiff.cpp -o hmc_normal_autodiff.cpp -L/path/to/mcmc/lib -lmcmc


----
