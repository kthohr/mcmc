.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Examples and Tests
==================

.. contents:: :local:

----

API
---

The MCMCLib API follows a relatively simple convention, with most algorithms called using the following syntax:

.. code::
    
    algorithm_id(<initial values>, <log posterior kernel function of the target distribution>, <storage for posterior draws>, <additional data for the log posterior kernel function>);

The inputs, in order, are:

* A vector of initial values used to define the starting point of the algorithm.
* A user-specified function that returns the log posterior kernel value of the target distribution.
* An array to store the posterior draws.
* The final input is optional: it is any object that contains additional data necessary to evaluate the log posterior kernel function.

For example, the RWMH algorithm is called using

.. code:: cpp

    rwmh(const ColVec_t& initial_vals, std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, Mat_t& draws_out, void* target_data);


----

Example
-------

The code below uses the RWMH algorithm generate draws of the mean of a Gaussian likelihood function.

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

        settings.rwmh_settings.par_scale = 0.4;
        settings.rwmh_settings.n_burnin_draws = 2000;
        settings.rwmh_settings.n_keep_draws = 2000;

        //

        Eigen::MatrixXd draws_out;
        mcmc::rwmh(initial_val, log_target_dens, draws_out, &dta, settings);

        //
    
        std::cout << "de mean:\n" << draws_out.colwise().mean() << std::endl;
        std::cout << "acceptance rate: " << static_cast<double>(settings.rwmh_settings.n_accept_draws) / settings.rwmh_settings.n_keep_draws << std::endl;
        
        //
    
        return 0;
    }

On x86-based computers, this example can be compiled using:

.. code:: bash

    g++ -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ rwmh_normal_mean.cpp -o rwmh_normal_mean.out -L./../.. -lmcmc


----

Test suite
----------

You can build the test suite as follows:

.. code:: bash

    # compile tests
    cd ./tests
    ./setup
    cd ./examples
    ./configure -l eigen
    make
    ./rwmh.test
