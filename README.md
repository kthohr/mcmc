# MCMCLib &nbsp; [![Build Status](https://github.com/kthohr/mcmc/actions/workflows/main.yml/badge.svg)](https://github.com/kthohr/mcmc/actions/workflows/main.yml) [![Coverage Status](https://codecov.io/github/kthohr/mcmc/coverage.svg?branch=master)](https://codecov.io/github/kthohr/mcmc?branch=master) [![License](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](./LICENSE) [![Documentation Status](https://readthedocs.org/projects/mcmclib/badge/?version=latest)](https://mcmclib.readthedocs.io/en/latest/?badge=latest)

MCMCLib is a lightweight C++ library of Markov Chain Monte Carlo (MCMC) methods.

Features:

* A C++11/14/17 library of well-known MCMC algorithms.
* Parallelized samplers designed for multi-modal distributions, including:
    - Adaptive Equi-Energy Sampler (AEES)
    - Differential Evolution (DE)
* For fast and efficient matrix-based computation, MCMCLib supports the following templated linear algebra libraries:
  * [Armadillo](http://arma.sourceforge.net/)
  * [Eigen](http://eigen.tuxfamily.org/index.php) (version >= 3.4.0)
* Automatic differentiation functionality is available through use of the [Autodiff library](https://autodiff.github.io)
* OpenMP-accelerated algorithms for parallel computation. 
* Straightforward linking with parallelized BLAS libraries, such as [OpenBLAS](https://github.com/xianyi/OpenBLAS).
* Available as a single precision (``float``) or double precision (``double``) library.
* Available as a header-only library, or as a compiled shared library.
* Released under a permissive, non-GPL license.

### Contents:
* [Algorithms](#algorithms)
* [Documentation](#documentation)
* [General API](#api)
* [Installation](#installation)
* [R Compatibility](#r-compatibility)
* [Examples](#examples)
* [Automatic Differentiation](#automatic-differentiation)
* [Author and License](#author)

## Algorithms

A list of currently available algorithms includes:

* Adaptive Equi-Energy Sampler (AEES)
* Differential Evolution (DE-MCMC)
* Hamiltonian Monte Carlo (HMC)
* Metropolis-adjusted Langevin algorithm (MALA)
* No-U-Turn Sampler (NUTS)
* Random Walk Metropolis-Hastings (RWMH)
* Riemannian Manifold Hamiltonian Monte Carlo (RM-HMC)

## Documentation

Full documentation is available online:

[![Documentation Status](https://readthedocs.org/projects/mcmclib/badge/?version=latest)](https://mcmclib.readthedocs.io/en/latest/?badge=latest)

A PDF version of the documentation is available [here](https://buildmedia.readthedocs.org/media/pdf/mcmclib/latest/mcmclib.pdf).

## API

The MCMCLib API follows a relatively simple convention, with most algorithms called in the following manner:
```
algorithm_id(<initial values>, <log posterior kernel function of the target distribution>, <storage for posterior draws>, <additional data for the log posterior kernel function>);
```
The inputs, in order, are:
* A vector of initial values used to define the starting point of the algorithm.
* A user-specified function that returns the log posterior kernel value of the target distribution.
* An array to store the posterior draws.
* The final input is optional: it is any object that contains additional data necessary to evaluate the log posterior kernel function.

For example, the RWMH algorithm is called using:

``` cpp
bool rwmh(const ColVec_t& initial_vals, std::function<fp_t (const ColVec_t& vals_inp, void* target_data)> target_log_kernel, Mat_t& draws_out, void* target_data);
```

where ``ColVec_t`` is used to represent, e.g., ``arma::vec`` or ``Eigen::VectorXd`` types.

## Installation

MCMCLib is available as a compiled shared library, or as header-only library, for Unix-alike systems only (e.g., popular Linux-based distros, as well as macOS). Use of this library with Windows-based systems, with or without MSVC, **is not supported**.

### Requirements

MCMCLib requires either the Armadillo or Eigen C++ linear algebra libraries. (Note that Eigen version 3.4.0 requires a C++14-compatible compiler.)

Before including the header files, define **one** of the following:
``` cpp
#define MCMC_ENABLE_ARMA_WRAPPERS
#define MCMC_ENABLE_EIGEN_WRAPPERS
```

Example:
``` cpp
#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"
```

### Installation Method 1: Shared Library

The library can be installed on Unix-alike systems via the standard `./configure && make` method.

First clone the library and any necessary submodules:

``` bash
# clone mcmc into the current directory
git clone https://github.com/kthohr/mcmc ./mcmc

# change directory
cd ./mcmc

# clone necessary submodules
git submodule update --init
```

Set (one) of the following environment variables *before* running `configure`:

``` bash
export ARMA_INCLUDE_PATH=/path/to/armadillo
export EIGEN_INCLUDE_PATH=/path/to/eigen
```

Finally:

``` bash
# build and install with Eigen
./configure -i "/usr/local" -l eigen -p
make
make install
```

The final command will install MCMCLib into `/usr/local`.

Configuration options (see `./configure -h`):

&nbsp; &nbsp; &nbsp; **Primary**
* `-h` print help
* `-i` installation path; default: the build directory
* `-f` floating-point precision mode; default: `double`
* `-l` specify the choice of linear algebra library; choose `arma` or `eigen`
* `-m` specify the BLAS and Lapack libraries to link with; for example, `-m "-lopenblas"` or `-m "-framework Accelerate"`
* `-o` compiler optimization options; defaults to `-O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG`
* `-p` enable OpenMP parallelization features (*recommended*)

&nbsp; &nbsp; &nbsp; **Secondary**
* `-c` a coverage build (used with Codecov)
* `-d` a 'development' build
* `-g` a debugging build (optimization flags set to `-O0 -g`)

&nbsp; &nbsp; &nbsp; **Special**
* `--header-only-version` generate a header-only version of MCMCLib (see [below](#installation-method-2-header-only-library))
<!-- * `-R` RcppArmadillo compatible build by setting the appropriate R library directories (R, Rcpp, and RcppArmadillo) -->

## Installation Method 2: Header-only Library

MCMCLib is also available as a header-only library (i.e., without the need to compile a shared library). Simply run `configure` with the `--header-only-version` option:

```bash
./configure --header-only-version
```

This will create a new directory, `header_only_version`, containing a copy of MCMCLib, modified to work on an inline basis. With this header-only version, simply include the header files (`#include "mcmc.hpp`) and set the include path to the `head_only_version` directory (e.g.,`-I/path/to/mcmclib/header_only_version`).

## R Compatibility

To use MCMCLib with an R package, first generate a header-only version of the library (see [above](#installation-method-2-header-only-library)). Then simply add a compiler definition before including the MCMCLib files.

* For RcppArmadillo:
```cpp
#define MCMC_USE_RCPP_ARMADILLO
#include "mcmc.hpp"
```

At this time, builds using `RcppEigen` are not supported as MCMCLib requires a version of Eigen >= v3.4.0.

## Example

To illustrate MCMCLib at work, consider the problem of sampling values of the mean parameter of a normal distribution.

Code:

``` cpp
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
  
    std::cout << "rwmh mean:\n" << draws_out.colwise().mean() << std::endl;
    std::cout << "acceptance rate: " << static_cast<double>(settings.rwmh_settings.n_accept_draws) / settings.rwmh_settings.n_keep_draws << std::endl;
    
    //
 
    return 0;
}
```

On x86-based computers, this example can be compiled using:

``` bash
g++ -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ rwmh_normal_mean.cpp -o rwmh_normal_mean.out -L./../.. -lmcmc
```

Check the `/examples` directory for additional examples, and https://mcmclib.readthedocs.io/en/latest/ for a detailed description of each algorithm.

## Automatic Differentiation

By combining Eigen with the [Autodiff library](https://autodiff.github.io), MCMCLib provides experimental support for automatic differentiation. 

The example below uses forward-mode automatic differentiation to compute the gradient of the Gaussian likelihood function, and the HMC algorithm to sample from the posterior distribution of the mean and variance parameters.

``` cpp
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
```

Compile with:

``` bash
g++ -Wall -std=c++17 -O3 -march=native -ffp-contract=fast -I/path/to/eigen -I/path/to/autodiff -I/path/to/mcmc/include hmc_normal_autodiff.cpp -o hmc_normal_autodiff.cpp -L/path/to/mcmc/lib -lmcmc
```

See the [documentation](https://mcmclib.readthedocs.io/en/latest/autodiff.html) for more details on this topic.

## Author

Keith O'Hara

## License

Apache Version 2
