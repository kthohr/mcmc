# MCMCLib &nbsp; [![Build Status](https://travis-ci.org/kthohr/mcmc.svg?branch=master)](https://travis-ci.org/kthohr/mcmc) [![Coverage Status](https://codecov.io/github/kthohr/mcmc/coverage.svg?branch=master)](https://codecov.io/github/kthohr/mcmc?branch=master)

MCMCLib is a lightweight C++ library of Markov Chain Monte Carlo (MCMC) methods.

Features:

* Parallelized C++11 implementations of several well-known MCMC methods, including:
    - Random Walk Metropolis-Hastings (RWMH);
    - Metropolis-adjusted Langevin algorithm (MALA);
    - Hamiltonian Monte Carlo (HMC); and
    - Riemannian Manifold HMC.

* Samplers designed for multi-modal distributions:
    - Equi-Energy sampling; and
    - Differential Evolution (DE).
* Built on the [Armadillo C++ linear algebra library](http://arma.sourceforge.net/) for fast and efficient matrix-based computation.

## Status

The library is actively maintained, and is still being extended.

Algorithms:

* `rwmh`
* `mala`
* `hmc`
* `rmhmc`
* `aees`
* `de`

## Syntax

MCMCLib functions are generally defined as
```
algorithm(<initial values>, <draws output>, <log kernel target distribution>, <optional: data for target distribution>, <optional: algorithm settings>)
```
where the inputs, in order, are:
* a vector of initial values that define the starting point for the algorithm, and will contain the solution vector at completion;
* the objective function to be minimized (or zeroed-out);
* (optional) any additional parameters passed to the objective function; and
* (optional) control and tuning parameters for the MCMC algorithms.

For example, the RWMH algorithm is called using:
``` cpp
bool rwmh(const arma::vec& initial_vals, arma::mat& draws_out, std::function<double (const arma::vec& vals_inp, void* target_data)> target_log_kernel, void* target_data);
```

## Installation

The library is installed in the usual way:

```bash
# clone mcmc
git clone -b master --single-branch https://github.com/kthohr/mcmc ./mcmc
# build and install
cd ./mcmc
./configure
make
make install
```

The last line will install MCMCLib into `/usr/local`.

There are several configure options available:
* `-c` a coverage build (used with Codecov)
* `-d` 'development' build with install names set to the build directory (as opposed to an install path)
* `-g` a debugging build (optimization flags set to: `-O0 -g`)
* `-m` specify the BLAS and Lapack libraries to link against; for example, `-m "-lopenblas"` or `-m "-framework Accelerate"`
* `-o` compiler optimization options; defaults to `-O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG`
* `-p` enable parallelization features (using OpenMP)


## Example

Objective: Sample the mean parameter from a normal distribution.

Code:

``` cpp
#include "mcmc.hpp"
 
struct norm_data {
    double sigma;
    arma::vec x;
 
    double mu_0;
    double sigma_0;
};
 
double ll_dens(const arma::vec& vals_inp, void* ll_data)
{
    const double mu = vals_inp(0);
    const double pi = arma::datum::pi;
 
    norm_data* dta = reinterpret_cast<norm_data*>(ll_data);
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
    norm_data* dta = reinterpret_cast< norm_data* >(ll_data);
 
    const double mu_0 = dta->mu_0;
    const double sigma_0 = dta->sigma_0;
    const double pi = arma::datum::pi;
 
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
    const int n_data = 100; // simulated data length
    const double mu = 2.0;  // true mean
 
    norm_data dta;
    dta.sigma = 1.0;
    dta.mu_0 = 1.0;
    dta.sigma_0 = 2.0;
 
    arma::vec x_dta = mu + arma::randn(n_data,1);
    dta.x = x_dta;
 
    arma::vec initial_val(1);
    initial_val(0) = 1.0;
 
    arma::mat draws_out;
    mcmc::rwmh(initial_val,draws_out,log_target_dens,&dta);
 
    return 0;
}
```

See http://www.kthohr.com/mcmclib.html for a detailed description of each algorithm, and more examples.

## Author

Keith O'Hara

## License

GPL (>= 2)


