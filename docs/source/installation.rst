.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _installation:

Installation
============

MCMCLib is available as a compiled shared library, or as header-only library, for Unix-alike systems only (e.g., popular Linux-based distros, as well as macOS). Note that use of this library with Windows-based systems, with or without MSVC, **is not supported**.


Requirements
------------

MCMCLib requires either the Armadillo or Eigen C++ linear algebra libraries. (Note that Eigen version 3.4.0 requires a C++14-compatible compiler.)

The following options should be declared **before** including the MCMCLib header files. 

- OpenMP functionality is enabled by default if the ``_OPENMP`` macro is detected (e.g., by invoking ``-fopenmp`` with GCC or Clang). 

  - To explicitly enable OpenMP features, use:

  .. code:: cpp

    #define MCMC_USE_OPENMP

  - To explicitly disable OpenMP functionality, use:

  .. code:: cpp

    #define MCMC_DONT_USE_OPENMP

- To use MCMCLib with Armadillo or Eigen:

  .. code:: cpp

    #define MCMC_ENABLE_ARMA_WRAPPERS
    #define MCMC_ENABLE_EIGEN_WRAPPERS

  Example:

  .. code:: cpp

    #define MCMC_ENABLE_EIGEN_WRAPPERS
    #include "mcmc.hpp"

- To use MCMCLib with RcppArmadillo:

  .. code:: cpp

    #define MCMC_USE_RCPP_ARMADILLO

  Example:

  .. code:: cpp

    #define MCMC_USE_RCPP_ARMADILLO
    #include "mcmc.hpp"


----

Installation Method 1: Shared Library
-------------------------------------

The library can be installed on Unix-alike systems via the standard ``./configure && make`` method.

The primary configuration options can be displayed by calling ``./configure -h``, which results in:

.. code:: bash

    $ ./configure -h

    MCMCLib Configuration

    Main options:
    -c    Code coverage build
            (default: disabled)
    -d    Developmental build
            (default: disabled)
    -f    Floating-point number type
            (default: double)
    -g    Debugging build (optimization flags set to -O0 -g)
            (default: disabled)
    -h    Print help
    -i    Install path (default: current directory)
            Example: /usr/local
    -l    Choice of linear algebra library
            Examples: -l arma or -l eigen
    -m    Specify the BLAS and Lapack libraries to link against
            Examples: -m "-lopenblas" or -m "-framework Accelerate"
    -o    Compiler optimization options
            (default: -O3 -march=native -ffp-contract=fast -flto -DARMA_NO_DEBUG)
    -p    Enable OpenMP parallelization features
            (default: disabled)

    Special options:
    --header-only-version    Generate a header-only version of MCMCLib

If choosing a shared library build, set (one) of the following environment variables *before* running `configure`:

.. code:: bash

    export ARMA_INCLUDE_PATH=/path/to/armadillo
    export EIGEN_INCLUDE_PATH=/path/to/eigen

Then, to set the install path to ``/usr/local``, use Armadillo as the linear algebra library, and enable OpenMP features, we would run:

.. code:: bash

    ./configure -i "/usr/local" -l arma -p

Following this with the standard ``make && make install`` would build the library and install into ``/usr/local``.

----

Installation Method 2: Header-only Library
------------------------------------------

MCMCLib is also available as a header-only library (i.e., without the need to compile a shared library). Simply run ``configure`` with the ``--header-only-version`` option:

.. code:: bash

    ./configure --header-only-version

This will create a new directory, ``header_only_version``, containing a copy of MCMCLib, modified to work on an inline basis. 
With this header-only version, simply include the header files (``#include "mcmc.hpp``) and set the include path to the ``head_only_version`` directory (e.g.,``-I/path/to/mcmclib/header_only_version``).
