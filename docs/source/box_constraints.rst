.. Copyright (c) 2011-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Box Constraints
===============

This section provides implementation details for how MCMCLib handles box constraints.

For a parameter :math:`\theta_j` defined on a bounded interval :math:`[a_j,b_j]`, where :math:`a_j < b_j`, we use the generalized logit transform:

.. math::

    g(\theta_j) = \phi_j := \ln \left( \frac{x_j - a_j}{b_j - x_j} \right)

with corresponding inverse transform

.. math::
	\theta_j = g^{-1} (\phi_j) = \frac{a_j + b_j \exp(\phi_j)}{1 + \exp(\phi_j)}

Note that the support of :math:`\phi_j` is :math:`\mathbb{R}`.

The log posterior kernel function of the :math:`J \times 1` vector :math:`\boldsymbol{\theta}` is given by:

.. math::

    K(\boldsymbol{\theta} | Y) = \ln L( Y | \boldsymbol{\theta} ) + \ln \pi (\boldsymbol{\theta}) 

where :math:`L` is the likelihood function of the data :math:`Y` parametrized by :math:`\boldsymbol{\theta}`, and :math:`\pi` is the joint prior distribution over :math:`\boldsymbol{\theta}`. 
The log posterior kernel function of the transformed parameter vector :math:`\boldsymbol{\phi} = g(\boldsymbol{\theta})` is then computed as

.. math::

	K(\boldsymbol{\phi} | Y) = \ln L( Y | g^{-1} (\boldsymbol{\phi}) ) + \ln \pi (g^{-1} (\boldsymbol{\phi})) + \ln | J(\boldsymbol{\phi}) |

where :math:`|J|` is the modulus of the Jacobian determinant matrix :math:`J`---that is, the determinant of a matrix with :math:`(i,j)` elements equal to

.. math::

	\frac{\partial [g^{-1} (\phi)]_i}{\partial \phi_j} = \frac{\partial \theta_i}{\partial \phi_j}

If the parameters :math:`\boldsymbol{\theta}` are assumed to be *a-priori* independent, then

.. math::
	\pi (\boldsymbol{\theta}) = \pi_1 (\theta_1) \cdots \pi_J (\theta_J)

Given our specification for :math:`g`, the Jacobian term will be a diagonal matrix with non-negative elements: the log Jacobian adjustment for parameter :math:`j` is given by

.. math::
	\ln J_{j,j} := \ln \left( \frac{d \theta_j}{d \phi_j} \right) = \ln(b_j - a_j) + \phi_j - 2 \ln (1 + \exp(\phi_j))

As the determinant of a diagonal matrix is the product of its diagonal elements, the final term in can be computed as a sum:

.. math::
	\ln | J(\boldsymbol{\phi}) | = \sum_{j=1}^J \ln | J_{j,j} | = \sum_{j=1}^J \left[ \ln(b_j - a_j) + \phi_j - 2 \ln (1 + \exp(\phi_j)) \right]
