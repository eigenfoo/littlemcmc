.. module:: littlemcmc

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/quickstart.ipynb>`_.

.. _quickstart:

LittleMCMC Quickstart
=====================

LittleMCMC is a lightweight and performant implementation of HMC and
NUTS in Python, spun out of the PyMC project. In this quickstart
tutorial, we will walk through the main use case of LittleMCMC, and
outline the various modules that may be of interest.

Table of Contents
-----------------

-  `Who should use LittleMCMC? <#who-should-use-littlemcmc>`__
-  `Sampling <#how-to-sample>`__

   -  `Inspecting the Output of
      lmc.sample <#inspecting-the-output-of-lmc-sample>`__

-  `Other Modules <#other-modules>`__

Who should use LittleMCMC?
--------------------------

LittleMCMC is a fairly barebones library with a very niche use case.
Most users will probably find that
`PyMC3 <https://github.com/pymc-devs/pymc3>`__ will satisfy their needs,
with better strength of support and quality of documentation.

There are two expected use cases for LittleMCMC. Firstly, if you:

1. Have model with only continuous parameters,
2. Are willing to manually transform all of your model’s parameters to
   the unconstrained space (if necessary),
3. Have functions to compute the log probability of your model and its
   derivative, exposed through a Python callable,
4. And all you need is an implementation of HMC/NUTS (preferably in
   Python) to sample from the posterior,

then you should consider using LittleMCMC.

Secondly, if you want to run algorithmic experiments on HMC/NUTS (in
Python), without having to develop around the heavy machinery that
accompanies other probabilistic programming frameworks (like
`PyMC3 <https://github.com/pymc-devs/pymc3/>`__, `TensorFlow
Probability <https://github.com/tensorflow/probability/>`__ or
`Stan <https://github.com/stan-dev/stan>`__), then you should consider
running your experiments in LittleMCMC.

How to Sample
-------------

.. code:: python

    import numpy as np
    import scipy
    import littlemcmc as lmc

.. code:: python

    def logp_func(x, loc=0, scale=1):
        return np.log(scipy.stats.norm.pdf(x, loc=loc, scale=scale))
    
    
    def dlogp_func(x, loc=0, scale=1):
        return -(x - loc) / scale
    
    
    def logp_dlogp_func(x, loc=0, scale=1):
        return logp_func(x, loc=loc, scale=scale), dlogp_func(x, loc=loc, scale=scale)

.. code:: python

    trace, stats = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        size=1,
        draws=1000,
        tune=500,
        step=lmc.NUTS(logp_dlogp_func=logp_dlogp_func, size=1),
        chains=4,
        cores=1,
        progressbar=None  # HTML progress bars don't render well in RST.
    )

Inspecting the Output of ``lmc.sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    trace




.. parsed-literal::

    array([[-0.15004566, -0.46170896, -0.19921587, ..., -0.83863543,
            -0.21860966,  1.19312616]])



.. code:: python

    trace.shape




.. parsed-literal::

    (1, 4000)



.. code:: python

    stats




.. parsed-literal::

    {'depth': array([2, 1, 1, ..., 2, 2, 2]),
     'step_size': array([2.05084533, 2.05084533, 2.05084533, ..., 2.05084533, 2.05084533,
            2.05084533]),
     'tune': array([False, False, False, ..., False, False, False]),
     'mean_tree_accept': array([0.98804566, 0.96665999, 1.        , ..., 0.71715969, 1.        ,
            0.82667303]),
     'step_size_bar': array([1.38939851, 1.38939851, 1.38939851, ..., 1.38939851, 1.38939851,
            1.38939851]),
     'tree_size': array([3., 1., 1., ..., 3., 3., 3.]),
     'diverging': array([False, False, False, ..., False, False, False]),
     'energy_error': array([ 2.32073322e-04,  3.39084572e-02, -3.08542532e-02, ...,
             2.22621388e-02, -1.16581736e-01,  2.44673926e-01]),
     'energy': array([0.98408598, 1.02562518, 0.99620082, ..., 2.83266304, 1.15420445,
            1.74209033]),
     'max_energy_error': array([ 0.01815012,  0.03390846, -0.03085425, ...,  0.54999299,
            -0.11658174,  0.30105821]),
     'model_logp': array([-0.93019538, -1.02552611, -0.93878201, ..., -1.27059323,
            -0.94283363, -1.63071355])}



.. code:: python

    stats["diverging"].shape




.. parsed-literal::

    (4000,)



Other Modules
-------------

LittleMCMC exposes:

1. Two step methods (a.k.a. samplers): ```littlemcmc.HamiltonianMC``
   (Hamiltonian Monte
   Carlo) <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.HamiltonianMC.html#littlemcmc.HamiltonianMC>`__
   and the ```littlemcmc.NUTS`` (No-U-Turn
   Sampler) <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.NUTS.html#littlemcmc.NUTS>`__
2. Various quadpotentials (a.k.a. mass matrices or inverse metrics) in
   ```littlemcmc.quadpotential`` <https://littlemcmc.readthedocs.io/en/latest/api.html#quadpotentials-a-k-a-mass-matrices>`__,
   along with mass matrix adaptation routines
3. Dual-averaging step size adaptation in
   ```littlemcmc.step_sizes`` <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.step_sizes.DualAverageAdaptation.html#littlemcmc.step_sizes.DualAverageAdaptation>`__
4. A leapfrog integrator in
   ```littlemcmc.integration`` <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.integration.CpuLeapfrogIntegrator.html#littlemcmc.integration.CpuLeapfrogIntegrator>`__

These modules should allow for easy experimentation with the sampler.
Please refer to the `API
Reference <https://littlemcmc.readthedocs.io/en/latest/api.html>`__ for
more information.
