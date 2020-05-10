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

-  `Customizing the Default NUTS
   Sampler <#customizing-the-default-nuts-sampler>`__
-  `Other Modules <#other-modules>`__

Who should use LittleMCMC?
--------------------------

LittleMCMC is a fairly barebones library with a very niche use case.
Most users will probably find that
`PyMC3 <https://github.com/pymc-devs/pymc3>`__ will satisfy their needs,
with better strength of support and quality of documentation.

There are two expected use cases for LittleMCMC. Firstly, if you:

1. Have a model with only continuous parameters,
2. Are willing to manually transform all of your model’s parameters to
   the unconstrained space (if necessary),
3. Have a Python function/callable that:

   1. computes the log probability of your model and its derivative
   2. is `pickleable <https://docs.python.org/3/library/pickle.html>`__
   3. outputs an array with the same shape as its input

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

    # By default: 4 chains in 4 cores, 500 tuning steps and 1000 sampling steps.
    trace, stats = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        model_ndim=1,
        progressbar=None,  # HTML progress bars don't render well in RST.
    )


.. parsed-literal::

    /home/george/littlemcmc/venv/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      


Inspecting the Output of ``lmc.sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Shape is (num_chains, num_samples, num_parameters)
    trace.shape




.. parsed-literal::

    (4, 1000, 1)



.. code:: python

    # The first 2 samples across all chains and parameters
    trace[:, :2, :]




.. parsed-literal::

    array([[[ 0.92958231],
            [ 0.92958231]],
    
           [[-1.06231693],
            [-1.11589309]],
    
           [[-0.73177109],
            [-0.66975061]],
    
           [[ 0.8923907 ],
            [ 0.97253646]]])



.. code:: python

    stats.keys()




.. parsed-literal::

    dict_keys(['depth', 'step_size', 'tune', 'mean_tree_accept', 'step_size_bar', 'tree_size', 'diverging', 'energy_error', 'energy', 'max_energy_error', 'model_logp'])



.. code:: python

    # Again, shape is (num_chains, num_samples, num_parameters)
    stats["depth"].shape




.. parsed-literal::

    (4, 1000, 1)



.. code:: python

    # The first 2 tree depths across all chains and parameters
    stats["depth"][:, :2, :]




.. parsed-literal::

    array([[[2],
            [1]],
    
           [[1],
            [1]],
    
           [[2],
            [1]],
    
           [[2],
            [1]]])



Customizing the Default NUTS Sampler
------------------------------------

By default, ``lmc.sample`` samples using NUTS with sane defaults. These
defaults can be override by either:

1. Passing keyword arguments from ``lmc.NUTS`` into ``lmc.sample``, or
2. Constructing an ``lmc.NUTS`` sampler, and passing that to
   ``lmc.sample``. This method also allows you to choose to other
   samplers, such as ``lmc.HamiltonianMC``.

For example, suppose you want to increase the ``target_accept`` from the
default ``0.8`` to ``0.9``. The following two cells are equivalent:

.. code:: python

    trace, stats = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        model_ndim=1,
        target_accept=0.9,
        progressbar=None,
    )

.. code:: python

    step = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=1, target_accept=0.9)
    trace, stats = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        model_ndim=1,
        step=step,
        progressbar=None,
    )


.. parsed-literal::

    /home/george/littlemcmc/venv/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      


For a list of keyword arguments that ``lmc.NUTS`` accepts, please refer
to the `API reference for
``lmc.NUTS`` <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.NUTS.html#littlemcmc.NUTS>`__.

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
