
.. module:: littlemcmc

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/quickstart.ipynb>`_.

.. _quickstart:

LittleMCMC Quickstart
=====================

LittleMCMC is a lightweight and performant implementation of HMC and
NUTS in Python, spun out of the PyMC project. In this quickstart
tutorial, we will introduce LittleMCMC

Table of Contents
-----------------

-  `Who should use LittleMCMC? <#Who-should-use-LittleMCMC?>`__
-  `Sampling <#Sampling>`__

   -  `Inspecting the Output of
      ``lmc.sample`` <#Inspecting-the-Output-of-lmc.sample>`__

-  `Other Modules <#Other-Modules>`__

Who should use LittleMCMC?
--------------------------

.. raw:: html

   <div class="alert alert-block alert-info">

LittleMCMC is a fairly bare bones library with a very niche use case.
Most users will probably find that
`PyMC3 <https://github.com/pymc-devs/pymc3>`__ will satisfy their needs,
with better strength of support and quality of documentation.

.. raw:: html

   </div>

If you:

1. Have model with only continuous parameters,
2. Are willing to manually “unconstrain” all of your model’s parameters
   (if necessary),
3. Have methods to compute the log probability of the model and its
   derivative, exposed via a Python callable,
4. And all you need is an implementation of HMC/NUTS (preferably in
   Python) to sample from your model

then you should consider using LittleMCMC!

Sampling
--------

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

    trace, stats, results = lmc.sample(
        logp_dlogp_func=logp_dlogp_func,
        size=1,
        draws=1000,
        tune=500,
        step=lmc.NUTS(logp_dlogp_func=logp_dlogp_func, size=1),
        chains=4,
        cores=4,
        progressbar="notebook"
    )


.. parsed-literal::

    /Users/george/miniconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      
    /Users/george/miniconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      


.. parsed-literal::

    


Inspecting the Output of ``lmc.sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    trace




.. parsed-literal::

    array([-0.38331274, -1.76994233, -0.67234733, ...,  0.27817656,
            0.29250676,  0.42966184])



.. code:: python

    trace.shape




.. parsed-literal::

    (4000,)



.. code:: python

    stats




.. parsed-literal::

    {'depth': array([1, 1, 1, ..., 1, 2, 1]),
     'step_size': array([0.94586326, 0.94586326, 0.94586326, ..., 2.16938615, 2.16938615,
            2.16938615]),
     'tune': array([False, False, False, ..., False, False, False]),
     'mean_tree_accept': array([1.        , 0.43665689, 1.        , ..., 0.98765583, 0.72296808,
            0.97965297]),
     'step_size_bar': array([1.20597596, 1.20597596, 1.20597596, ..., 1.28614833, 1.28614833,
            1.28614833]),
     'tree_size': array([1., 1., 1., ..., 1., 3., 1.]),
     'diverging': array([False, False, False, ..., False, False, False]),
     'energy_error': array([-0.25675836,  0.82860753, -0.74393026, ...,  0.01242099,
             0.00169732,  0.02055688]),
     'energy': array([1.25393394, 2.56056236, 1.91071276, ..., 0.95981431, 1.76229677,
            1.02575724]),
     'max_energy_error': array([-0.25675836,  0.82860753, -0.74393026, ...,  0.01242099,
             0.56981615,  0.02055688]),
     'model_logp': array([-0.99240286, -2.48528646, -1.144964  , ..., -0.95762963,
            -0.96171864, -1.01124318])}



.. code:: python

    stats["diverging"].shape




.. parsed-literal::

    (4000,)



Other Modules
-------------

LittleMCMC exposes:

1. Two step methods: Hamiltonian Monte Carlo (HMC) and the No-U-Turn
   Sampler (NUTS)
2. Quadpotentials (a.k.a. mass matrices or inverse metrics)
3. Dual-averaging step size adaptation
4. Leapfrog integration

Refer to the `API
Reference <https://littlemcmc.readthedocs.io/en/latest/api.html>`__ for
more information.
