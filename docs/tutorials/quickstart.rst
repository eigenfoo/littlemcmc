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
        progressbar="notebook"
    )



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=1500.0), HTML(value='')))


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=1500.0), HTML(value='')))


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=1500.0), HTML(value='')))


.. parsed-literal::

    



.. parsed-literal::

    HBox(children=(FloatProgress(value=0.0, max=1500.0), HTML(value='')))


.. parsed-literal::

    


Inspecting the Output of ``lmc.sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    trace




.. parsed-literal::

    array([[ 0.98638645,  1.594521  ,  0.89394842, ...,  0.21827979,
             0.35573737, -0.23779242]])



.. code:: python

    trace.shape




.. parsed-literal::

    (1, 4000)



.. code:: python

    stats




.. parsed-literal::

    {'depth': array([1, 1, 2, ..., 2, 1, 2]),
     'step_size': array([1.94303615, 1.94303615, 1.94303615, ..., 1.94303615, 1.94303615,
            1.94303615]),
     'tune': array([False, False, False, ..., False, False, False]),
     'mean_tree_accept': array([1.        , 0.72530117, 0.85098591, ..., 0.990029  , 0.98398404,
            1.        ]),
     'step_size_bar': array([1.44061287, 1.44061287, 1.44061287, ..., 1.44061287, 1.44061287,
            1.44061287]),
     'tree_size': array([1., 1., 3., ..., 3., 1., 3.]),
     'diverging': array([False, False, False, ..., False, False, False]),
     'energy_error': array([-0.04662742,  0.32116831, -0.3567352 , ..., -0.0042028 ,
             0.0161456 , -0.0143246 ]),
     'energy': array([1.66214065, 2.33856467, 3.0377291 , ..., 0.99966459, 0.98942996,
            0.98859287]),
     'max_energy_error': array([-0.04662742,  0.32116831,  0.47937797, ...,  0.01749271,
             0.0161456 , -0.01805088]),
     'model_logp': array([-1.40541765, -2.19018714, -1.31851043, ..., -0.94276157,
            -0.98221307, -0.94721115])}



.. code:: python

    stats["diverging"].shape




.. parsed-literal::

    (4000,)



Other Modules
-------------

LittleMCMC exposes:

1. Two step methods: `Hamiltonian Monte Carlo
   (HMC) <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.HamiltonianMC.html#littlemcmc.HamiltonianMC>`__
   and the `No-U-Turn Sampler
   (NUTS) <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.NUTS.html#littlemcmc.NUTS>`__.
2. Classes for various
   `quadpotentials <https://littlemcmc.readthedocs.io/en/latest/api.html#quadpotentials-a-k-a-mass-matrices>`__
   (a.k.a. mass matrices or inverse metrics) and mass matrix adaptation
   routines
3. A class for `dual-averaging step size
   adaptation <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.step_sizes.DualAverageAdaptation.html#littlemcmc.step_sizes.DualAverageAdaptation>`__
4. `A leapfrog
   integrator <https://littlemcmc.readthedocs.io/en/latest/generated/littlemcmc.integration.CpuLeapfrogIntegrator.html#littlemcmc.integration.CpuLeapfrogIntegrator>`__

These modules should allow for easy experimentation with the sampler.
Please refer to the `API
Reference <https://littlemcmc.readthedocs.io/en/latest/api.html>`__ for
more information.
