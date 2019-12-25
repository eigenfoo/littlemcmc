
.. module:: littlemcmc

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/quickstart.ipynb>`_.

.. _quickstart:

Quickstart
==========

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
