.. _api:

.. currentmodule:: littlemcmc

API Reference
=============

.. _sampling_api:

Sampling
--------

.. autosummary::
    :toctree: generated/

    sample

.. _step_methods_api:

Step Methods
------------

.. autosummary::
    :toctree: generated/

    HamiltonianMC
    NUTS

.. _quadpotentials_api:

Quadpotentials (a.k.a. Mass Matrices)
-------------------------------------

.. autosummary::
    :toctree: generated/

    quad_potential
    QuadPotentialDiag
    QuadPotentialFull
    QuadPotentialFullInv
    QuadPotentialDiagAdapt
    QuadPotentialFullAdapt

.. _step_sizes_api:

Dual Averaging Step Size Adaptation
-----------------------------------

.. autosummary::
    :toctree: generated/

    step_sizes.DualAverageAdaptation

.. _integrators_api:

Integrators 
-----------

.. autosummary::
    :toctree: generated/

    integration.CpuLeapfrogIntegrator
