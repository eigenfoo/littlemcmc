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

Dual Averaging Step Size Adaptation
-----------------------------------

.. autosummary::
    :toctree: generated/

    DualAverageAdaptation

Integrators 
-----------

.. autosummary::
    :toctree: generated/

    CpuLeapfrogIntegrator
