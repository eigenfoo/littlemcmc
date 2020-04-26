================
About LittleMCMC
================

LittleMCMC is a lightweight, performant implementation of Hamiltonian Monte
Carlo (HMC) and the No-U-Turn Sampler (NUTS) in Python. This document aims to
explain and contextualize the motivation and purpose of LittleMCMC. For an
introduction to the user-facing API, refer to the `quickstart tutorial
<https://littlemcmc.readthedocs.io/en/latest/tutorials/quickstart.html>`_.


Motivation and Purpose
----------------------

Bayesian inference and probabilistic computation is complicated and has many moving
parts[1]_. As a result, many probabilistic programming frameworks (or any library that
automates Bayesian inference) are monolithic libraries that handle everything from model
specification (including automatic differentiation of the joint log probability), to
inference (usually via Markov chain Monte Carlo or variational inference), to diagnosis
and visualization of the inference results[2]_. PyMC3 and Stan are two excellent
examples of such monolithic frameworks.

However, such monoliths require users to buy in to entire frameworks or ecosystems. For
example, a user that has specified their model in one framework but who now wishes to
migrate to another library (to take advantage of certain better-supported features, say)
must now reimplement their models from scratch in the new framework.

LittleMCMC remedies this exact use case: by isolating PyMC's HMC/NUTS code in a
standalone library, users with their own log probability function and its derivative may
use PyMC's battle-tested HMC/NUTS samplers.


LittleMCMC in Context
---------------------

LittleMCMC stands on the shoulders of giants (that is, giant open source projects). Most
obviously, LittleMCMC builds from (or, more accurately, is a spin-off project from) the
PyMC project (both PyMC3 and PyMC4).

In terms of prior art, LittleMCMC is similar to several other open-source libraries,
such as `NUTS by Morgan Fouesneau <https://github.com/mfouesneau/NUTS/>`_ or `Sampyl by
Mat Leonard <https://github.com/mcleonard/sampyl/>`_. However, these libraries do not
offer the same functionality as LittleMCMC: for example, they do not allow for easy
changes of the mass matrix (instead assuming that an identity mass matrix), or they do
not raise sampler errors or track sampler statistics such as divergences, energy, etc.

By offering step methods, integrators, quadpotentials and the sampling loop in separate
Python modules, LittleMCMC offers not just a battle-tested sampler, but also an
extensible one: users may configure the samplers as they wish.


.. [1]_ To be convinced of this fact, one can refer to `Michael Betancourt's *Probabilistic Computation* case study <https://betanalpha.github.io/assets/case_studies/probabilistic_computation.html>`_ or `*A Conceptual Introduction to Hamiltonian Monte Carlo* <https://arxiv.org/abs/1701.02434>`_.

.. [2]_ For more detail, one can refer to `this blog post on the anatomy of probabilistic programming frameworks <https://eigenfoo.xyz/prob-prog-frameworks/>`_, the `PyMC3 developer guide <https://docs.pymc.io/developer_guide.html>`_, or `Michael Betancourt's *Introduction to Stan* <https://betanalpha.github.io/assets/case_studies/stan_intro.html#2_the_stan_ecosystem>`_
