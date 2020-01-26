.. _install:

Installation
============

.. note:: LittleMCMC is developed for Python 3.6 or later.

LittleMCMC is a pure Python library, so it can be easily installed by using
``pip`` or directly from source.

Using ``pip``
-------------

LittleMCMC can be installed using `pip <https://pip.pypa.io>`_.

.. code-block:: bash

    pip install littlemcmc

From Source
-----------

The source code for LittleMCMC can be downloaded `from GitHub
<https://github.com/eigenfoo/littlemcmc>`_ by running

.. code-block:: bash

    git clone https://github.com/eigenfoo/littlemcmc.git
    cd littlemcmc/
    python setup.py install

Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then,
in the root of the project directory, execute:

.. code-block:: bash

    pytest -v

All of the tests should pass. If any of the tests don't pass and you can't
figure out why, `please open an issue on GitHub
<https://github.com/eigenfoo/littlemcmc/issues>`_.
