.. PTtools documentation master file, created by
   sphinx-quickstart on Thu Jun 24 14:09:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PTtools documentation!
=====================================

PTtools is a Python library for calculating hydrodynamical quantities around expanding bubbles
of the new phase in an early universe phase transition,
and the resulting gravitational wave power spectrum in the Sound Shell Model.

.. plot:: fig/relativistic_combustion.py

Getting started
---------------

PTtools is `available on PyPI <https://pypi.org/project/pttools-gw/>`_ and can be installed with pip:

.. code-block:: bash

  pip3 install --upgrade pttools-gw[numbalsoda,performance]

The ``[numbalsoda]`` and ``[performance]`` flags are optional,
and you can omit them if they are not available on your platform.

PTtools is also `available on Docker Hub <https://hub.docker.com/r/cfthy/pttools>`_ and can be installed with:

.. code-block:: bash

  docker pull cfthy/pttools:main


.. toctree::
   :caption: Contents:
   :maxdepth: 2

   install
   usage
   auto_examples/index
   gen_modules/pttools/pttools
   gen_modules/tests/tests
   history
   acknowledgements
   see_also
   dev
   sg_execution_times


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
