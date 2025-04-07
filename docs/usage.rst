Usage
=====

Basic usage
-----------
To install PTtools, please first follow the instructions in the :doc:`installation guide <install>`.
Then download and run one of the examples in the :doc:`example gallery <auto_examples/index>`.


Numba performance
-----------------
The computationally intensive parts of PTtools are JIT-compiled using
`Numba <https://numba.pydata.org/>`_.
Therefore, the first calls to PTtools may take tens of seconds, but once the it's compiled,
it's significantly faster than pure Python.

Therefore, if you are running several simulations, you can save time by running these as a single script
so that PTtools has to be compiled only once.
Jupyter notebooks and IPython shells can also be used to effectively cache the compiled PTtools.

If you're quickly developing scripts that don't need the power of Numba,
`you can disable it <https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#disabling-jit-compilation>`_
for your script.
This is configured by an environment variable, which can be set in the Bash shell as:

.. code-block:: bash

  NUMBA_DISABLE_JIT=1 python3 your_script.py

Alternatively you can set the environment variable in your Python code before importing Numba:

.. code-block:: python

  import os
  os.environ["NUMBA_DISABLE_JIT"] = "1"
  import numba

Numba errors
------------
If you get a Numba error when using PTtools, please do the following.

- Check that the parameters you give to the PTtools functions are of the types specified by their type hints.
- Upgrade Numba and other libraries to the latest versions.
  PTtools should support a wide range of Numba versions, but some of the older versions may have subtle bugs that
  apply only to a few versions.
  Incompatibilities between the versions of Numba, NumPy and llvmlite may also cause errors.
- If the issue persists, please create an issue in the :issue:`issue tracker <>`.

Parallelism
-----------
The most of the computation is serial, but some steps benefit significantly from parallel CPU resources.
These include:

- :meth:`pttools.ssmtools.calculators.sin_transform()`
- :meth:`pttools.ssmtools.spectrum.spec_den_v()`
