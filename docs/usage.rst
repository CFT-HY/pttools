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

- :meth:`pttools.ssm.sin_transform.sin_transform()`
- :meth:`pttools.ssm.spec_den_v.spec_den_v()`

Exporting data
--------------
The models, bubbles and spectra can be exported as JSON with their ``export()`` methods,
and in large numbers as HDF5 files with :class:`pttools.export.exporter.Exporter`.
A single HDF5 file can contain hundreds of thousands of spectra,
and the bubbles and models that are shared by several spectra are stored only once.
Each field is stored as a separate dataset, so that it can be read at once as a NumPy array.

.. code-block:: python

  from pttools.export import Exporter, Importer, Preset, Table

  with Exporter("spectra.h5", spectrum_fields=[Preset.MINIMAL, "f"]) as exporter:
      exporter.add_many(spectra)

  with Importer("spectra.h5", verify=True) as importer:
      params = importer.read_scalars(Table.SPECTRA)
      omgw0_h2 = importer.read(Table.SPECTRA, "omgw0_h2")
      profiles_v = importer.read(Table.BUBBLES, "v")
      spectrum = importer.load_spectrum(0)

The exported fields are selected with the presets of :class:`pttools.utils.fields.Preset`,
field names, and custom :class:`pttools.utils.fields.Field` objects.
The available fields are listed in
:mod:`pttools.models.export`, :mod:`pttools.bubble.export`, :mod:`pttools.ssm.export` and :mod:`pttools.omgw0.export`.
When the exporter is closed, it writes a SHA-256 checksum file next to the HDF5 file,
which can be verified with :func:`pttools.export.checksum.verify_checksum` or ``sha256sum --check``.
For parallel computation, the fields can be extracted in the worker processes with
:attr:`pttools.export.exporter.Exporter.extractor`, as described in the documentation of the exporter.
