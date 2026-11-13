"""Interactive viewer for the datasets of bubbles and spectra exported to HDF5 files.

The viewer is a `Dash <https://dash.plotly.com/>`_ application, which is created by
:py:func:`pttools.plot.dataset.app.create_app` and started by :py:func:`pttools.plot.dataset.app.run`.
For an example, see :ref:`sphx_glr_auto_examples_dataset_view.py`.
Dash is an optional dependency, which can be installed with ``pip install "pttools-gw[dashboard]"``.
Therefore, this package is not imported by :py:mod:`pttools.plot`,
and :py:mod:`pttools.plot.dataset.app` has to be imported explicitly.
The figures of :py:mod:`pttools.plot.dataset.figures` need only Plotly.
"""

from .figures import *
