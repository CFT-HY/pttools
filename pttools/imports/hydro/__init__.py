r"""Import the power spectra and global quantities of hydrodynamic lattice simulations.

This supports the output of the simulation code (SCOTTS) of the Computational Field Theory group
of the University of Helsinki, as published in the Zenodo release of :correia_2026:`\ `.
The data can be compared with the Sound Shell Model with the common interface of
:py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
and the suppression factor of the Sound Shell Model can be fitted to it with
:py:mod:`pttools.ssm.suppression.fit`.

- :py:mod:`pttools.imports.hydro.quantities`: the quantities and the format of the power spectrum files
- :py:mod:`pttools.imports.hydro.run`: :py:class:`HydroRun`, a simulation run, its parameters,
  global quantities and time series
- :py:mod:`pttools.imports.hydro.spectrum`: :py:class:`HydroSpectrum`, the spectra of a run at a given time
- :py:mod:`pttools.imports.hydro.correia_2025`: the runs of :correia_2026:`\ `
"""

from .correia_2025 import *
from .quantities import *
from .run import *
from .spectrum import *
