r"""Import the power spectra and global quantities of hydrodynamic lattice simulations.

This supports the output of the simulation code (SCOTTS) of the Computational Field Theory group
of the University of Helsinki, as published in the Zenodo release of :correia_2025:`\ `.
The data can be compared with the Sound Shell Model with the common interface of
:py:class:`pttools.ssm.spectrum_base.SpectrumBase`,
and the suppression factor of the Sound Shell Model can be fitted to it with
:py:mod:`pttools.ssm.suppression.fit`.

- :py:mod:`pttools.lattice.params`: :py:class:`LatticeRunParams`, the parameters of a run
- :py:mod:`pttools.lattice.quantities`: the quantities and the integrals over their power spectra
- :py:mod:`pttools.lattice.readers`: the readers of the input file, the log of the global quantities
  and the power spectrum files
- :py:mod:`pttools.lattice.run`: :py:class:`LatticeRun`, a simulation run, its parameters,
  global quantities and time series
- :py:mod:`pttools.lattice.spectrum`: :py:class:`LatticeSpectrum`, the spectra of a run at a given time
- :py:mod:`pttools.lattice.chrw25`: the runs of :correia_2025:`\ `
- :py:mod:`pttools.lattice.asymptote`: the extrapolation of the GW power to late times
"""

from .asymptote import *
from .chrw25 import *
from .params import *
from .quantities import *
from .readers import *
from .run import *
from .spectrum import *
