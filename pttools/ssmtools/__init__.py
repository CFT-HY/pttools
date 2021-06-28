r"""
Functions for calculating velocity and gravitational wave power spectra from
a first-order phase transition in the Sound Shell Model.

See Hindmarsh 2018, Hindmarsh & Hijazi 2019.

Author: Mark Hindmarsh 2015-20

Changes 06/20
^^^^^^^^^^^^^

- use analytic formula for high-k sin transforms.
  Should eliminate spurious high-k signal in GWPS from numerical error.
- sin_transform now handles array z, simplifying its calling elsewhere
- resample_uniform_xi function introduced to simply coding for sin_transform of lam
- Allow calls to power spectra and spectral density functions
  with 2-component params list, i.e. params = [v_wall, alpha_n] (parse_params)
  exponential nucleation with parameters (1,) assumed.
- reduced NQDEFAULT from 2000 to 320, to reduce high-k numerical error when using numerical sin transform

Changes planned 06/20
^^^^^^^^^^^^^^^^^^^^^

- improve docstrings
- introduce function for physical GW power spectrum today
- Check default nucleation type for nu function.
- Allow first three letters to specify nucleation type
"""

from .calculators import *
from .const import *
from .spectrum import *
from .ssm import *
