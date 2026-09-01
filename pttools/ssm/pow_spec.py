"""Power spectrum functions"""

import numpy as np

from pttools.speedup import njit
import pttools.type_hints as th


@njit(cache=True)
def pow_spec(z: th.FloatOrArr, spec_den: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Power spectrum from spectral density at dimensionless wavenumber z.
    $$\mathcal{P}(z) = \frac{z^3}{2 \pi^2} \tilde{P}(z)$$

    :gw_pt_ssm:`\ ` eq. 4.18, but without the factor of 2.
    :gowling_2021:`\ ` eq. 2.14, but without the factor of $3K^2$

    :param z: dimensionless wavenumber $z$
    :param spec_den: spectral density
    :return: power spectrum
    """
    return z**3 / (2. * np.pi ** 2) * spec_den
