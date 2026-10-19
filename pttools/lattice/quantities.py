r"""The quantities in the output files of the hydrodynamic lattice simulations and functions for processing them.

The power spectra $\mathcal{P}(k)$ are normalised as in :correia_2025:`\ ` eq. 15,
$$\int \frac{dk}{k} \mathcal{P}_Y(k) = \langle Y^2 \rangle,$$
and their bins are linearly spaced with $\Delta k = 2\pi / L$.
For the format of the files, see :py:mod:`pttools.lattice.readers`.
"""

import enum

import numpy as np

import pttools.type_hints as th


@enum.unique
class Quantity(enum.StrEnum):
    r"""The quantities whose power spectra are written by the simulation code.

    The suffixes ``div``, ``rot`` and ``tot`` refer to the compressional (longitudinal),
    vortical (transverse) and total components of a vector field.
    The vortical spectra contain the sum of both transverse polarisations.
    """

    #: $v^i$, fluid 3-velocity
    VEL_DIV = "vel-div"
    VEL_ROT = "vel-rot"
    VEL_TOT = "vel-tot"
    #: $X^i = \sqrt{w} \gamma v^i$, so that $U^i = X^i / \sqrt{\bar{w}}$ of :correia_2025:`\ ` eq. 20
    X_DIV = "X-div"
    X_ROT = "X-rot"
    X_TOT = "X-tot"
    #: $J^i$, momentum density
    J_DIV = "J-div"
    J_ROT = "J-rot"
    J_TOT = "J-tot"
    #: $\Pi_{ij}$, transverse-traceless shear stress
    SHST_TT = "shst-TT"
    #: $\mathcal{P}_{\dot{h}} / (32 \pi G)$, transverse-traceless GW energy density, see :correia_2025:`\ ` eq. 25
    GW_TT = "gw-TT"
    #: $e$, energy density
    E = "e"
    #: $\phi$, scalar field
    PHI = "phi"

    @property
    def file_prefix(self) -> str:
        """Prefix of the power spectrum files of the quantity."""
        return f"{self.value}-ps-step"


def mean_square(k: th.FloatArr1D, p: th.FloatArr1D) -> float:
    r"""$\langle Y^2 \rangle$, mean square of a field computed from its power spectrum.

    $$\langle Y^2 \rangle = \int \frac{dk}{k} \mathcal{P}_Y(k)$$
    :correia_2025:`\ ` eq. 15.

    :param k: $k$, wavenumbers
    :param p: $\mathcal{P}_Y(k)$, power spectrum
    :return: $\langle Y^2 \rangle$
    """
    valid = k > 0
    return float(np.trapezoid(p[valid] / k[valid], k[valid]))


def integral_scale(k: th.FloatArr1D, p: th.FloatArr1D) -> float:
    r"""$\xi$, integral scale of a field computed from its power spectrum.

    $$\xi_Y = \frac{1}{\langle Y^2 \rangle} \int \frac{dk}{k^2} \mathcal{P}_Y(k)$$
    :correia_2025:`\ ` eq. 39.

    :param k: $k$, wavenumbers
    :param p: $\mathcal{P}_Y(k)$, power spectrum
    :return: $\xi_Y$, in the same length units as $1/k$
    """
    valid = k > 0
    return float(np.trapezoid(p[valid] / k[valid]**2, k[valid])) / mean_square(k, p)
