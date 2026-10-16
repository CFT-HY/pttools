r"""The quantities in the output files of the hydrodynamic lattice simulations and functions for processing them.

Each power spectrum file ``<quantity>-ps-step<N>.txt`` has three columns:
the wavenumber $k$ in lattice units, the power spectrum $\mathcal{P}(k)$
normalised as in :correia_2025:`\ ` eq. 15,
$$\int \frac{dk}{k} \mathcal{P}_Y(k) = \langle Y^2 \rangle,$$
and the number of lattice modes in the bin.
The bins are linearly spaced with $\Delta k = 2\pi / L$.
"""

import enum
import os

import numpy as np

import pttools.type_hints as th


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


def read_power_spectrum(path: str | os.PathLike[str]) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""Read a power spectrum file.

    The file has the whitespace-separated columns $k$, $\mathcal{P}(k)$ and the number of modes in each bin,
    and no header.
    It's read with :py:func:`numpy.loadtxt`, as it's faster than :py:func:`pandas.read_csv` for these files.
    When reading all the 13132 power spectrum files of the runs of :correia_2025:`\ `
    (about 4000 rows each) with NumPy 2.5 and pandas 3.0, the times were:

    - :py:func:`numpy.loadtxt`: 22.8 s
    - :py:func:`pandas.read_csv` with the C engine: 29.8 s,
      where 0.43 % of the values differed from those of :py:func:`numpy.loadtxt` by one unit of the last place
    - :py:func:`pandas.read_csv` with the C engine and ``float_precision="round_trip"``: 48.8 s,
      where the values were identical to those of :py:func:`numpy.loadtxt`

    :param path: path of the file
    :return: $k$, $\mathcal{P}(k)$ and the number of modes in each bin
    """
    data = np.loadtxt(path, ndmin=2)
    # The implementation with pandas, which is slower, as described above:
    # data = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy(dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2]


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

    $$\xi = \frac{1}{\langle Y^2 \rangle} \int \frac{dk}{k^2} \mathcal{P}_Y(k)$$
    :correia_2025:`\ ` eq. 39.

    :param k: $k$, wavenumbers
    :param p: $\mathcal{P}_Y(k)$, power spectrum
    :return: $\xi$, in the same length units as $1/k$
    """
    valid = k > 0
    return float(np.trapezoid(p[valid] / k[valid]**2, k[valid])) / mean_square(k, p)
