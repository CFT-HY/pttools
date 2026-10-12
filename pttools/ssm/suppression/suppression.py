"""Interpolate or extrapolate kinetic suppression data in the sound shell model."""

import enum
import logging
import os
from pathlib import Path

import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

from pttools.ssm.suppression.suppression_ssm_data import SUPPRESSION_FOLDER
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

logger: logging.Logger = logging.getLogger(__name__)

# TODO: Why is there a difference in the low-alpha low-vw region between hybrids and no hybrids data set?


class SuppressionMethod(enum.StrEnum):
    """Methods for enabling and disabling suppression and for handling extrapolation."""

    #: Return 1 as the suppression factor.
    NONE = "none"
    #: Return NaN outside the convex hull of the suppression points.
    NO_EXT = "no_ext"
    #: Return the nearest suppression value when outside the convex hull of the suppression points.
    EXT_CONSTANT = DEFAULT = "ext_constant"
    # EXT_LINEAR_UBARF = "ext_linear_ubarf"  # TODO: not implemented yet


class Suppression:
    r"""Suppression factors from a given dataset.

    When adding corrections and extensions to the GW spectra,
    please ensure that the suppression factor datasets are still valid.
    If not, please recompute the datasets by comparing the new PTtools GW spectra to the lattice results.
    (When recomputing, remember to disable suppression from PTtools
    so that the spectrum is not scaled by the old suppression.)
    The suppression datasets of :gowling_2021:`\ ` use $R_*$ instead of $\beta$,
    and therefore the thermal suppression of bubble nucleation of :ajmi_2022:`\ ` and
    :py:func:pttools.ssm.nucleation.r_star: does not affect them.
    """

    def __init__(
            self,
            v_walls: th.FloatArr1D,
            alpha_ns: th.FloatArr1D,
            suppressions: th.FloatArr1D,
            name: str):
        r"""Create a suppression dataset from the given data points.

        :param v_walls: ${v}_\text{wall}$, wall speeds of the data points
        :param alpha_ns: $\alpha_n$, transition strengths of the data points
        :param suppressions: suppression factors of the data points
        :param name: name of the dataset
        :raises ValueError: if the arrays are not of the same size
        """
        if not v_walls.size == alpha_ns.size == suppressions.size:
            raise ValueError(
                f"Input arrays must have the same size. Got: {v_walls.size}, {alpha_ns.size}, {suppressions.size}")
        self.v_walls: th.FloatArr1D = v_walls
        self.alpha_ns: th.FloatArr1D = alpha_ns
        self.suppressions: th.FloatArr1D = suppressions
        self.name: str = name

        self.points: tuple[th.FloatArr1D, th.FloatArr1D] = (self.v_walls, self.alpha_ns)
        self.alpha_n_min: float = self.alpha_ns.min()
        self.alpha_n_max: float = self.alpha_ns.max()
        self.v_wall_min: float = self.v_walls.min()
        self.v_wall_max: float = self.v_walls.max()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], name: str) -> "Suppression":
        """Load a suppression dataset from a text file or a ``.npz`` file.

        The file should contain the columns or arrays ``vw_sim``, ``alpha_sim`` and ``sup_ssm``.
        A text file should have a header row of column names, as the files created by
        :py:func:`pttools.ssm.suppression.suppression_ssm_data.suppression_ssm_calculator.calc_sup_ssm` and
        :py:func:`pttools.ssm.suppression.suppression_ssm_data.suppression_hydro_calculator.calc_sup_hydro`.

        :param path: path of the file
        :param name: name of the dataset
        :return: the suppression dataset
        """
        keys = ("vw_sim", "alpha_sim", "sup_ssm")
        if Path(path).suffix == ".npz":
            with np.load(path) as data:
                v_walls, alpha_ns, suppressions = (np.asarray(data[key], dtype=np.float64) for key in keys)
        else:
            data = np.genfromtxt(path, names=True, ndmin=1)
            v_walls, alpha_ns, suppressions = (np.asarray(data[key], dtype=np.float64) for key in keys)
        return Suppression(v_walls=v_walls, alpha_ns=alpha_ns, suppressions=suppressions, name=name)

    def merge(self, other: "Suppression", name: str, atol: float = 1e-6) -> "Suppression":
        r"""Create a new dataset that combines the data points of this dataset and another one.

        The data points of the other dataset replace the data points of this dataset
        that have the same ${v}_\text{wall}$ and $\alpha_n$ within the tolerance ``atol``.

        :param other: the dataset whose data points are added, and which take precedence
        :param name: name of the new dataset
        :param atol: absolute tolerance for considering two data points to be the same
        :return: the combined dataset
        """
        keep = np.ones(self.v_walls.size, dtype=np.bool_)
        for v_wall, alpha_n in zip(other.v_walls, other.alpha_ns, strict=True):
            keep &= ~(np.isclose(self.v_walls, v_wall, rtol=0, atol=atol)
                      & np.isclose(self.alpha_ns, alpha_n, rtol=0, atol=atol))
        return Suppression(
            v_walls=np.concatenate((self.v_walls[keep], other.v_walls)),
            alpha_ns=np.concatenate((self.alpha_ns[keep], other.alpha_ns)),
            suppressions=np.concatenate((self.suppressions[keep], other.suppressions)),
            name=name
        )

    @property
    def limits_str(self) -> str:
        """The ranges of the wall speeds and transition strengths of the dataset as a human-readable string."""
        return \
            f"{self.v_wall_min:.3f} < v_wall < {self.v_wall_max:.3f}, " \
            f"{self.alpha_n_min:.3f} < alpha_n < {self.alpha_n_max:.3f}"

    def peak(self) -> tuple[float, float, float]:
        r"""Find the data point with the largest suppression factor.

        :return: ${v}_\text{wall}$, $\alpha_n$ and the suppression factor of the data point
        """
        ind = self.suppressions.argmax()
        return self.v_walls[ind], self.alpha_ns[ind], self.suppressions[ind]

    def suppression[T: FloatOrArr](
            self,
            v_wall: T,
            alpha_n: th.FloatOrArr,
            method: SuppressionMethod = SuppressionMethod.DEFAULT,
            interpolation: th.Interpolation = "linear") -> T:
        """Interpolate the suppression factor for the given points.

        If either of the arguments is an array, this will return a 2D grid indexed as ``[alpha_n, v_wall]``.
        """
        is_scalar = np.isscalar(v_wall) and np.isscalar(alpha_n)

        if method == SuppressionMethod.NONE:
            if is_scalar:
                return 1.  # pyrefly: ignore[bad-return]
            # This has the same shape as the grid that is interpolated below.
            return np.ones((np.size(alpha_n), np.size(v_wall)))  # pyrefly: ignore[bad-return]
        if method not in (SuppressionMethod.NO_EXT, SuppressionMethod.EXT_CONSTANT):
            raise ValueError(f"Got invalid suppression method: {method}")

        if is_scalar:
            point = (v_wall, alpha_n)
            # The SciPy stubs do not accept scalar coordinates for xi, but SciPy itself does.
            sup_scalar = interpolate.griddata(  # pyrefly: ignore[no-matching-overload]
                points=self.points,
                values=self.suppressions,
                xi=point,
                method=interpolation
            )
            if np.isnan(sup_scalar):
                if method == SuppressionMethod.EXT_CONSTANT:
                    sup_scalar = interpolate.griddata(  # pyrefly: ignore[no-matching-overload]
                        points=self.points,
                        values=self.suppressions,
                        xi=point,
                        method="nearest"
                    )
                else:
                    logger.warning(
                        "Got NaN as the suppression factor for v_wall=%s, alpha_n=%s. "
                        "Are you outside the convex hull of suppression points? "
                        "The points are in the range v_wall=[%s, %s], alpha_n=[%s, %s].",
                        v_wall, alpha_n,
                        self.v_wall_min, self.v_wall_max,
                        self.alpha_n_min, self.alpha_n_max
                    )
            return sup_scalar.item()

        mesh = np.meshgrid(v_wall, alpha_n)
        sup = interpolate.griddata(
            points=self.points,
            values=self.suppressions,
            xi=mesh,
            method=interpolation
        )
        if method == SuppressionMethod.EXT_CONSTANT:
            nans = np.isnan(sup)
            if np.any(nans):
                sup[nans] = interpolate.griddata(
                    points=self.points,
                    values=self.suppressions,
                    xi=(mesh[0][nans], mesh[1][nans]),
                    method="nearest"
                )
        return sup  # pyrefly: ignore[bad-return]


def alpha_n_max_approx[T: FloatOrArr](v_wall: T) -> T:
    r"""Approximate $\alpha_{n,\text{max}}({v}_\text{wall})$."""
    # typing.cast() is not used below, since Numba cannot compile it.
    return 1/3 * (1 + 3 * v_wall ** 2) / (1 - v_wall ** 2)  # pyrefly: ignore[bad-return]


def alpha_n_max[T: FloatOrArr](v_wall: T) -> T:
    r"""$\alpha_{n,\text{max}}({v}_\text{wall})$."""
    # vw, al
    # [0.24000, 0.34000]
    # [0.44000, 0.50000]
    # [0.56000, 0.67000]
    if isinstance(v_wall, np.ndarray):
        ret = M2 * v_wall + C2
        small_vws = v_wall < V_WALL_KINK
        ret[small_vws] = M1 * v_wall[small_vws] + C1
        return ret  # pyrefly: ignore[bad-return]
    if v_wall < V_WALL_KINK:
        return M1 * v_wall + C1  # pyrefly: ignore[bad-return]
    return M2 * v_wall + C2  # pyrefly: ignore[bad-return]


def extend(
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        suppressions: th.FloatArr1D,
        v_wall_ext: float = 0.24,
        alpha_n_ext: float = 0.005) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""Extend the suppression dataset to the low ${v}_\text{wall}$ and low $\alpha_n$ region.

    To improve the extrapolation of the suppression factor when later using grid data,
    a point at $({v}_\text{wall}, \alpha_n) = $ (``v_wall_ext``, ``alpha_n_ext``) is added.
    Its suppression factor is extrapolated linearly in $\alpha_n$
    from the data points of the dataset at ${v}_\text{wall} =$ ``v_wall_ext``.

    :param v_walls: ${v}_\text{wall}$, wall speeds of the data points
    :param alpha_ns: $\alpha_n$, transition strengths of the data points
    :param suppressions: suppression factors of the data points
    :param v_wall_ext: ${v}_\text{wall}$ of the added point, which must be one of the wall speeds of the dataset
    :param alpha_n_ext: $\alpha_n$ of the added point
    :return: the extended arrays of the wall speeds, transition strengths and suppression factors
    :raises ValueError: if the dataset has fewer than two points at ``v_wall_ext``
    """
    at_v_wall = np.isclose(v_walls, v_wall_ext, rtol=0, atol=1e-9)
    if np.count_nonzero(at_v_wall) < 2:  # noqa: PLR2004
        raise ValueError(f"The dataset must have at least two points at v_wall={v_wall_ext} for the extension.")
    order = np.argsort(alpha_ns[at_v_wall])
    spl = InterpolatedUnivariateSpline(alpha_ns[at_v_wall][order], suppressions[at_v_wall][order], k=1, ext=0)
    v_walls_ext = np.concatenate(([v_wall_ext], v_walls))
    alpha_ns_ext = np.concatenate(([alpha_n_ext], alpha_ns))
    suppressions_ext = np.concatenate(([float(spl(alpha_n_ext))], suppressions))
    return v_walls_ext, alpha_ns_ext, suppressions_ext


# Constants for alpha_n_max
#: ${v}_\text{wall}$ of the data point at which the fit changes from the first line segment to the second
V_WALL_KINK: float = 0.44
M1: float = (0.5 - 0.34) / (0.44 - 0.24)  # dal/dvw
M2: float = (0.67 - 0.5) / (0.56 - 0.44)
C1: float = 0.34 - M1 * 0.24
C2: float = 0.67000 - M2 * 0.56000

NO_HYBRIDS: Suppression = Suppression.from_file(
    SUPPRESSION_FOLDER / "suppression_no_hybrids_ssm.txt", name="No hybrids")
NO_HYBRIDS_EXT: Suppression = Suppression(
    *extend(v_walls=NO_HYBRIDS.v_walls, alpha_ns=NO_HYBRIDS.alpha_ns, suppressions=NO_HYBRIDS.suppressions),
    name="No hybrids, extended"
)
WITH_HYBRIDS: Suppression = Suppression.from_file(
    SUPPRESSION_FOLDER / "suppression_2_ssm.txt", name="With hybrids")
#: Suppression factors fitted to the $4096^3$ simulations of :correia_2025:`\ `
#: (${v}_\text{wall} = 0.92, \alpha_n = 0.67$ and ${v}_\text{wall} = 0.44, \alpha_n = 0.5$),
#: computed with
#: :py:mod:`pttools.ssm.suppression.suppression_ssm_data.suppression_hydro_calculator`.
#: This has only two data points, and therefore it cannot be interpolated on its own.
#: Please use the combined datasets below instead.
CORREIA_2025: Suppression = Suppression.from_file(
    SUPPRESSION_FOLDER / "suppression_correia_2025.txt", name="Correia et al. 2025")
#: :py:data:`NO_HYBRIDS_EXT`, where the data points at the parameters of :py:data:`CORREIA_2025`
#: are replaced with those of the higher-resolution simulations
NO_HYBRIDS_EXT_CORREIA_2025: Suppression = NO_HYBRIDS_EXT.merge(
    CORREIA_2025, name="No hybrids, extended, with Correia et al. 2025")
#: :py:data:`WITH_HYBRIDS`, where the data points at the parameters of :py:data:`CORREIA_2025`
#: are replaced with those of the higher-resolution simulations
WITH_HYBRIDS_CORREIA_2025: Suppression = WITH_HYBRIDS.merge(
    CORREIA_2025, name="With hybrids, with Correia et al. 2025")
DEFAULT_SUPPRESSION: Suppression = NO_HYBRIDS_EXT
SUPPRESSIONS: list[Suppression] = [
    NO_HYBRIDS, NO_HYBRIDS_EXT, WITH_HYBRIDS, NO_HYBRIDS_EXT_CORREIA_2025, WITH_HYBRIDS_CORREIA_2025]
