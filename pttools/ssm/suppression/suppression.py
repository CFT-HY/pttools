"""Interpolate or extrapolate kinetic suppression data in the sound shell model."""

import enum
import logging
import os

import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

from pttools.ssm.suppression.suppression_ssm_data.suppression_ssm_calculator import SUPPRESSION_FOLDER
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

logger = logging.getLogger(__name__)

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
    r"""Suppression factors from a given dataset

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
        if not v_walls.size == alpha_ns.size == suppressions.size:
            raise ValueError(
                f"Input arrays must have the same size. Got: {v_walls.size}, {alpha_ns.size}, {suppressions.size}")
        self.v_walls: th.FloatArr1D = v_walls
        self.alpha_ns: th.FloatArr1D = alpha_ns
        self.suppressions: th.FloatArr1D = suppressions
        self.name: str = name

        self.points = (self.v_walls, self.alpha_ns)
        self.alpha_n_min: float = self.alpha_ns.min()
        self.alpha_n_max: float = self.alpha_ns.max()
        self.v_wall_min: float = self.v_walls.min()
        self.v_wall_max: float = self.v_walls.max()

    @classmethod
    def from_file(cls, path: str, name: str) -> "Suppression":
        with np.load(path) as data:
            return Suppression(
                v_walls=data["vw_sim"],
                alpha_ns=data["alpha_sim"],
                suppressions=data["sup_ssm"],
                name=name
            )

    @property
    def limits_str(self) -> str:
        return \
            f"{self.v_wall_min:.3f} < v_wall < {self.v_wall_max:.3f}, " \
            f"{self.alpha_n_min:.3f} < alpha_n < {self.alpha_n_max:.3f}"

    def peak(self) -> tuple[float, float, float]:
        ind = self.suppressions.argmax()
        return self.v_walls[ind], self.alpha_ns[ind], self.suppressions[ind]

    def suppression(
            self,
            v_wall: th.FloatOrArr,
            alpha_n: th.FloatOrArr,
            method: SuppressionMethod = SuppressionMethod.DEFAULT,
            interpolation: th.Interpolation = "linear") -> th.FloatOrArr:
        """Interpolate the suppression factor for the given points

        If given arrays, this will return a 2D grid.
        """
        is_scalar = np.isscalar(v_wall) and np.isscalar(alpha_n)

        if method == SuppressionMethod.NONE:
            return 1. if is_scalar else np.ones_like((v_wall.size, alpha_n.size))
        if method not in (SuppressionMethod.NO_EXT, SuppressionMethod.EXT_CONSTANT):
            raise ValueError(f"Got invalid suppression method: {method}")

        mesh: tuple[th.FloatOrArr, th.FloatOrArr] = (v_wall, alpha_n) if is_scalar else np.meshgrid(v_wall, alpha_n)
        sup = interpolate.griddata(
            points=self.points,
            values=self.suppressions,
            xi=mesh,
            method=interpolation
        )
        if is_scalar:
            if np.isnan(sup):
                if method == SuppressionMethod.EXT_CONSTANT:
                    sup = interpolate.griddata(
                        points=self.points,
                        values=self.suppressions,
                        xi=mesh,
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
            return sup.item()

        if method == SuppressionMethod.EXT_CONSTANT:
            nans = np.isnan(sup)
            if np.any(nans):
                sup[nans] = interpolate.griddata(
                    points=self.points,
                    values=self.suppressions,
                    xi=(mesh[0][nans], mesh[1][nans]),
                    method="nearest"
                )
        return sup


def alpha_n_max_approx[T: FloatOrArr](v_wall: T) -> T:
    r"""Approximate $\alpha_{n,\text{max}}({v}_\text{wall})$"""
    return 1/3 * (1 + 3 * v_wall ** 2) / (1 - v_wall ** 2)


def alpha_n_max[T: FloatOrArr](v_wall: T) -> T:
    r"""$\alpha_{n,\text{max}}({v}_\text{wall})$"""
    # vw, al
    # [0.24000, 0.34000]
    # [0.44000, 0.50000]
    # [0.56000, 0.67000]
    if np.isscalar(v_wall) and v_wall < 0.44:
        return M1 * v_wall + C1
    ret = M2 * v_wall + C2
    small_vws = v_wall < 0.44
    ret[small_vws] = M1 * v_wall[small_vws] + C1
    return ret


def extend(
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        suppressions: th.FloatArr1D) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    """
    To improve the extrapolation of the suppression factor when later using grid data, first extend the
    low vw and low alpha region as follows
    """
    # alpha values in suppression dataset for vw = 0.24
    ssm_sup_vw_0_24_alphas = np.array([0.05000, 0.07300, 0.11000, 0.16000, 0.23000, 0.34000])
    # Suppression values for vw = 0.24
    ssm_sup_vw_0_24 = np.array([0.01675, 0.01218, 0.00696, 0.00251, 0.00054, 0.00007])

    spl = InterpolatedUnivariateSpline(ssm_sup_vw_0_24_alphas, ssm_sup_vw_0_24, k=1, ext=0)

    ssm_sup_vw_0_24_alphas_ext = np.array([0.00500, 0.05000, 0.07300, 0.11000, 0.16000, 0.23000, 0.34000])
    ssm_sup_vw_0_24_ext = spl(ssm_sup_vw_0_24_alphas_ext)

    # create the extrapolated dataset
    v_walls_ext = np.concatenate(([0.24], v_walls))
    alpha_ns_ext = np.concatenate(([ssm_sup_vw_0_24_alphas_ext[0]], alpha_ns))
    suppressions_ext = np.concatenate(([ssm_sup_vw_0_24_ext[0]], suppressions))
    return v_walls_ext, alpha_ns_ext, suppressions_ext


# Constants for alpha_n_max
M1: float = (0.5 - 0.34) / (0.44 - 0.24)  # dal/dvw
M2: float = (0.67 - 0.5) / (0.56 - 0.44)
C1: float = 0.34 - M1 * 0.24
C2: float = 0.67000 - M2 * 0.56000

NO_HYBRIDS = Suppression.from_file(os.path.join(SUPPRESSION_FOLDER, "suppression_no_hybrids_ssm.npz"), name="No hybrids")
NO_HYBRIDS_EXT = Suppression(
    *extend(v_walls=NO_HYBRIDS.v_walls, alpha_ns=NO_HYBRIDS.alpha_ns, suppressions=NO_HYBRIDS.suppressions),
    name="No hybrids, extended"
)
WITH_HYBRIDS = Suppression.from_file(os.path.join(SUPPRESSION_FOLDER, "suppression_2_ssm.npz"), name="With hybrids")
DEFAULT_SUPPRESSION: Suppression = NO_HYBRIDS_EXT
SUPPRESSIONS: list[Suppression] = [NO_HYBRIDS, NO_HYBRIDS_EXT, WITH_HYBRIDS]
