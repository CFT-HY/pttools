"""Interpolate or extrapolate kinetic suppression data in the sound shell model."""

import enum
import logging
import os

import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

from pttools.omgw0.suppression.suppression_ssm_data.suppression_ssm_calculator import SUPPRESSION_FOLDER
import pttools.type_hints as th

logger = logging.getLogger(__name__)

# :TODO why is there a difference in the low alpha low vw region between hybrids and no hybrids data set?


class SuppressionMethod(enum.StrEnum):
    NONE = "none"
    NO_EXT = DEFAULT = "no_ext"
    EXT_CONSTANT = "ext_constant"


class Suppression:
    def __init__(
            self,
            v_walls: np.ndarray[int, np.float64],
            alpha_ns: np.ndarray[int, np.float64],
            suppressions: np.ndarray[int, np.float64],
            name: str = None):
        if not (v_walls.size == alpha_ns.size == suppressions.size):
            raise ValueError(
                f"Input arrays must have the same size. Got: {v_walls.size}, {alpha_ns.size}, {suppressions.size}")
        self.v_walls = v_walls
        self.alpha_ns = alpha_ns
        self.suppressions = suppressions
        self.name = name

        self.points = (self.v_walls, self.alpha_ns)
        self.alpha_n_min: float = np.min(self.alpha_ns)

    @classmethod
    def from_file(cls, path: str, name: str = None) -> "Suppression":
        data = np.load(path)
        return Suppression(v_walls=data["vw_sim"], alpha_ns=data["alpha_sim"], suppressions=data["sup_ssm"], name=name)

    def limits_str(self) -> str:
        return \
            f"{self.v_walls.min():.3f} < v_wall < {self.v_walls.max():.3f}, " \
            f"{self.alpha_n_min:.3f} < alpha_n < {self.alpha_ns.max():.3f}"

    def suppression(
            self,
            v_wall: th.FloatOrArr,
            alpha_n: th.FloatOrArr,
            method: SuppressionMethod,
            interpolation: th.Interpolation = "linear") -> th.FloatOrArr:
        """
        current simulation data bounds are
        0.24<vw<0.96
        0.05<alpha<0.67
        methods options :
        - "no_ext" = returns NaN outside of data region
        - "ext_constant" = extends the boundaries with a constant value
        - "ext_linear_Ubarf" = :TODO extend with linear Ubarf
        """
        is_scalar = np.isscalar(v_wall) and np.isscalar(alpha_n)

        if method == SuppressionMethod.NONE:
            return 1. if is_scalar else np.ones_like((v_wall.size, alpha_n.size))
        elif method not in (SuppressionMethod.NO_EXT, SuppressionMethod.EXT_CONSTANT):
            raise ValueError(f"Got invalid suppression method: {method}")

        if is_scalar:
            mesh = (v_wall, alpha_n)
            v_wall_mesh, alpha_n_mesh = v_wall, alpha_n
        else:
            mesh = np.meshgrid(v_wall, alpha_n)
            v_wall_mesh, alpha_n_mesh = mesh

        sup = interpolate.griddata(
            self.points,
            self.suppressions,
            mesh,
            method=interpolation
        )
        if method == SuppressionMethod.EXT_CONSTANT:
            sup[alpha_n_mesh < self.alpha_n_min] = 1
        if is_scalar:
            if np.isnan(sup):
                logger.warning(
                    "Got NaN as the suppression factor for v_wall=%s, alpha_n=%s. Are you outside the range?")
        else:
            sup[alpha_n_mesh > alpha_n_max_approx(v_wall_mesh)] = np.nan
        return sup


def alpha_n_max_approx(vw: th.FloatOrArr) -> th.FloatOrArr:
    """
    Approximate form of alpha_n_max function
    """
    return 1/3 * (1 + 3*vw**2) / (1 - vw**2)


def alpha_n_max(v_wall: th.FloatOrArr) -> th.FloatOrArr:
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
        v_walls: np.ndarray[int, np.float64],
        alpha_ns: np.ndarray[int, np.float64],
        suppressions: np.ndarray[int, np.float64]) -> tuple[np.ndarray[int, np.float64], np.ndarray[int, np.float64], np.ndarray[int, np.float64]]:
    """
    To improve the extrapolation of the suppression factor when later using gridata, first extend the
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
M1 = (0.5 - 0.34) / (0.44 - 0.24)  # dal/dvw
M2 = (0.67 - 0.5) / (0.56 - 0.44)
C1 = 0.34 - M1 * 0.24
C2 = 0.67000 - M2 * 0.56000

NO_HYBRIDS = Suppression.from_file(os.path.join(SUPPRESSION_FOLDER, "suppression_no_hybrids_ssm.npz"), name="No hybrids")
NO_HYBRIDS_EXT = Suppression(
    *extend(v_walls=NO_HYBRIDS.v_walls, alpha_ns=NO_HYBRIDS.alpha_ns, suppressions=NO_HYBRIDS.suppressions),
    name="No hybrids, extended"
)
WITH_HYBRIDS = Suppression.from_file(os.path.join(SUPPRESSION_FOLDER, "suppression_2_ssm.npz"), name="With hybrids")
DEFAULT = NO_HYBRIDS_EXT
SUPPRESSIONS = [NO_HYBRIDS, NO_HYBRIDS_EXT, WITH_HYBRIDS]
