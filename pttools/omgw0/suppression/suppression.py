"""
Interpolate or extrapolate kinetic suppression data in the sound shell model.

Created on wed 4 Aug  2021

@author: chloeg
"""

import enum
import logging
import os

import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

from pttools.omgw0.suppression.suppression_ssm_data.suppression_ssm_calculator import SUPPRESSION_FOLDER
import pttools.type_hints as th

logger = logging.getLogger(__name__)

SUPPRESSION_PATH = os.path.join(SUPPRESSION_FOLDER, "suppression_no_hybrids_ssm.npz")

# :TODO why is there a difference in the low alpha low vw region between hybrids and no hybrids data set?

ssm_sup_data = np.load(SUPPRESSION_PATH)
vws_sim = ssm_sup_data["vw_sim"]
alphas_sim = ssm_sup_data["alpha_sim"]
ssm_sup = ssm_sup_data["sup_ssm"]

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
vws_sim_ext = np.concatenate(([0.24], vws_sim))
alphas_sim_ext = np.concatenate(([ssm_sup_vw_0_24_alphas_ext[0]], alphas_sim))
ssm_sup_ext = np.concatenate(([ssm_sup_vw_0_24_ext[0]], ssm_sup))


# Todo: Update this to enum.StrEnum, when Python 3.11 becomes the oldest supported version.
class SuppressionMethod(str, enum.Enum):
    NONE = "none"
    NO_EXT = DEFAULT = "no_ext"
    EXT_CONSTANT = "ext_constant"


def alpha_n_max_approx(vw: th.FloatOrArr) -> th.FloatOrArr:
    """
    Approximate form of alpha_n_max function
    """
    return 1/3 * (1 + 3*vw**2) / (1 - vw**2)


def alpha_n_max(vw: float) -> float:
    #  vw ,al
    # [0.24000,0.34000]
    # [0.44000, 0.50000]
    # [0.56000,0.67000]
    if vw < 0.44:
        m1 = (0.5 - 0.34) / (0.44 - 0.24)  # dal/dvw
        c1 = 0.34 - m1 * 0.24
        return m1 * vw + c1
    m2 = (0.67 - 0.5)/(0.56 - 0.44)
    c2 = 0.67000 - m2 * 0.56000
    return m2 * vw + c2


def get_suppression_factor(vw: float, alpha: float, method: SuppressionMethod = SuppressionMethod.DEFAULT):
    """
    current simulation data bounds are
    0.24<vw<0.96
    0.05<alpha<0.67
    methods options :
    - "no_ext" = returns NaN outside of data region
    - "ext_constant" = extends the boundaries with a constant value
    - "ext_linear_Ubarf" = :TODO extend with linear Ubarf
    """
    if method == SuppressionMethod.NONE:
        return 1
    if alpha > alpha_n_max_approx(vw):
        supp_factor = np.nan
    else:
        vv_n, aa_n = np.meshgrid(vw, alpha)
        if method == SuppressionMethod.NO_EXT:
            supp_factor = interpolate.griddata((vws_sim, alphas_sim), ssm_sup, (vv_n, aa_n), method="linear")[0]
        elif method == SuppressionMethod.EXT_CONSTANT:
            if alpha < np.min(alphas_sim_ext):
                supp_factor = 1
            else:
                supp_factor = interpolate.griddata((vws_sim_ext, alphas_sim_ext), ssm_sup_ext, (vv_n, aa_n), method="linear")[0]
        else:
            raise ValueError(f"Got invalid suppression method: {method}")
    if np.isnan(supp_factor):
        logger.warning("Got NaN as the suppression factor. Are you outside the range?")
    return supp_factor


def get_suppression_factor_with_hybrids(vws: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    0.24<vw<0.96
    0.05<alpha<0.67
    """
    vv_n, aa_n = np.meshgrid(vws, alphas)
    suppression_path_hybrids = os.path.join(os.path.dirname(__file__), "suppression_2_ssm.npz")
    ssm_sup_data_hybrids = np.load(suppression_path_hybrids)
    vws_sim_hybrids = ssm_sup_data_hybrids["vw_sim"]
    alphas_sim_hybrids = ssm_sup_data_hybrids["alpha_sim"]
    ssm_sup_hybrids = ssm_sup_data_hybrids["sup_ssm"]

    return interpolate.griddata((vws_sim_hybrids, alphas_sim_hybrids), ssm_sup_hybrids, (vv_n, aa_n), method="linear")
