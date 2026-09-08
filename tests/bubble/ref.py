"""Reference values from literature."""

from abc import ABC

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.models.bag import BagModel
from pttools.models.model import Model
from pttools.type_hints import FloatArr1D


class Reference(ABC):
    """Reference values from literature."""

    bubbles: list[Bubble]
    MODEL: Model = BagModel(a_s=1.1, a_b=1, V_s=1)
    # Input parameters
    ALPHA_NS: FloatArr1D
    V_WALLS: FloatArr1D
    # Reference values
    ALPHA_PLUS_REF: FloatArr1D
    KAPPA_REF: FloatArr1D
    BVA_KE_FRAC_REF: FloatArr1D
    OMEGA_REF: FloatArr1D
    # UBARF_REF: FloatArr1D
    V_SH_REF: FloatArr1D


class RefBag(Reference):
    r"""Reference values generated with PTtools."""

    ALPHA_NS = np.array(np.repeat([0.1, 0.2, 0.3], 3))
    V_WALLS = np.array(np.tile([0.3, 0.7, 0.8], 3))

    BVA_KE_FRAC_REF = np.array([
        1.12012152e-02, 4.10228822e-02, 2.13351667e-02,
        3.58402474e-02, 9.40854610e-02, 7.71674313e-02,
        6.67623755e-02, 1.44117154e-01, 1.33526554e-01
    ])
    KAPPA_REF = np.array([0.1227, 0.4512, 0.2346, 0.2141, 0.5645, 0.4630, 0.2881, 0.6245, 0.5786])
    OMEGA_REF = np.array([0.8773, 0.5574, 0.7683, 0.7854, 0.4408, 0.5496, 0.7113, 0.3794, 0.4305])


class RefLectureNotes(Reference):
    r"""Reference values from :notes:`\ ` fig. 15."""

    ALPHA_NS = np.array([0.1, 0.1, 0.1])
    V_WALLS = np.array([0.4, 0.7, 0.8])

    ALPHA_PLUS_REF = np.array([0.078, 0.037, 0.100])
    BVA_KE_FRAC_REF = np.array([0.0172, 0.0411, 0.0213])
    KAPPA_REF = np.array([0.189, 0.452, 0.235])
    OMEGA_REF = np.array([0.815, 0.559, 0.769])
    UBARF_REF = np.array([0.119, 0.184, 0.133])
    V_SH_REF = np.array([0.579, 0.715, 0.800])


class RefHindmarshHijazi(Reference):
    r"""Reference values from :gw_pt_ssm:`\ ` fig. 10."""

    ALPHA_NS = np.array([0.578, 0.151, 0.091])
    V_WALLS = np.array([0.5, 0.7, 0.77])

    ALPHA_PLUS_REF = np.array([0.263, 0.052, 0.091])
    BVA_KE_FRAC_REF = np.array([0.223, 0.0684, 0.022])
    KAPPA_REF = np.array([0.610, 0.522, 0.264])
    OMEGA_REF = np.array([0.395, 0.491, 0.744])
    # This reference does not have Ubarf values.
    V_SH_REF = np.array([0.641, 0.721, 0.770])
