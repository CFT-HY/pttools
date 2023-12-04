r"""Compare results with those from the code of :giese_2021:`\ `"""

import unittest
import typing as tp

import numpy as np
import pytest

from pttools import models
from pttools.analysis.parallel import create_bubbles
from pttools.bubble.bubble import Bubble
from pttools.speedup import conditional_decorator, IS_OSX
from tests.utils import assert_allclose


def assert_kappa(css2: float, csb2: float, kappa_ref: np.ndarray, rtol: float = 1e-7, atol: float = 0):
    r"""Compare kappa results to those of figure 2 of :giese_2021:`\ `"""
    alpha_thetabar_ns = np.array([0.01, 0.1, 0.3])
    v_walls = np.linspace(0.2, 0.9, 8, endpoint=True)
    model = models.ConstCSModel(css2=css2, csb2=csb2, a_s=5, a_b=1, V_s=1)
    bubbles, kappas = create_bubbles(
        model=model, v_walls=v_walls, alpha_ns=alpha_thetabar_ns,
        func=get_kappa, bubble_kwargs={"theta_bar": True, "allow_invalid": True}, allow_bubble_failure=True)

    assert_allclose(kappas, kappa_ref, rtol=rtol, atol=atol)


def compare(
        model: models.Model,
        alpha_ns: tp.Union[np.ndarray, tp.List[float]],
        v_walls: tp.Union[np.ndarray, tp.List[float]],
        ref: tp.Union[np.ndarray, tp.List[float]], rtol: float):
    data = np.zeros_like(v_walls)
    for i, (alpha_n, v_wall) in enumerate(zip(alpha_ns, v_walls)):
        bubble = Bubble(model=model, v_wall=v_wall, alpha_n=alpha_n)
        data[i] = bubble.kappa
    assert_allclose(data, ref, rtol=rtol)


def get_kappa(bubble: Bubble) -> float:
    if not bubble.solved:
        return np.nan
    return bubble.kappa


get_kappa.return_type = float
get_kappa.fail_value = np.nan


class GieseTest(unittest.TestCase):
    @staticmethod
    def test_bag():
        model = models.BagModel(a_s=1.1, a_b=1, V_s=1)
        alpha_ns = [0.578, 0.151]
        v_walls = [0.5, 0.7]
        kappa_ref = [0.6080227349254361, 0.5013547591468864]
        # compare(model, alpha_ns, v_walls, kappa_ref, rtol=0.037)
        # The tolerance had to be increased when re-solving of thin shells was implemented.
        compare(model, alpha_ns, v_walls, kappa_ref, rtol=0.039)

    @staticmethod
    def test_const_cs():
        model = models.ConstCSModel(css2=1/3, csb2=(1/np.sqrt(3) - 0.01)**2, a_s=1.5, a_b=1, V_s=1)
        alpha_ns = [0.578, 0.151]
        v_walls = [0.5, 0.7]
        kappa_ref = [0.5962033181328875, 0.46887359879796786]
        # compare(model, alpha_ns, v_walls, kappa_ref, rtol=0.077)
        # The tolerance had to be increased when re-solving of thin shells was implemented.
        compare(model, alpha_ns, v_walls, kappa_ref, rtol=0.085)

    @pytest.mark.xfail(IS_OSX, reason="Bug on macOS")
    def test_kappa33(self):
        kappa_ref = np.array([
            [0.00741574, 0.01450964, 0.02653822, 0.05782794, 0.18993211, 0.04904255, 0.0265001, 0.01816217],
            [0.06921508, 0.12306195, 0.18896878, 0.28120044, 0.41124208, 0.44902447, 0.23408969, 0.15686063],
            [0.18355195, 0.28898783, 0.38244706, 0.47270278, 0.56546988, 0.6231619, 0.57442573, 0.36606867]
        ])
        assert_kappa(css2=1/3, csb2=1/3, kappa_ref=kappa_ref, rtol=7.3e-3)

    @staticmethod
    @unittest.expectedFailure
    def test_kappa34():
        kappa_ref = np.array([
            [0.00709227, 0.01341795, 0.02375007, 0.05032617, 0.04033708, 0.01933255, 0.01273443, 0.00937875],
            [0.06570444, 0.11244944, 0.16679036, 0.24170456, 0.28645206, 0.18837795, 0.11564567, 0.0850799],
            [0.17210839, 0.25999769, 0.3327127, 0.40143091, 0.44930404, 0.4318333, 0.29546582, 0.21279852]
        ])
        assert_kappa(css2=1/3, csb2=1/4, kappa_ref=kappa_ref, rtol=0.61)

    @staticmethod
    @unittest.expectedFailure
    def test_kappa43():
        kappa_ref = np.array([
            [0.00730452, 0.01543951, 0.0342464, 0.1244067, np.nan, 0.04904255, 0.0265001, 0.01816217],
            [0.06740045, 0.12497735, 0.2058207, 0.32704976, 0.47932218, 0.50230337, 0.23408969, 0.15686063],
            [0.17642267, 0.28297756, 0.38474954, 0.48622994, 0.58952004, 0.65276938, 0.59256005, 0.36606867]
        ])
        assert_kappa(css2=1/4, csb2=1/3, kappa_ref=kappa_ref, rtol=0.29)

    @staticmethod
    @unittest.expectedFailure
    def test_kappa44():
        kappa_ref = np.array([
            [0.00700235, 0.01434407, 0.03086914, 0.11071288, 0.04033708, 0.01933255, 0.01273443, 0.00937875],
            [0.06410723, 0.11460043, 0.18265583, 0.28383528, 0.34206562, 0.18837795, 0.11564567, 0.0850799],
            [0.16570685, 0.25535541, 0.33629884, 0.41548919, 0.4724849, 0.45536877, 0.29546582, 0.21279852]
        ])
        assert_kappa(css2=1/4, csb2=1/4, kappa_ref=kappa_ref, rtol=0.62)
