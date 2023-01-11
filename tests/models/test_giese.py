"""Compare results with those from the code of Giese et al., 2021"""

import unittest

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools import models
from tests.utils import assert_allclose


def compare(model, alpha_ns, v_walls, ref, rtol):
    data = []
    for alpha_n, v_wall in zip(alpha_ns, v_walls):
        bubble = Bubble(model=model, v_wall=v_wall, alpha_n=alpha_n)
        bubble.solve()
        data.append(bubble.kappa)
    assert_allclose(data, ref, rtol=rtol)


class GieseTest(unittest.TestCase):
    def test_bag(self):
        model = models.BagModel(a_s=1.1, a_b=1, V_s=1)
        alpha_ns = [0.578, 0.151]
        v_walls = [0.5, 0.7]
        ref = [0.6080227349254361, 0.5013547591468864]
        compare(model, alpha_ns, v_walls, ref, rtol=0.037)

    def test_const_cs(self):
        model = models.ConstCSModel(css2=1/3, csb2=(1/np.sqrt(3) - 0.01)**2, a_s=1.5, a_b=1, V_s=1)
        alpha_ns = [0.578, 0.151]
        v_walls = [0.5, 0.7]
        ref = [0.5962033181328875, 0.46887359879796786]
        compare(model, alpha_ns, v_walls, ref, rtol=0.077)
