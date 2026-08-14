"""Unit tests for droplets"""

import unittest

from pttools.bubble import Bubble
from pttools.models import BagModel


class DropletTest(unittest.TestCase):
    @staticmethod
    def test_bhusal():
        r"""Benchmark point of :bhusal_2026:`\ ` eq. 60

        $$\Psi = 1 - \frac{\delta a}{a} = 0.84, \alpha_+ = 0.05
        \Rightarrow \xi_w = {v}_- = 0.42, -\xi_d = {v}_+ = 0.32$$
        """
        model = BagModel(a_s=1./0.84, a_b=1.)
        bubble = Bubble(model, v_wall=0.42, alpha_n=0.06)
        # The droplet module is not yet in the PTtools repository
        try:
            from examples.droplet.droplet import Droplet
            Droplet(bubble=bubble, v_wall=-0.32)
        except ImportError:
            pass
