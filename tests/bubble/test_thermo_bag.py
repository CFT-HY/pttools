import unittest

from pttools.bubble import quantities


class ThermoBagTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_kappa(self):
        quantities.get_kappa(v_wall=0.5, alpha_n=0.1)

    @unittest.expectedFailure
    def test_kappa_de(self):
        quantities.get_kappa_de(v_wall=0.5, alpha_n=0.1)

    @unittest.expectedFailure
    def test_kappa_dq(self):
        quantities.get_kappa_dq(v_wall=0.5, alpha_n=0.1)

    @unittest.expectedFailure
    def test_ke_de_frac_bag(self):
        quantities.get_ke_de_frac_bag(v_wall=0.5, alpha_n=0.1)

    def test_ke_frac_bag(self):
        quantities.get_ke_frac_bag(v_wall=0.5, alpha_n=0.1)

    @unittest.expectedFailure
    def test_ke_frac_new_bag(self):
        quantities.get_ke_frac_new_bag(v_wall=0.5, alpha_n=0.1)

    def test_ubarf2(self):
        quantities.get_ubarf2(v_wall=0.5, alpha_n=0.1)

    @unittest.expectedFailure
    def test_ubarf2_new_bag(self):
        quantities.get_ubarf2_new_bag(v_wall=0.5, alpha_n=0.1)
