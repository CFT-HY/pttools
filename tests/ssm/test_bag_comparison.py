"""Tests for comparing the results of the Spectrum class to the old bag model interface"""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.bubble.thermo import ubarf2
from pttools.bubble import CS2_BAG_SCALAR_PTR, DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG
from pttools.bubble.thermo_bag import de_from_w_bag
from pttools.models import BagModel
from pttools.ssm import SSMSpectrum, pow_spec
from pttools import ssm
import pttools.type_hints as th
from pttools.utils.assertions import assert_allclose


class SpectrumTest(unittest.TestCase):
    """Tests for comparing the results of the Spectrum class to the old bag model interface"""
    V_WALLS: th.FloatArr1D = np.array([0.5, 0.7, 0.77])
    ALPHA_NS: th.FloatArr1D = np.array([0.578, 0.151, 0.091])
    model: BagModel
    bubbles: list[Bubble]
    spectra: list[SSMSpectrum]
    spectra_lambda: list[SSMSpectrum]
    z: th.FloatArr1D

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = BagModel(a_s=1.1, a_b=1, V_s=2)
        cls.bubbles = [
            Bubble(cls.model, v_wall=cls.V_WALLS[i], alpha_n=cls.ALPHA_NS[i])
            for i in range(cls.V_WALLS.size)
        ]
        cls.spectra = [SSMSpectrum(bubble) for bubble in cls.bubbles]
        cls.spectra_lambda = []
        for bubble in cls.bubbles:
            spectrum = SSMSpectrum(bubble, compute=False)
            spectrum.compute(lambda_correction=True)
            cls.spectra_lambda.append(spectrum)
        cls.z = cls.spectra[0].y

    def test_de(self):
        # The arrays have different sizes and cannot therefore be combined to a 2D array
        de_bag = [
            de_from_w_bag(
                w=bubble.w, xi=bubble.xi, v_wall=bubble.v_wall, alpha_n=bubble.alpha_n,
                df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD)
            for bubble in self.bubbles
        ]
        e = [
            bubble.model.e(bubble.w, bubble.phase)
            for bubble in self.bubbles
        ]
        de = [ei - ei[-1] for ei in e]
        for de_i, de_bag_i in zip(de, de_bag):
            assert_allclose(de_i, de_bag_i)

    def test_a2(self):
        a2_old = np.array([
            ssm.a2_e_conserving_bag(
                self.z, v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i],
                cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
                v_ip=bubble.v, w_ip=bubble.w, xi=bubble.xi,
                v_sh=bubble.v_sh
            )[0]
            for i, bubble in enumerate(self.bubbles)
        ])
        a2_new = np.array([
            ssm.A2_e_conserving(
                v=bubble.v, w=bubble.w, xi=bubble.xi, e=bubble.e, z=self.z,
                v_wall=bubble.v_wall, v_sh=bubble.v_sh, cs=ssm.CS0)[0]
            for bubble in self.bubbles
        ])
        # a2_new2 = np.array([spectrum.a2 for spectrum in self.spectra])
        assert_allclose(a2_new, a2_old)
        # assert_allclose(a2_new2, a2_old)

    def test_spec_den_v(self):
        """This test has lambda_correction=True,
        as disabling it would require a somewhat looser tolerance for some of the points.
        """
        old = np.array([
            ssm.spec_den_v_bag(
                self.z,
                (v_wall, alpha_n),
                lambda_correction=True
            )
            for v_wall, alpha_n in zip(self.V_WALLS, self.ALPHA_NS)
        ])
        new = np.array([
            spectrum.spec_den_v * \
            ubarf2(
                v=spectrum.bubble.v,
                w=spectrum.bubble.w,
                xi=spectrum.bubble.xi,
                v_wall=spectrum.bubble.v_wall,
                ek_bva=spectrum.bubble.kinetic_energy_density,
                w_bar=spectrum.bubble.wn
            )
            for spectrum in self.spectra_lambda
        ])
        assert_allclose(
            new, old,
            # This tolerance had to be loosened when upgrading the implementations of A2, Gamma, ubarf2 and w_bar.
            # rtol=0.283
            rtol=0.329
        )

    def test_gw(self):
        """This test has lambda_correction=True,
        as disabling it would require a somewhat looser tolerance for some of the points.
        This may be either due to numerical differences, or lambda_correction might hide some other difference.
        """
        old = np.array([
            ssm.power_gw_bag(
                self.z,
                (v_wall, alpha_n),
                lambda_correction=True
            ) * spectrum.source_lifetime_factor
            for v_wall, alpha_n, spectrum in zip(self.V_WALLS, self.ALPHA_NS, self.spectra_lambda)
        ])
        new = np.array([
            pow_spec(z=spectrum.y, spec_den=spectrum.spec_den_gw_ssm) * \
            ubarf2(
                v=spectrum.bubble.v,
                w=spectrum.bubble.w,
                xi=spectrum.bubble.xi,
                v_wall=spectrum.bubble.v_wall,
                ek_bva=spectrum.bubble.kinetic_energy_density,
                w_bar=spectrum.bubble.wn
            ) ** 2
            for spectrum in self.spectra_lambda
        ])
        assert_allclose(
            new, old,
            # This tolerance had to be loosened when upgrading the implementations of A2, Gamma, ubarf2 and w_bar.
            # rtol=0.519
            rtol=0.551
        )


if __name__ == "__main__":
    unittest.main()
