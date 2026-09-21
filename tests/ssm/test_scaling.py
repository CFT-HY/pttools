"""Tests for the scaling factors of the GW power spectrum."""

import unittest

import numpy as np

from pttools.ssm import barotropic, scaling


class ScalingTest(unittest.TestCase):
    """Tests for pttools.ssm.scaling."""

    def test_H_star_eta_sh(self):
        self.assertAlmostEqual(scaling.H_star_eta_sh(r_star=0.1, ubarf=0.05), 2.)
        # In the approximation ubarf = sqrt(K) the two functions agree.
        self.assertAlmostEqual(scaling.H_star_eta_sh_approx(r_star=0.1, K=0.05**2), 2.)

    def test_H_star_eta_v_limits(self):
        """Short sources: H_* eta_v -> H_* Delta eta_v. Long sources: H_* eta_v -> H_* eta_* / l(nu)."""
        for nu in (0., 0.1, 0.3):
            with self.subTest(nu=nu):
                # Short source
                r_star, ubarf = 1e-6, 0.1
                eta_ratio = barotropic.eta_ratio(ubarf=ubarf, r_star=r_star, nu=nu)
                upsilon = barotropic.source_lifetime_factor(ubarf=ubarf, r_star=r_star, nu=nu)
                # H_* Delta eta_v = H_* eta_* * Delta eta_v / eta_* = (1 + nu) eta_ratio = r_* / ubarf
                self.assertAlmostEqual(scaling.H_star_eta_v(upsilon, nu) / ((1 + nu) * eta_ratio), 1, places=4)
                self.assertAlmostEqual(
                    scaling.H_star_eta_v(upsilon, nu) / scaling.H_star_eta_sh(r_star, ubarf), 1, places=4)
                # Long source
                upsilon = barotropic.source_lifetime_factor(ubarf=0.01, r_star=1e4, nu=nu)
                self.assertAlmostEqual(scaling.H_star_eta_v(upsilon, nu), (1 + nu) / (1 + 2 * nu), places=5)

    def test_H_star_eta_v_radiation(self):
        """For nu = 0 and N_sh = 1: H_* eta_v = 1 - 1 / (1 + x), x = H_* eta_sh."""
        r_star, ubarf = 0.3, 0.2
        x = scaling.H_star_eta_sh(r_star=r_star, ubarf=ubarf)
        upsilon = barotropic.source_lifetime_factor(ubarf=ubarf, r_star=r_star, N_sh=1., nu=0.)
        self.assertAlmostEqual(scaling.H_star_eta_v(upsilon, nu=0.), 1 - 1 / (1 + x))

    def test_H_star_eta_v_old_is_cosmic_time_version(self):
        """The old formula 1 - 1/sqrt(1 + 2x) is Upsilon_1 = 1 - eta_*/eta_end
        with the source duration x measured in cosmic time in a radiation-dominated Universe,
        where eta/eta_* = sqrt(t/t_*) and H_* t_* = 1/2.
        """
        for x in (0.01, 0.5, 2., 50.):
            with self.subTest(x=x):
                eta_end_over_eta_star = np.sqrt(1 + 2 * x)
                upsilon_1 = barotropic.Upsilon(r=1 / eta_end_over_eta_star, l=1.)
                self.assertAlmostEqual(scaling.H_star_eta_v_old(x), upsilon_1)
        # Limits
        self.assertAlmostEqual(scaling.H_star_eta_v_old(1e-6) / 1e-6, 1., places=4)
        self.assertAlmostEqual(scaling.H_star_eta_v_old(1e8), 1., places=3)

    def test_H_star_eta_v_old2(self):
        self.assertEqual(scaling.H_star_eta_v_old2(0.5), 0.5)
        self.assertEqual(scaling.H_star_eta_v_old2(3.), 1.)

    def test_J(self):
        r_star, ubarf, nu = 0.1, 0.2, 0.05
        upsilon = barotropic.source_lifetime_factor(ubarf=ubarf, r_star=r_star, nu=nu)
        J_ref = r_star * (1 + nu) * upsilon
        self.assertAlmostEqual(scaling.J(r_star=r_star, H_star_eta_v=scaling.H_star_eta_v(upsilon, nu)), J_ref)
        self.assertAlmostEqual(scaling.J_full(r_star=r_star, ubarf=ubarf, nu=nu), J_ref)
        K = ubarf**2
        self.assertAlmostEqual(
            scaling.J_old(r_star=r_star, K=K),
            r_star * (1 - 1 / np.sqrt(1 + 2 * r_star / np.sqrt(K)))
        )


if __name__ == "__main__":
    unittest.main()
