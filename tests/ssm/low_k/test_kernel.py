r"""Tests for the kernel functions of the low-k approximation."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from pttools.ssm.barotropic import Upsilon
from pttools.ssm.const import CS0
from pttools.ssm.low_k import kernel_int_bracket, kernel_low, kernel_low_bag
from pttools.utils import assert_allclose
from tests.ssm.low_k.common import NUS, kernel_low_numerical

TAU_STAR: float = 10.
TAU_END: float = 30.


class KernelLowTest(unittest.TestCase):
    r"""Tests for the low-frequency kernel $\Delta_\text{low}$."""

    def test_green_function(self) -> None:
        r"""Compare the kernel with a numerical integration over the Green's function.

        The kernel of :giombi_2026:`\ ` eq. 2.38 is integrated numerically
        using the exact Green's function of :giombi_2026:`\ ` eq. 2.17
        instead of its small-argument expansion eq. 2.40.
        The deviation is of order $(z \tau_\text{end})^{1 + 2\nu}$, as given by eq. 2.40.
        """
        z = 1e-4
        for nu in NUS:
            with self.subTest(nu=nu):
                assert_allclose(
                    kernel_low(z=z, nu=nu, tau_star=TAU_STAR, tau_end=TAU_END),
                    kernel_low_numerical(z=z, nu=nu, tau_star=TAU_STAR, tau_end=TAU_END),
                    rtol=1e-3
                )

    def test_green_function_bag(self) -> None:
        r"""Compare the bag model kernel with a numerical integration over the Green's function.

        For $\nu = 0$ the Green's function is $G_k(\eta, \eta_1) = \frac{1}{k} \sin(k(\eta - \eta_1))$
        (:giombi_2026:`\ ` eq. 2.18),
        and the non-oscillating part of the kernel is
        $\frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$
        (:giombi_2024_cs:`\ ` eq. 3.4).
        """
        assert_allclose(
            kernel_low_bag(tau_star=TAU_STAR, tau_end=TAU_END),
            kernel_low_numerical(z=1e-4, nu=0., tau_star=TAU_STAR, tau_end=TAU_END),
            rtol=1e-3
        )

    def test_reference(self) -> None:
        r"""Test the kernel against the limit $z \rightarrow 0$ of :giombi_2026:`\ ` eq. C.6a.

        $$\Delta_\text{sw}^\text{lo} \rightarrow
        \left( \frac{k \eta_*}{2} \right)^{-2\nu} \frac{\Gamma^2 \left( \frac{1}{2} + \nu \right)}{4\pi}
        \frac{1}{4\nu^2} \left[ 1 - \left( \frac{\tau_*}{\tau_\text{end}} \right)^{2\nu} \right]^2$$
        """
        z = np.logspace(-4, -1, 5)
        for nu in NUS:
            with self.subTest(nu=nu):
                ref = (z * TAU_STAR / 2)**(-2 * nu) * gamma(0.5 + nu)**2 / (4 * np.pi) \
                    * (1 - (TAU_STAR / TAU_END)**(2 * nu))**2 / (4 * nu**2)
                assert_allclose(kernel_low(z=z, nu=nu, tau_star=TAU_STAR, tau_end=TAU_END), ref, rtol=1e-13)

    def test_bag_limit(self) -> None:
        r"""Test that the kernel converges to the bag model kernel as $\nu \rightarrow 0$.

        :giombi_2024_cs:`\ ` eq. 3.4
        """
        bag = kernel_low_bag(tau_star=TAU_STAR, tau_end=TAU_END)
        assert_allclose(bag, 0.25 * np.log(TAU_END / TAU_STAR)**2, rtol=1e-14)
        assert_allclose(kernel_low(z=1., nu=0., tau_star=TAU_STAR, tau_end=TAU_END), bag, rtol=1e-14)
        for nu in (1e-3, 1e-6, 1e-9):
            with self.subTest(nu=nu):
                # With z tau_* = 2 the factor (z tau_* / 2)^(-2 nu) is unity,
                # and the corrections from Gamma^2(1/2 + nu) and Upsilon_{2 nu}^2 are of order nu.
                assert_allclose(
                    kernel_low(z=2 / TAU_STAR, nu=nu, tau_star=TAU_STAR, tau_end=TAU_END), bag, rtol=10 * nu)

    def test_long_source(self) -> None:
        r"""Test the limit of a long-lasting source $\tau_\text{end} \gg \tau_*$.

        For $\nu > 0$ the kernel converges to a finite value,
        as $\Upsilon_{2\nu} \rightarrow 1 / (2\nu)$, whereas for $\nu = 0$ it grows logarithmically without bound.
        :giombi_2024_cs:`\ ` sec. 3.1,
        :giombi_2026:`\ ` sec. 3.1.
        """
        z = 0.1
        for nu in NUS:
            with self.subTest(nu=nu):
                limit = (z * TAU_STAR / 2)**(-2 * nu) * gamma(0.5 + nu)**2 / (4 * np.pi) / (4 * nu**2)
                tau_ends = TAU_STAR * np.logspace(1, 6, 6)
                kernels = np.asarray(kernel_low(z=z, nu=nu, tau_star=TAU_STAR, tau_end=tau_ends))
                assert np.all(np.diff(kernels) > 0)
                assert np.all(kernels < limit)
                assert_allclose(kernels[-1], limit, rtol=4 * 1e-6**(2 * nu))
        bag = np.asarray(kernel_low_bag(tau_star=TAU_STAR, tau_end=TAU_STAR * np.array([1e2, 1e4])))
        assert_allclose(bag[1], 4 * bag[0])

    def test_scaling(self) -> None:
        r"""Test that $\tilde{P}_\text{gw}^\text{low} \propto z^{-2\nu}$.

        Then $\mathcal{P}_\text{gw} \propto k^{3 - 2\nu}$.

        :giombi_2026:`\ ` eq. 3.7
        """
        z = np.logspace(-4, -1, 10)
        for nu in NUS:
            with self.subTest(nu=nu):
                kernels = kernel_low(z=z, nu=nu, tau_star=TAU_STAR, tau_end=TAU_END)
                assert_allclose(np.polyfit(np.log(z), np.log(z**3 * kernels), 1)[0], 3 - 2 * nu)

    def test_short_source(self) -> None:
        r"""Test that the kernel grows quadratically with the source duration when $\Delta \eta_\text{v} \ll \eta_*$.

        :giombi_2024_cs:`\ ` sec. 5
        """
        delta = 1e-4
        tau_end = TAU_STAR * (1 + delta)
        for nu in (0, *NUS):
            with self.subTest(nu=nu):
                kernel = kernel_low(z=2 / TAU_STAR, nu=nu, tau_star=TAU_STAR, tau_end=tau_end) \
                    if nu else kernel_low_bag(tau_star=TAU_STAR, tau_end=tau_end)
                assert_allclose(kernel, gamma(0.5 + nu)**2 / (4 * np.pi) * delta**2, rtol=2 * delta)

    def test_upsilon_limit(self) -> None:
        r"""Test that $\Upsilon_\ell(r)$ is continuous at $\ell = 0$, where $\Upsilon_0(r) = -\ln r$."""
        r = np.array([0.1, 0.5, 2.])
        assert_allclose(Upsilon(r=r, l=0.), -np.log(r), rtol=1e-15)
        for l in (1e-12, -1e-12, 1e-6):  # noqa: E741
            assert_allclose(Upsilon(r=r, l=l), -np.log(r), rtol=2 * abs(l))
        for l in (-1.5, 0.5, 1., 3.):  # noqa: E741
            assert_allclose(Upsilon(r=r, l=l), (1 - r**l) / l, rtol=1e-14)


class KernelIntTest(unittest.TestCase):
    r"""Tests for the intermediate-frequency kernel."""

    @staticmethod
    def mu_integral(cs: float) -> float:
        r"""Angular integral of the long-lasting intermediate-frequency kernel.

        $$\int_{-1}^1 d\mu (1 - \mu^2)^2 \frac{1 + c_s^2 \mu^2}{(1 - c_s^2 \mu^2)^2}$$
        :giombi_2024_cs:`\ ` eq. 3.9c, 3.10
        """
        return quad(
            lambda mu: (1 - mu**2)**2 * (1 + cs**2 * mu**2) / (1 - cs**2 * mu**2)**2,
            -1, 1, epsabs=0, epsrel=1e-13
        )[0]

    def test_mu_integral(self) -> None:
        r"""Test the bracket against the numerical integration over the polar angle.

        Inserting the kernel of :giombi_2024_cs:`\ ` eq. 3.9c into eq. 3.10 gives
        $$\tilde{P}_\text{gw}^\text{int} = \frac{\mathcal{I}_v}{2 \tau_* z^2}
        \int_{-1}^1 d\mu (1 - \mu^2)^2 \frac{1 + c_s^2 \mu^2}{(1 - c_s^2 \mu^2)^2},$$
        which should be equal to eq. 3.11.
        """
        for cs in (0.05, 0.3, 0.5, CS0, 0.9):
            with self.subTest(cs=cs):
                assert_allclose(4 / (3 * cs**4) * kernel_int_bracket(cs=cs), self.mu_integral(cs) / 2, rtol=1e-8)

    def test_series(self) -> None:
        r"""Test the bracket against its power series.

        Using the series of $\operatorname{arctanh}(c_s)$, the bracket of :giombi_2024_cs:`\ ` eq. 3.11 is
        $$3 - 2 c_s^2 - \frac{3}{c_s}(1 - c_s^2) \operatorname{arctanh}(c_s)
        = \sum_{n=2}^\infty \frac{6}{4n^2 - 1} c_s^{2n}
        = \frac{2}{5} c_s^4 + \frac{6}{35} c_s^6 + \mathcal{O}(c_s^8).$$
        Therefore, as $c_s \rightarrow 0$, $\frac{4}{3 c_s^4} [\ldots] \rightarrow \frac{8}{15}
        = \frac{1}{2} \int_{-1}^1 d\mu (1 - \mu^2)^2$, which corresponds to $\Delta_\text{int}$ without the
        $c_s$-dependent factor of eq. 3.9c.
        """
        cs = np.array([0.05, 0.2, 0.5, CS0])
        n = np.arange(2, 200)[:, np.newaxis]
        series = np.sum(6 / (4 * n**2 - 1) * cs**(2 * n), axis=0)
        # The precision is limited by the cancellation of the leading terms for small cs.
        assert_allclose(kernel_int_bracket(cs=cs), series, rtol=1e-10)
        # The relative correction is 3/7 c_s^2.
        assert_allclose(4 / (3 * 0.05**4) * kernel_int_bracket(cs=0.05), 8 / 15 * (1 + 3 / 7 * 0.05**2), rtol=1e-5)

    def test_radiation(self) -> None:
        r"""Test the value for radiation $c_s^2 = 1/3$.

        $$3 - \frac{2}{3} - 2\sqrt{3} \operatorname{arctanh} \left( \frac{1}{\sqrt{3}} \right)
        = \frac{7}{3} - \sqrt{3} \ln \left( 2 + \sqrt{3} \right)$$
        """
        assert_allclose(kernel_int_bracket(cs=CS0), 7 / 3 - np.sqrt(3) * np.log(2 + np.sqrt(3)), rtol=1e-12)
