r"""Tests for the exact sine and cosine transforms of piecewise-linear profiles."""

import typing as tp
import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models.bag import BagModel
from pttools.ssm.sin_transform_exact import (
    sharpen_padding,
    sound_shell_transforms_pl_parallel,
    sound_shell_transforms_pl_single,
)
from pttools.ssm.ssm import f_fp_l, lam
import pttools.type_hints as th

#: Piecewise-constant profile with jumps: 1 on [0.3, 0.5], 2 on [0.5, 0.6] and 0 elsewhere
XI: th.FloatArr1D = np.array([0, 0.3, 0.3, 0.5, 0.5, 0.6, 0.6, 1])
V: th.FloatArr1D = np.array([0, 0, 1, 1, 2, 2, 0, 0])
Z: th.FloatArr1D = np.logspace(0, 5, 200)


def integral(
        antiderivative: tp.Callable[[float, th.FloatArr1D], th.FloatArr1D],
        z: th.FloatArr1D) -> th.FloatArr1D:
    """Integral of the profile V times a function with the given antiderivative."""
    return (antiderivative(0.5, z) - antiderivative(0.3, z)) + 2 * (antiderivative(0.6, z) - antiderivative(0.5, z))


class SinTransformExactTest(unittest.TestCase):
    """Tests for :py:func:`pttools.ssm.sin_transform_exact.sound_shell_transforms_pl_parallel`."""

    @staticmethod
    def test_analytical() -> None:
        r"""The transforms of a piecewise-constant profile agree with the analytical results."""
        s_v, c_v, s_l = sound_shell_transforms_pl_parallel(Z, XI, V, V)
        # \int sin(z xi) = -cos(z xi) / z
        s_v_ref = integral(lambda x, z: -np.cos(z * x) / z, Z)
        # \int xi cos(z xi) = xi sin(z xi) / z + cos(z xi) / z^2
        c_v_ref = integral(lambda x, z: x * np.sin(z * x) / z + np.cos(z * x) / z**2, Z)
        # \int xi sin(z xi) = -xi cos(z xi) / z + sin(z xi) / z^2
        s_l_ref = integral(lambda x, z: -x * np.cos(z * x) / z + np.sin(z * x) / z**2, Z)
        np.testing.assert_allclose(s_v, s_v_ref, rtol=0, atol=1e-13)
        np.testing.assert_allclose(c_v, c_v_ref, rtol=0, atol=1e-13)
        np.testing.assert_allclose(s_l, s_l_ref, rtol=0, atol=1e-13)

    @staticmethod
    def test_linear() -> None:
        r"""The transform of $\xi$ on $[0, 1]$ is exact also for the segments that use the series."""
        # Uneven points, of which some are so close to each other that the series is used
        xi = np.sort(np.concatenate((np.linspace(0, 1, 101), 0.5 + np.logspace(-12, -6, 7))))
        s_v = sound_shell_transforms_pl_single(Z, xi, xi, xi)[0]
        np.testing.assert_allclose(s_v, (np.sin(Z) - Z * np.cos(Z)) / Z**2, rtol=0, atol=1e-11)

    @staticmethod
    def test_parallel() -> None:
        """The parallel and single-threaded versions give the same results."""
        for par, single in zip(
                sound_shell_transforms_pl_parallel(Z, XI, V, V),
                sound_shell_transforms_pl_single(Z, XI, V, V),
                strict=True):
            np.testing.assert_array_equal(par, single)

    @staticmethod
    def test_sharpen_padding() -> None:
        """The padding points of a fluid shell are moved onto the jumps."""
        model = BagModel.cached(alpha_n_min=0.01)
        for v_wall, alpha_n in ((0.5, 0.1), (0.7, 0.1), (0.9, 0.05)):
            bubble = Bubble.cached(model, v_wall=v_wall, alpha_n=alpha_n)
            xi = sharpen_padding(bubble.xi, bubble.v)
            # The jump to still fluid ahead of the shell
            assert xi[-2] == bubble.xi[-3]
            np.testing.assert_array_equal(np.delete(xi, [1, -2]), np.delete(bubble.xi, [1, -2]))
        # The wall of a subsonic deflagration
        bubble = Bubble.cached(model, v_wall=0.5, alpha_n=0.1)
        assert sharpen_padding(bubble.xi, bubble.v)[1] == bubble.xi[2]

    @staticmethod
    def test_fp() -> None:
        r"""The analytical $f'(z)$ agrees with finite differences on a dense $z$ array."""
        bubble = Bubble.cached(BagModel.cached(alpha_n_min=0.01), v_wall=0.7, alpha_n=0.1)
        lm = lam(v=bubble.v, w=bubble.w, e=bubble.e)
        z = np.logspace(np.log10(60), np.log10(200), 20001)
        f_val, fp_val, _ = f_fp_l(
            z=z, xi=bubble.xi, v=bubble.v, lam=lm, v_wall=bubble.v_wall, v_sh=bubble.v_sh, parallel=True)
        fp_fd = np.gradient(f_val, z)
        np.testing.assert_allclose(fp_val[1:-1], fp_fd[1:-1], rtol=0, atol=1e-5 * np.max(np.abs(fp_val)))


if __name__ == "__main__":
    unittest.main()
