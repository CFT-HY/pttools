r"""Exact sine and cosine transforms of piecewise-linear profiles for the Sound Shell Model."""

import typing as tp

import numba
import numpy as np

from pttools.speedup import njit, njit_parallel_pair
import pttools.type_hints as th

#: Below this $z \Delta\xi$, the segment terms are computed with a series to avoid cancellation.
Z_DXI_SERIES: float = 1e-3


def _sound_shell_transforms_pl(
        z: th.FloatArr1D,
        xi: th.FloatArr1D,
        v: th.FloatArr1D,
        lam: th.FloatArr1D) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""Exact sine and cosine transforms of the piecewise-linear interpolants of the fluid profiles.

    Computes for each $z$
    $$S_{v}(z) = \int d\xi\, v(\xi) \sin(z\xi), \quad
    C_{v}(z) = \int d\xi\, \xi v(\xi) \cos(z\xi), \quad
    S_{\lambda}(z) = \int d\xi\, \xi \lambda(\xi) \sin(z\xi),$$
    where $v$, $\xi v$ and $\xi \lambda$ are interpolated linearly between the points $\xi$.
    These are exact for the interpolants at all $z$, unlike the trapezoidal rule,
    which requires $z \Delta\xi \ll 1$, and unlike the approximation of
    :py:func:`pttools.ssm.sin_transform_approx.sin_transform_approx`, which neglects the $O(1/z^2)$ terms.
    Consecutive points with the same $\xi$ are discontinuities, and they are included exactly.

    For a segment $[a, b]$ with the linear function $g$ and its slope $s = (g_b - g_a)/(b - a)$,
    $$\int_a^b g \sin(z\xi) d\xi = -\frac{g_b \cos(zb) - g_a \cos(za)}{z} + s \frac{\sin(zb) - \sin(za)}{z^2},$$
    $$\int_a^b g \cos(z\xi) d\xi = \frac{g_b \sin(zb) - g_a \sin(za)}{z} + s \frac{\cos(zb) - \cos(za)}{z^2}.$$
    The trigonometric functions are evaluated once per point and shared by the neighbouring segments.
    For segments with $z (b-a) <$ :py:data:`Z_DXI_SERIES`, the second terms are computed from their series
    $s \frac{\sin(zb) - \sin(za)}{z^2} \approx (g_b - g_a) \frac{\cos(za) + \cos(zb)}{2z}$
    (and correspondingly for the cosine), since the difference of the trigonometric functions would cancel.
    The error of this is $O((z(b-a))^2)$ relative to these terms.

    :param z: $z$ values, which should be large enough for the $1/z^2$ terms not to cancel, e.g. $z > 1$
    :param xi: $\xi$ points in ascending order, possibly with duplicates at the discontinuities
    :param v: fluid velocity $v$ at the points
    :param lam: energy fluctuation $\lambda$ at the points
    :return: $S_{v}$, $C_{v}$ and $S_{\lambda}$ for each $z$
    """
    xv = xi * v
    xl = xi * lam
    s_v = np.empty(z.size)
    c_v = np.empty(z.size)
    s_l = np.empty(z.size)
    for k in numba.prange(z.size):
        zk = z[k]
        inv_z = 1. / zk
        inv_z2 = inv_z * inv_z
        sum_s_v = 0.
        sum_c_v = 0.
        sum_s_l = 0.
        c_a = np.cos(zk * xi[0])
        s_a = np.sin(zk * xi[0])
        for i in range(xi.size - 1):
            c_b = np.cos(zk * xi[i+1])
            s_b = np.sin(zk * xi[i+1])
            d = xi[i+1] - xi[i]
            if d > 0:
                if zk * d < Z_DXI_SERIES:
                    # (sin zb - sin za) / (b - a) ~ z (cos za + cos zb) / 2, and correspondingly for cos
                    ds = 0.5 * zk * (c_a + c_b)
                    dc = -0.5 * zk * (s_a + s_b)
                else:
                    ds = (s_b - s_a) / d
                    dc = (c_b - c_a) / d
                sum_s_v += -(v[i+1] * c_b - v[i] * c_a) * inv_z + (v[i+1] - v[i]) * ds * inv_z2
                sum_c_v += (xv[i+1] * s_b - xv[i] * s_a) * inv_z + (xv[i+1] - xv[i]) * dc * inv_z2
                sum_s_l += -(xl[i+1] * c_b - xl[i] * c_a) * inv_z + (xl[i+1] - xl[i]) * ds * inv_z2
            c_a = c_b
            s_a = s_b
        s_v[k] = sum_s_v
        c_v[k] = sum_c_v
        s_l[k] = sum_s_l
    return s_v, c_v, s_l


sound_shell_transforms_pl_parallel: tp.Callable[..., tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]]
sound_shell_transforms_pl_single: tp.Callable[..., tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]]
sound_shell_transforms_pl_parallel, sound_shell_transforms_pl_single = njit_parallel_pair(
    _sound_shell_transforms_pl, nogil=True, cache=True)


@njit(cache=True, nogil=True)
def sharpen_padding(xi: th.FloatArr1D, v: th.FloatArr1D) -> th.FloatArr1D:
    r"""Move the padding points of a fluid shell onto the profile, so that the jumps to still fluid are sharp.

    :py:func:`pttools.bubble.fluid.sound_shell_generic` pads the profile with still fluid,
    with points at $\Delta\xi = 1/n_{\xi}$ behind the first and ahead of the last point of the profile.
    The linear interpolant therefore has ramps of width $\Delta\xi$ at the shock (or the wall of a detonation),
    and at the wall of a subsonic deflagration, which would dampen the exact transforms at $z \gtrsim 1/\Delta\xi$.
    This returns a copy of $\xi$, in which these padding points have the $\xi$ of their neighbouring profile points,
    making the jumps discontinuities.

    :param xi: $\xi$ of the fluid shell
    :param v: fluid velocity $v$ of the fluid shell
    :return: $\xi$ with the padding points moved
    """
    xi_sharp = xi.copy()
    # The padding points and the profile points next to them
    if xi.size >= 4:  # noqa: PLR2004
        if v[-1] == 0 and v[-2] == 0 and v[-3] != 0:
            xi_sharp[-2] = xi[-3]
        if v[0] == 0 and v[1] == 0 and v[2] != 0:
            xi_sharp[1] = xi[2]
    return xi_sharp
