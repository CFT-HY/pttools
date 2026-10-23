"""Sound Shell Model functions."""

import enum
import logging

import numpy as np

from pttools import speedup
from pttools.bubble.thermo import w_bar as w_bar_func
from pttools.speedup import njit
from pttools.ssm import NucType, const, lifetime_distribution_momentum
from pttools.ssm.nucleation import beta_R_star0, lifetime_distribution
from pttools.ssm.resample import resample_merged_uniform_xi
from pttools.ssm.sin_transform import sin_transform
from pttools.ssm.sin_transform_exact import (
    sharpen_padding,
    sound_shell_transforms_pl_parallel,
    sound_shell_transforms_pl_single,
)
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


@enum.unique
class DE_Method(enum.StrEnum):
    r"""Method for computing $\lvert A(z) \rvert^2$."""

    # TODO: Improve this docstring
    ALTERNATE = "alternate"
    STANDARD = "standard"


@enum.unique
class LambdaCorrection(enum.StrEnum):
    r"""Kinetic energy term of the energy fluctuation variable $\lambda$, see :py:func:`lam`.

    The energy density of the fluid is $T^{00} = e + w \gamma^2 v^2$.
    For the self-similar flow of a single bubble, energy conservation
    $\partial_t T^{00} + \nabla_i T^{0i} = 0$ gives
    $$\int_0^1 d\xi \xi^2 \left( T^{00}(\xi) - e_n \right) = 0,$$
    as $T^{0i} = w \gamma^2 v^i$ vanishes outside the shock.
    The total energy fluctuation of the fluid shell is therefore zero only if $\lambda$ includes the kinetic term.
    Otherwise $\int dV \lambda \neq 0$, and $\lvert A(z) \rvert^2$ does not vanish as $z \to 0$,
    which adds spurious power to the velocity and GW power spectra at low $z$
    and shifts their peaks to lower $z$.
    The difference is of the order of the kinetic energy fraction,
    and is therefore significant for strong phase transitions.
    """

    #: $\lambda = (T^{00} - \bar{e}) / \bar{w}$ with $T^{00} = e + w \gamma^2 v^2$,
    #: which conserves the energy of the fluid shell.
    EXACT = "exact"
    #: $\lambda = (e + w v^2 - \bar{e}) / \bar{w}$, the non-relativistic kinetic term of the versions of PTtools
    #: before 2026-04, e.g. those used for :correia_2025:`\ ` and for computing the suppression datasets of
    #: :gowling_2021:`\ `. It does not fully conserve energy when $\gamma^2 \gg 1$ in the fluid shell.
    LEGACY = "legacy"
    #: $\lambda = (e - \bar{e}) / \bar{w}$, the linearised energy density fluctuation of :gw_pt_ssm:`\ ` eq. 3.20,
    #: which does not include the kinetic energy and therefore does not conserve energy.
    NONE = "none"


#: Default kinetic energy term of $\lambda$
DEFAULT_LAMBDA_CORRECTION: LambdaCorrection = LambdaCorrection.EXACT


@enum.unique
class Method(enum.StrEnum):
    r"""Method for computing $\lvert A(z) \rvert^2$."""

    # TODO: Improve this docstring
    E_CONSERVING = "e_conserving"
    F_ONLY = "f_only"
    WITH_G = "with_g"


@njit(nogil=True)
def A2_e_conserving(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        e: th.FloatArr1D,
        z: th.FloatArr1D,
        v_wall: float,
        v_sh: float,
        cs: float,
        z_st_thresh: float = const.Z_ST_THRESH,
        n_xi: int = const.DEFAULT_N_XI_SSM,
        parallel: bool = True,
        lambda_correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION,
        exact_large_z: bool = True,
        w_bar: float | None = None) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""
    Returns the value of $\lvert A(z) \rvert^2$.

    Here $\lvert \text{Plane wave amplitude} \rvert^2 = T^3 \lvert A(z) \rvert^2$.

    :param v: velocity profile $v$
    :param w: enthalpy profile $w$
    :param xi: $\xi$ of the profiles
    :param e: energy density profile $e$
    :param v_wall: wall speed $v_\text{wall}$
    :param v_sh: shock speed $v_\text{sh}$
    :param cs: speed of sound $c_s$
    :param z: array of scaled wavenumbers $z = kR_*$.
    :param z_st_thresh: limit above which to use approximate sin transform
    :param n_xi: Number of $\xi$ points for uniform resampling
    :param parallel: Whether to use multiple threads
    :param lambda_correction: kinetic energy term of $\lambda$, see :py:class:`LambdaCorrection`
    :param exact_large_z: for $z >$ z_st_thresh, use the exact transforms of the piecewise-linear profiles
        (:py:func:`pttools.ssm.sin_transform_exact.sound_shell_transforms_pl_parallel`)
        and the analytical $f'(z)$ instead of the approximate sine transform and finite differences
    :param w_bar: $\bar{w}$, average enthalpy density for $\lambda$, see :py:func:`lam`.
        If not given, it's computed with :py:func:`pttools.bubble.thermo.w_bar`.
    :return: $\lvert A(z) \rvert^2, \frac{1}{2} f'(z)^2, \frac{1}{2}(c_s l(z))^2$ (same size as $z$)
    """
    #: $\lambda(z)$
    w_ref = w_bar_func(w=w, xi=xi, v_wall=v_wall) if w_bar is None else w_bar
    lm = lam(v=v, w=w, e=e, w_bar=w_ref, correction=lambda_correction)
    _f_val, fp_val, l_val = f_fp_l(
        z=z, xi=xi, v=v, lam=lm, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, n_xi=n_xi,
        parallel=parallel, exact_large_z=exact_large_z)
    return A2_fp_csl(fp=fp_val, cs=cs, l=l_val), 0.5 * fp_val ** 2, 0.5 * (cs * l_val) ** 2


@njit(cache=True)
def A2_fp_csl(fp: th.FloatArr, cs: float, l: th.FloatArr) -> th.FloatArr:  # noqa: E741
    r"""$\lvert A(z) \rvert^2$ from $f'(z)$ and $c_s l(z)$.

    $$\lvert A(z) \rvert^2 = = \frac{1}{4} \left[ (f'(z))^2 + (c_s l(z))^2 \right]$$
    :gw_pt_ssm:`\ ` eq. 4.11
    This contains information about the shape of the fluid shells.
    """
    return 0.25 * (fp ** 2 + (cs * l) ** 2)


@njit
def f(
        z: th.FloatArr,
        xi: th.FloatArr,
        v: th.FloatArr,
        v_wall: float,
        v_sh: float | None,
        z_st_thresh: float = const.Z_ST_THRESH,
        parallel: bool = True) -> th.FloatArr:
    r"""$f(z)$, sine transform of the velocity profile.

    $$f(z) = \frac{4\pi}{z} \int_0^\infty d\xi v_{\text{ip}}(\xi) \sin(z\xi)$$
    :gw_pt_ssm:`\ ` eq. 4.5.
    """
    return 4. * np.pi / z * sin_transform(
        z=z, xi=xi, f=v, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh, parallel=parallel
    )


@njit(nogil=True)
def f_fp_l(
        z: th.FloatArr1D,
        xi: th.FloatArr1D,
        v: th.FloatArr1D,
        lam: th.FloatArr1D,
        v_wall: float,
        v_sh: float | None,
        z_st_thresh: float = const.Z_ST_THRESH,
        n_xi: int = const.DEFAULT_N_XI_SSM,
        parallel: bool = True,
        exact_large_z: bool = True) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""$f(z)$, $f'(z)$ and $l(z)$, the sine transforms of the velocity and energy fluctuation profiles.

    For $z \leq$ z_st_thresh, $f(z)$ and $l(z)$ are computed with :py:func:`f` and :py:func:`l`,
    i.e. with the trapezoidal rule, and $f'(z)$ with finite differences.
    For larger $z$, if exact_large_z is True, they are computed with the exact transforms of the
    piecewise-linear profiles,
    :py:func:`pttools.ssm.sin_transform_exact.sound_shell_transforms_pl_parallel`,
    after making the jumps to still fluid sharp with :py:func:`pttools.ssm.sin_transform_exact.sharpen_padding`,
    and $f'(z)$ is computed analytically as
    $$f'(z) = -\frac{f(z)}{z} + \frac{4\pi}{z} \int d\xi\, v(\xi) \xi \cos(z\xi),$$
    since the spacing of the $z$ arrays can exceed the period of the oscillations of $f(z)$ at large $z$.
    Otherwise, the approximate sine transform of :py:func:`pttools.ssm.sin_transform_approx.sin_transform_approx`
    and finite differences are used for all $z$.

    :param z: $z = kR_*$, in ascending order
    :param xi: $\xi$ of the profiles
    :param v: velocity profile $v$
    :param lam: energy fluctuation profile $\lambda$
    :param v_wall: wall speed $v_\text{wall}$
    :param v_sh: shock speed $v_\text{sh}$
    :param z_st_thresh: threshold of $z$ for the trapezoidal rule
    :param n_xi: number of $\xi$ points for the resampling of $\lambda$ for the trapezoidal rule
    :param parallel: whether to use multiple threads
    :param exact_large_z: whether to use the exact transforms for large $z$
    :return: $f(z)$, $f'(z)$ and $l(z)$
    """
    if not exact_large_z:
        f_val = f(z=z, xi=xi, v=v, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, parallel=parallel)
        fp_val = speedup.gradient(f_val) / speedup.gradient(z)
        l_val = l(
            z=z, xi=xi, lam=lam, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, n_xi=n_xi, parallel=parallel)
        return f_val, fp_val, l_val

    lo = z <= z_st_thresh
    hi = ~lo
    z_lo = z[lo]
    z_hi = z[hi]
    f_val = np.empty_like(z)
    fp_val = np.empty_like(z)
    l_val = np.empty_like(z)
    if z_lo.size:
        # Below the threshold the trapezoidal rule is accurate, and the approximation is not used.
        f_val[lo] = f(z=z_lo, xi=xi, v=v, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, parallel=parallel)
        l_val[lo] = l(
            z=z_lo, xi=xi, lam=lam, v_wall=v_wall, v_sh=v_sh, z_st_thresh=z_st_thresh, n_xi=n_xi,
            parallel=parallel)
    if z_hi.size:
        xi_sharp = sharpen_padding(xi, v)
        if parallel:
            s_v, c_v, s_l = sound_shell_transforms_pl_parallel(z_hi, xi_sharp, v, lam)
        else:
            s_v, c_v, s_l = sound_shell_transforms_pl_single(z_hi, xi_sharp, v, lam)
        f_hi = 4. * np.pi / z_hi * s_v
        f_val[hi] = f_hi
        fp_val[hi] = -f_hi / z_hi + 4. * np.pi / z_hi * c_v
        l_val[hi] = 4. * np.pi / z_hi * s_l
    if z_lo.size:
        # At small z, the analytical f'(z) would suffer from cancellation, so finite differences are used.
        fp_all = speedup.gradient(f_val) / speedup.gradient(z)
        fp_val[lo] = fp_all[lo]
    return f_val, fp_val, l_val


@njit(nogil=True)
def l(  # noqa: E743
        z: th.FloatArr,
        xi: th.FloatArr,
        lam: th.FloatArr,
        v_wall: float,
        v_sh: float | None,
        z_st_thresh: float = const.Z_ST_THRESH,
        n_xi: int = const.DEFAULT_N_XI_SSM,
        parallel: bool = True) -> th.FloatArr:
    r"""$l(z)$, sine transform of the energy fluctuation profile $\lambda$.

    $$l(z) = \frac{4\pi}{z} \int_0^\infty d\xi \lambda_\text{ip}(\xi) \xi \sin(z\xi)$$
    :gw_pt_ssm:`\ ` eq. 4.8.
    """
    xi_re, lam_re = resample_merged_uniform_xi(xi, lam, n_xi)

    # Some old implementation
    # lam_re = np.interp(xi_re,xi,lam_orig)
    # lam_ft = np.zeros_like(z)
    # for j in range(lam_ft.size):
    #     # Need to fix problem with ST of lam for detonations
    #     lam_ft[j] = (4.*np.pi/z[j]) * \
    #         calculators.sin_transform(z[j], xi_re, xi_re*lam_re, z_st_thresh=max(z))

    return 4. * np.pi / z * sin_transform(
        z=z, xi=xi_re, f=xi_re * lam_re, z_st_thresh=z_st_thresh, v_wall=v_wall, v_sh=v_sh, parallel=parallel
    )


@njit(cache=True)
def lam[S: th.Shape](
        v: th.FloatArrShaped[S],
        w: th.FloatArrShaped[S],
        e: th.FloatArrShaped[S],
        e_bar: float | None = None,
        w_bar: float | None = None,
        correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION) -> th.FloatArrShaped[S]:
    r"""$\lambda(x)$, energy fluctuation variable.

    $$\lambda(x) = \frac{T^{00}(x) - \bar{e}}{\bar{w}}, \quad T^{00} = e + w \gamma^2 v^2,$$
    which is the energy fluctuation variable of :gw_pt_ssm:`\ ` eq. 3.20 with the kinetic energy included,
    so that the energy of the fluid shell is conserved, see :py:class:`LambdaCorrection`.
    The kinetic term is given by :py:func:`lam_kinetic`.

    :param v: $v$
    :param w: $w$, of the same shape as $v$
    :param e: $e$, of the same shape as $v$
    :param e_bar: Mean energy density $\bar{e}$. If not given, assumed to be $\bar{e} = e_n$.
    :param w_bar: Mean enthalpy density $\bar{w}$. If not given, assumed to be $\bar{w} = w_n$.
    :param correction: kinetic energy term, see :py:class:`LambdaCorrection`
    """
    e_ref = e[-1] if e_bar is None else e_bar
    w_ref = w[-1] if w_bar is None else w_bar
    # A similar step is done in de_from_w_bag()
    return (e - e_ref) / w_ref + lam_kinetic(v=v, w=w, w_bar=w_ref, correction=correction)  # pyrefly: ignore[bad-return]


@njit(cache=True)
def lam_kinetic[S: th.Shape](
        v: th.FloatArrShaped[S],
        w: th.FloatArrShaped[S],
        w_bar: float,
        correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION) -> th.FloatArrShaped[S]:
    r"""Kinetic energy term of the energy fluctuation variable $\lambda$.

    $$\frac{w \gamma^2 v^2}{\bar{w}}$$ for :py:attr:`LambdaCorrection.EXACT`,
    $$\frac{w v^2}{\bar{w}}$$ for :py:attr:`LambdaCorrection.LEGACY`,
    and 0 for :py:attr:`LambdaCorrection.NONE`.

    :param v: $v$
    :param w: $w$, of the same shape as $v$
    :param w_bar: Mean enthalpy density $\bar{w}$
    :param correction: kinetic energy term, see :py:class:`LambdaCorrection`
    :return: the kinetic energy term, of the same shape as $v$
    """
    if correction == LambdaCorrection.EXACT.value:
        return w * v * v / ((1 - v * v) * w_bar)  # pyrefly: ignore[bad-return]
    if correction == LambdaCorrection.LEGACY.value:
        return w * v * v / w_bar  # pyrefly: ignore[bad-return]
    if correction == LambdaCorrection.NONE.value:
        return np.zeros_like(w)  # pyrefly: ignore[bad-return]
    raise ValueError("Unknown lambda correction")


@njit(cache=True)
def qT_lookup(T_tilde: th.FloatArr1D, z: th.FloatArr1D) -> th.FloatArr1D:
    """$z=qT$ lookup array.

    This is used by
    :py:func:`pttools.ssm.spec_den_v.spec_den_v` and
    :py:func:`pttools.ssm.ssm.A2_e_conserving`.
    """
    # z limits
    log10_z_min = np.log10(np.min(z))
    log10_z_max = np.log10(np.max(z))

    # try:
    # Todo: Check whether this could be replaced by logspace
    return 10 ** np.arange(
        log10_z_min + np.log10(T_tilde.min()),
        log10_z_max + np.log10(T_tilde.max()),
        step=(log10_z_max - log10_z_min) / z.size
    )
    # except ValueError as e:
    #     logger.error(
    #         "Could not compute qT_lookup with log10_z_min=%s, log10_T_min=%s, "
    #         "log10_z_max=%s, log10_T_max=%s, dlog10z=%s",
    #         log10_z_min, log10_T_min, log10_z_max, log10_T_max, dlog10z
    #     )
    #     raise e


@njit
def T_tilde(T_tilde_min: float, T_tilde_max: float, n: int) -> th.FloatArr1D:
    r"""Generate $\tilde{T}$ array."""
    return speedup.logspace(np.log10(T_tilde_min), np.log10(T_tilde_max), n)


@njit(nogil=True)
def ubarf2_from_a2(
        T_tilde: th.FloatArr1D,
        z: th.FloatArr1D,
        A2: th.FloatArr1D,
        v_wall: float,
        nuc_type: NucType) -> float:
    r"""Mean square fluid velocity $\bar{U}_f^2 \left( {\lvert A(z) \rvert}^2 \right)$.

    $$\bar{U}_f^2
    = \int \frac{dq}{q} \mathcal{P}_{\tilde{v}}(a)
    = \frac{2}{(\beta R_*)^3}
    \int d\tilde{T} \nu(\tilde{T}) \tilde{T}^3
    \int dz \frac{z^2}{2\pi^2} \lvert A(z) \rvert^2$$
    :gw_pt_ssm:`\ ` eq. 4.33
    This version takes into account the nucleation history, unlike
    :py:func:`pttools.bubble.thermo.ubarf2`.

    Here $\beta \equiv (8\pi)^\frac{1}{3} \frac{{v}_\text{wall}}{R_{\ast}}$ is the rate that makes the
    lifetime distribution $\nu$ dimensionless (:gw_pt_ssm:`\ ` p. 17), and $R_{\ast}$ is the actual mean bubble spacing.
    Therefore $\beta R_{\ast} = (8\pi)^\frac{1}{3} {v}_\text{wall}$ also with the thermal suppression
    of bubble nucleation of :ajmi_2022:`\ `,
    where $R_{\ast} = \Lambda R_{\ast,0}$ and $\beta = \beta_\text{nuc} / \Lambda$.
    As the bubbles fill space, $n_b \langle \frac{4\pi}{3} {v}_\text{wall}^3 T^3 \rangle = 1$,
    which gives $\nu_3 = 6$ for any nucleation history, including the suppressed one.
    Therefore $\bar{U}_f^2$ does not depend on the bubble spacing enlargement factor $\Lambda$.
    """
    if z.shape != A2.shape:
        raise TypeError(
            "z and A2 must be of the same shape. "
            f"Got z.shape={z.shape}, A2.shape={A2.shape}"
        )

    nu3 = 6. \
        if nuc_type in (NucType.EXPONENTIAL, NucType.SIMULTANEOUS) \
        else lifetime_distribution_momentum(nu=lifetime_distribution(T_tilde, nuc_type), T_tilde=T_tilde, n=3)
    beta_R = beta_R_star0(v_wall)
    return 2 / (beta_R**3 * 2 * np.pi**2) * nu3 * np.trapezoid(z ** 2 * A2, z)  # pyrefly: ignore[bad-return]
