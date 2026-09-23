r"""Chapman-Jouguet speed $v_{CJ}$.

The Chapman-Jouguet speed is the wall speed of a detonation, for which the fluid speed behind the wall
in the wall frame equals the speed of sound, $\tilde{v}_- = c_{s,-}({w}_-)$, :maki_msc:`\ ` eq. 2.97.
A detonation with $v_\text{wall} < v_{CJ}$ would be a strong detonation, which is unstable.
Therefore, $v_{CJ}$ is the minimum wall speed for detonations,
and the boundary between detonations and hybrids.

The junction conditions of :func:`pttools.bubble.v_plus.v_plus` depend on $\alpha_+$, which depends on ${w}_-$,
and therefore in general $v_{CJ}$ has to be solved numerically.
For the bag model and the constant sound speed model, $v_{CJ}$ has an analytical form.
"""

from collections.abc import Iterable
import itertools
import logging
import typing as tp

import numpy as np
from scipy.optimize import brentq, fsolve

from pttools.bubble.const import MU_BAG
from pttools.bubble.junction import w2_junction
from pttools.bubble.phase import Phase
from pttools.bubble.relativity import gamma2
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.v_plus import v_plus
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils.math import finite_edge

if tp.TYPE_CHECKING:
    from pttools.models.const_cs import ConstCSModel
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


# def gen_wn_solvable(model: "Model", alpha_n: float):
#     def wn_solvable(params: th.FloatArr1D) -> float:
#         r"""This function is zero when $w_n$ corresponds to the given $\alpha_n$"""
#         wn = params[0]
#         # return model.theta(wn, Phase.SYMMETRIC) - model.theta(wn, Phase.BROKEN) - 3/4 * wn * alpha_n
#         return model.alpha_n(wn) - alpha_n
#     return wn_solvable


# def chapman_jouguet_solvable(params: th.FloatArr1D, model: "Model", wn: float, wm_guess: float):
#     v_wall = params[0]
#     vm_guess = np.sqrt(model.cs2(wm_guess, Phase.BROKEN))
#     _, _, vm, wm = solve_boundary(
#         v_wall, wn, SolutionType.SUB_DEF, model, vm_guess=vm_guess, wm_guess=wm_guess)
#     return vm - np.sqrt(model.cs2(wm, Phase.BROKEN))
#
#
# def chapman_jouguet_vm_solvable(params: th.FloatArr1D, model: "Model", vp: float, wp: float):
#     """Not useful, as we don't know vp."""
#     vm = params[0]
#     wm = wp * gamma2(vp) * vp / (gamma2(vm) * vm)
#     cs = np.sqrt(model.cs2(wm, Phase.BROKEN))
#     return cs - vm


# def wm_vw_solvable(params: th.FloatArr1D, model: "Model", vp: float, wp: float):
#     r"""$$\Delta_\text{junc1}(w_-)$$ for detonations"""
#     wm = params[0]
#     vm = v_minus(vp, model.alpha_plus(wp, wm), SolutionType.DETON)
#     return junction_condition_deviation1(vp, wp, vm, wm)
#
#
# def wm_vw(wm_guess: float, model: "Model", vp: float, wp: float):
#     """$$w_-(v_w)$$"""
#     sol = fsolve(wm_vw_solvable, x0=np.array([wm_guess]), args=(model, vp, wp), full_output=True)
#     wm = sol[0][0]
#     if sol[2] != 1:
#         logger.error(
#             f"wm(vw) solution was not found for model={model.name}, vp={vp}, wp={wp}, wm_guess={wm_guess}. "
#             f"Using wm(vw)={wm}. "
#             f"Reason: {sol[3]}"
#         )
#     return wm
#
#
# def v_chapman_jouguet_solvable(params: th.FloatArr1D, model: "Model", wp: float, wm_guess: float = None):
#     vp = params[0]
#     # If a guess is not provided, use the bag model value.
#     wm_guess = w2_junction(vp, wp, const.CS0) if wm_guess is None else wm_guess
#     wm = wm_vw(wm_guess, model, vp, wp)
#     vm = v_minus(vp, model.alpha_plus(wp, wm))
#     cs = model.cs2(wm, Phase.BROKEN)
#     return vm - cs


# def v_chapman_jouguet_new(
#         model: "Model",
#         alpha_n: float,
#         wn: float = None,
#         wn_guess: float = None,
#         wm_guess: float = None,
#         extra_output: bool = False,
#         analytical: bool = True) -> tp.Union[float, tuple[float, float, float]]:
#     if analytical and model.DEFAULT_NAME == "bag":
#         return v_chapman_jouguet_bag(alpha_plus=alpha_n)
#
#     if wn is None:
#         wn = model.w_n(alpha_n, wn_guess=wn_guess)
#     v_cj_guess = v_chapman_jouguet_bag(alpha_plus=alpha_n)
#     sol = fsolve(
#         v_chapman_jouguet_solvable,
#         x0=np.array([v_cj_guess]),
#         args=(model, wn),
#         full_output=True
#     )
#     v_cj = sol[0][0]
#     if sol[2] != 1:
#         logger.error(
#             f"v_cj solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
#             f"Using v_cj={v_cj}. "
#             f"Reason: {sol[3]}"
#         )
#     return v_cj


# def v_chapman_jouguet_old2(
#         model: "Model",
#         alpha_n: float,
#         wn_guess: float = 1,
#         wm_guess: float = 1,
#         extra_output: bool = False,
#         analytical: bool = True) -> tp.Union[float, tuple[float, float, float]]:
#     if analytical and model.DEFAULT_NAME == "bag":
#         return v_chapman_jouguet_bag(alpha_plus=alpha_n)
#
#     wn = model.w_n(alpha_n, wn_guess=wn_guess)
#     vm_guess = model.cs2(wm_guess, Phase.BROKEN)
#     vm = fsolve(chapman_jouguet_vm_solvable, x0=np.array([vm_guess]), args=(model, vp, wn))


# def v_chapman_jouguet_old2(
#         model: "Model",
#         alpha_n: float,
#         wn_guess: float = 1,
#         wm_guess: float = 1,
#         extra_output: bool = False,
#         analytical: bool = True) -> tp.Union[float, tuple[float, float, float]]:
#     if analytical and model.DEFAULT_NAME == "bag":
#         return v_chapman_jouguet_bag(alpha_plus=alpha_n)
#
#     v_cj_guess = 0.5
#     # v_cj_guess = v_chapman_jouguet_old(model, alpha_n)
#     # return v_cj_guess
#
#     wn = model.w_n(alpha_n, wn_guess=wn_guess)
#     sol = fsolve(chapman_jouguet_solvable, x0=np.array([v_cj_guess]), args=(model, wn, wm_guess), full_output=True)
#     v_cj = sol[0][0]
#     if sol[2] != 1:
#         logger.error(
#             f"v_cj solution was not found for alpha_n={alpha_n}, model={model.name}, wn_guess={wn_guess}. "
#             f"Using v_cj={v_cj}. "
#             f"Reason: {sol[3]}"
#         )
#     return v_cj


def _handle_failure(msg: str, error_on_invalid: bool, log_invalid: bool) -> None:
    """Log and/or raise an error for a failed step of :func:`v_chapman_jouguet`."""
    if log_invalid:
        logger.error(msg)
    if error_on_invalid:
        raise RuntimeError(msg)


def v_chapman_jouguet_analytical[T: FloatOrArr](
        model: "Model",
        alpha_n: T,
        wn: float | None = None,
        wn_guess: float | None = None) -> T:
    r"""$v_{CJ}$, Chapman-Jouguet speed for the bag and constant sound speed models.

    For the bag model this is :func:`v_chapman_jouguet_bag` with $\alpha_+ = \alpha_n$,
    and for the constant sound speed model :func:`v_chapman_jouguet_const_cs`
    with $\alpha_{\bar{\Theta}_+} = \alpha_{\bar{\Theta}_n}$, which hold for all detonations.

    :param model: bag or constant sound speed model
    :param alpha_n: $\alpha_n$, transition strength at the nucleation temperature
    :param wn: ${w}_n$, enthalpy at the nucleation temperature. Computed from $\alpha_n$ if not given.
    :param wn_guess: starting guess for ${w}_n$
    :return: $v_{CJ}$, Chapman-Jouguet speed
    :raises ValueError: if the model is not supported
    """
    if model.DEFAULT_NAME == "bag":
        return v_chapman_jouguet_bag(alpha_plus=alpha_n)
    if model.DEFAULT_NAME == "const_cs":
        alpha_theta_bar_plus = model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=wn, wn_guess=wn_guess)
        # The model name check above ensures that the model is a ConstCSModel.
        return v_chapman_jouguet_const_cs(tp.cast("ConstCSModel", model), alpha_theta_bar_plus=alpha_theta_bar_plus)
    raise ValueError(f"No analytical Chapman-Jouguet speed is available for the model: {model.DEFAULT_NAME}")


def _extra_output_analytical[T: FloatOrArr](
        model: "Model",
        alpha_n: T,
        v_cj: T,
        wn: float | None,
        wn_guess: float | None) -> tuple[T, T, T]:
    r"""$\tilde{v}_-$ and $\alpha_+$ of a Chapman-Jouguet detonation for the analytical models.

    The sound speed of these models is constant within each phase,
    and therefore $\tilde{v}_- = c_{s,b}$ does not depend on ${w}_-$.
    Then ${w}_-$ is given by the junction condition of :func:`pttools.bubble.junction.w2_junction`.
    """
    wn_solved = tp.cast(T, model.wn(alpha_n, wn_guess=wn_guess) if wn is None else wn)
    vm_cj = tp.cast(T, np.sqrt(model.cs2(wn_solved, Phase.BROKEN)))
    wm = w2_junction(v_cj, wn_solved, vm_cj)
    # alpha_plus can be negative for Chapman-Jouguet detonations if c_{s,b}^2 < 1/3,
    # and therefore it is not validated.
    ap_cj = model.alpha_plus(wn_solved, wm, error_on_invalid=False, nan_on_invalid=False, log_invalid=False)
    return v_cj, vm_cj, ap_cj


@tp.overload
def v_chapman_jouguet(
        model: "Model",
        alpha_n: float,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None,
        extra_output: tp.Literal[False] = False,
        analytical: bool = True,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> float: ...
@tp.overload
def v_chapman_jouguet(
        model: "Model",
        alpha_n: float,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None,
        *,
        extra_output: tp.Literal[True],
        analytical: bool = True,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> tuple[float, float, float]: ...
@tp.overload
def v_chapman_jouguet(
        model: "Model",
        alpha_n: th.FloatArr,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None,
        extra_output: tp.Literal[False] = False,
        analytical: bool = True,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> th.FloatArr1D: ...
@tp.overload
def v_chapman_jouguet(
        model: "Model",
        alpha_n: th.FloatArr,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None,
        *,
        extra_output: tp.Literal[True],
        analytical: bool = True,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]: ...
def v_chapman_jouguet(
        model: "Model",
        alpha_n: th.FloatOrArr,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None,
        extra_output: bool = False,
        analytical: bool = True,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> th.FloatOrArr | tuple[th.FloatOrArr, th.FloatOrArr, th.FloatOrArr]:
    r"""$v_{CJ}$, Chapman-Jouguet speed.

    The Chapman-Jouguet speed is the wall speed of a detonation, for which
    $$\tilde{v}_- = c_{s,-}({w}_-),$$
    :maki_msc:`\ ` eq. 2.97.
    This is the minimum wall speed for stable detonations.

    For the bag and constant sound speed models the analytical solution of
    :func:`v_chapman_jouguet_analytical` is used by default.
    For the other models, or if ``analytical=False``, the Chapman-Jouguet speed is computed numerically.
    For detonations ${w}_+ = {w}_n$, and ${w}_-$ is solved with :func:`wm_chapman_jouguet`.
    Then $\tilde{v}_- = c_{s,-}({w}_-)$ and $\alpha_+({w}_+, {w}_-)$ are inserted to
    :func:`pttools.bubble.v_plus.v_plus` to get $v_{CJ} = \tilde{v}_+$.

    :param model: the equation of state
    :param alpha_n: $\alpha_n$, transition strength at the nucleation temperature
    :param wn: ${w}_n$, enthalpy at the nucleation temperature. Computed from $\alpha_n$ if not given.
    :param wn_guess: starting guess for ${w}_n$
    :param wm_guess: starting guess for ${w}_-$
    :param extra_output: whether to also return $\tilde{v}_-$ and $\alpha_+$
    :param analytical: whether to use the analytical solution when it is available
    :param error_on_invalid: whether to raise an error if the solution cannot be found
    :param nan_on_invalid: whether to return nan if the solution cannot be found
    :param log_invalid: whether to log the failures
    :return: $v_{CJ}$, or $(v_{CJ}, \tilde{v}_-, \alpha_+)$ if ``extra_output`` is True
    :raises RuntimeError: if the solution cannot be found and ``error_on_invalid`` is True
    """
    if analytical and model.DEFAULT_NAME in ("bag", "const_cs"):
        v_cj = v_chapman_jouguet_analytical(model, alpha_n, wn=wn, wn_guess=wn_guess)
        if extra_output:
            return _extra_output_analytical(model, alpha_n, v_cj, wn=wn, wn_guess=wn_guess)
        return v_cj

    if isinstance(alpha_n, Iterable):
        rets = np.array([v_chapman_jouguet(
            model, a_n,
            wn=wn, wn_guess=wn_guess, wm_guess=wm_guess, extra_output=extra_output, analytical=analytical,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        ) for a_n in alpha_n])
        if extra_output:
            # Each row is (v_cj, vm_cj, ap_cj) for one alpha_n.
            # Transposing gives a tuple of arrays, as in the analytical case.
            return tp.cast(tuple[th.FloatArr, th.FloatArr, th.FloatArr], tuple(rets.T))
        return rets

    return _v_chapman_jouguet_numerical(
        model, alpha_n,
        wn=wn, wn_guess=wn_guess, wm_guess=wm_guess, extra_output=extra_output,
        error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
    )


def _v_chapman_jouguet_numerical(
        model: "Model",
        alpha_n: float,
        wn: float | None,
        wn_guess: float | None,
        wm_guess: float | None,
        extra_output: bool,
        error_on_invalid: bool,
        nan_on_invalid: bool,
        log_invalid: bool) -> float | tuple[float, float, float]:
    r"""Numerical Chapman-Jouguet speed for a single $\alpha_n$. See :func:`v_chapman_jouguet`."""
    nan_output = (np.nan, np.nan, np.nan) if extra_output else np.nan
    if wn is None:
        wn = model.wn(
            alpha_n, wn_guess=wn_guess,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )
    if wn is None or np.isnan(wn):
        _handle_failure(
            f"Failed to find wn for alpha_n={alpha_n}",
            error_on_invalid=error_on_invalid, log_invalid=log_invalid
        )
        return nan_output

    # For detonations wp = wn
    wm = wm_chapman_jouguet(
        model, wp=wn, wm_guess=wm_guess,
        error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
    )
    if wm is None or np.isnan(wm):
        _handle_failure(
            f"Failed to find wm for alpha_n={alpha_n}, wn={wn}",
            error_on_invalid=error_on_invalid, log_invalid=log_invalid
        )
        return nan_output

    vm_cj = np.sqrt(model.cs2(wm, Phase.BROKEN))
    # alpha_plus can be negative for Chapman-Jouguet detonations if c_{s,-}^2 < 1/3,
    # and therefore it is not validated.
    ap_cj = model.alpha_plus(wn, wm, error_on_invalid=False, nan_on_invalid=False, log_invalid=False)
    v_cj = v_plus(vm_cj, ap_cj, sol_type=SolutionType.DETON)
    if np.isnan(v_cj):
        _handle_failure(
            f"Failed to find v_CJ for wn={wn}, wm={wm}, alpha_plus={ap_cj}",
            error_on_invalid=error_on_invalid, log_invalid=log_invalid
        )
    if extra_output:
        return v_cj, vm_cj, ap_cj
    return v_cj


@njit(cache=True)
def v_chapman_jouguet_bag[T: FloatOrArr](alpha_plus: T) -> T:
    r"""$v_{CJ}$, Chapman-Jouguet speed for the bag model.

    $\alpha_n$ can be given instead of $\alpha_+$, as
    "The two definitions of the transition strength coincide
    only in the case of detonations within the bag model."
    :notes:`\ ` p. 40

    $$v_{CJ}(\alpha_+) = \frac{1}{\sqrt{3}} \frac{1 + \sqrt{2\alpha_+ + 3 \alpha_+^2}}{1 + \alpha_+}$$
    The sources
    :notes:`\ ` eq. 7.34,
    :gw_pt_ssm:`\ ` eq. B.19 (and B.21) and
    :giombi_2024_gr:`\ ` eq. 2.23
    have a typo, due to which a factor of 2 is missing from the square root.
    These sources have the correct equation:
    :gowling_2021:`\ ` eq. 2.4 and
    :maki_msc:`\ ` eq. 2.95.
    :espinosa_2010:`\ `, eq. 97 is the same equation, but written slightly differently.
    It should be noted that $v_{CJ} \in [0, 1] \forall \alpha_+ \geq 0$.

    The Chapman-Jouguet speed can be different for other models,
    but for all detonations $v_\text{wall} \geq v_{CJ,\text{bag}}$.

    :param alpha_plus: $\alpha_+$, transition strength at the wall
    :return: $v_{CJ}$, Chapman-Jouguet speed
    """
    return 1/np.sqrt(3) * (1 + np.sqrt(2*alpha_plus + 3*alpha_plus**2)) / (1 + alpha_plus)


def v_chapman_jouguet_const_cs[T: FloatOrArr](model: "ConstCSModel", alpha_theta_bar_plus: T) -> T:
    r"""$v_{CJ}$, Chapman-Jouguet speed for the constant sound speed model.

    $$v_{CJ} = \frac{
    1 + \sqrt{ 3\alpha_{\bar{\Theta}_+} ( 1 - c_{s,b}^2 + 3 c_{s,b}^2 \alpha_{\bar{\Theta}_+} ) }
    }{
    \frac{1}{c_{s,b}} + 3 c_{s,b} \alpha_{\bar{\Theta}_+}
    }$$
    :giese_2020:`\ ` eq. 55,
    :maki_msc:`\ ` eq. 2.148.
    For detonations $\alpha_{\bar{\Theta}_+} = \alpha_{\bar{\Theta}_n}$.

    :param model: constant sound speed model
    :param alpha_theta_bar_plus: $\alpha_{\bar{\Theta}_+}$, transition strength for the pseudotrace at the wall
    :return: $v_{CJ}$, Chapman-Jouguet speed
    :raises ValueError: if $v_{CJ} \notin [c_{s,b}, 1]$
    """
    discriminant = 3*alpha_theta_bar_plus * (1 - model.csb2 + 3 * model.csb2 * alpha_theta_bar_plus)
    denominator = 1/model.csb + 3 * model.csb * alpha_theta_bar_plus
    ret = (1 + np.sqrt(discriminant)) / denominator
    # if np.any(ret > 1):
    #     if np.isscalar(ret):
    #         ret = 1 - np.sqrt(discriminant) / denominator
    #     else:
    #         inds = ret > 1
    #         ret[inds] = 1 - np.sqrt(discriminant[inds]) / denominator[inds]
    if np.any(ret < model.csb) or np.any(ret > 1):
        raise ValueError(f"Invalid v_CJ for alpha_theta_bar_plus={alpha_theta_bar_plus}: {ret}")
    return ret  # pyrefly: ignore[bad-return]


def v_chapman_jouguet_const_cs_reference(alpha_n: th.FloatArr1D, model: "ConstCSModel") -> th.FloatArr1D:
    r"""$v_{CJ}$, Chapman-Jouguet speed for the constant sound speed model with $\mu_- = 4$.

    When $\mu_- = 4$, $\alpha_+$ is independent of ${w}_-$,
    and therefore $v_{CJ}$ can be obtained by inserting $\alpha_+({w}_n)$ and $\tilde{v}_- = c_{s,b}$ to
    :func:`pttools.bubble.v_plus.v_plus`, :maki_msc:`\ ` p. 36.
    This is used as a reference for :func:`v_chapman_jouguet_const_cs`.

    :param alpha_n: $\alpha_n$, transition strength at the nucleation temperature
    :param model: constant sound speed model with $c_{s,b}^2 = \frac{1}{3}$
    :return: $v_{CJ}$, Chapman-Jouguet speed
    :raises ValueError: if $\mu_- \neq 4$
    """
    if model.mu_b != MU_BAG:
        raise ValueError(f"This reference only works for mu_b={MU_BAG}.")
    wn = model.wn(alpha_n)
    # w_- can be arbitrary, as alpha_+ does not depend on it when mu_b = 4.
    ap = model.alpha_plus(wp=wn, wm=1)
    return np.array([v_plus(model.csb, a, sol_type=SolutionType.DETON)] for a in ap)


def wm_chapman_jouguet(
        model: "Model",
        wp: float,
        wm_guess: float | None = None,
        error_on_invalid: bool = True,
        nan_on_invalid: bool = True,
        log_invalid: bool = True) -> float:
    r"""${w}_-$, enthalpy behind the wall for a Chapman-Jouguet detonation.

    Solves ${w}_-$ for which $\tilde{v}_- = c_{s,-}({w}_-)$ fulfills the junction conditions
    with the given ${w}_+$ as the root of :func:`wm_solvable_chapman_jouguet_log`.
    As $\tilde{v}_+ > \tilde{v}_-$ for detonations, the first junction condition gives ${w}_- > {w}_+$.
    The solution is first searched with :func:`scipy.optimize.fsolve` for $\ln {w}_-$ starting from ``wm_guess``,
    which ensures that ${w}_- > 0$.
    If this fails, the solution is searched by scanning ${w}_-$ from ${w}_+$ to :data:`WM_WP_RATIO_MAX_CJ` ${w}_+$
    for a change in the sign of :func:`wm_solvable_chapman_jouguet_log`,
    as ``fsolve`` can step to the region where :func:`pttools.bubble.v_plus.v_plus` has no detonation solution.

    :param model: the equation of state
    :param wp: ${w}_+$, enthalpy in front of the wall. For detonations ${w}_+ = {w}_n$.
    :param wm_guess: starting guess for ${w}_-$
    :param error_on_invalid: whether to raise an error if the solution cannot be found
    :param nan_on_invalid: whether to return nan if the solution cannot be found
    :param log_invalid: whether to log the failures
    :return: ${w}_-$, enthalpy behind the wall
    :raises RuntimeError: if the solution cannot be found and ``error_on_invalid`` is True
    """
    if wm_guess is None:
        # Use logarithmic midpoint between wp and w_crit as the starting guess
        wm_guess = wp if wp > model.w_crit else np.exp((np.log(wp) + np.log(model.w_crit))/2)
    # The SciPy stubs require func to return an array, but a scalar is also accepted at runtime.
    wm_sol = fsolve(  # pyrefly: ignore[no-matching-overload]
        wm_solvable_chapman_jouguet_log, x0=np.array([np.log(wm_guess)]), args=(model, wp), full_output=True)
    wm: float = float(np.exp(wm_sol[0][0]))
    if wm_sol[2] == 1 and wm > wp:
        return wm

    wm_bracket = _wm_chapman_jouguet_bracket(model, wp)
    if wm_bracket is not None:
        return wm_bracket

    msg = (
        f"w_- solution was not found for w_+={wp}, model={model.name}, wm_guess={wm_guess}. " +
        ("" if error_on_invalid else f"Using w_-={wm}. ") +
        f"Reason for the failure of fsolve: {wm_sol[3].replace("\n ", "")}"
    )
    if log_invalid:
        logger.error(msg)
    if error_on_invalid:
        raise RuntimeError(msg)
    if nan_on_invalid:
        return np.nan
    return wm


#: Maximum ${w}_- / {w}_+$ for the bracketing search of :func:`wm_chapman_jouguet`
WM_WP_RATIO_MAX_CJ: float = 1e3
#: Number of points in the bracketing search of :func:`wm_chapman_jouguet`
N_BRACKET_CJ: int = 300


def _wm_chapman_jouguet_bracket(model: "Model", wp: float) -> float | None:
    r"""Find ${w}_-$ for a Chapman-Jouguet detonation by scanning for a sign change of the deviation.

    :return: the smallest root with ${w}_- > {w}_+$, or None if no root was found
    """
    wm_max = min(WM_WP_RATIO_MAX_CJ * wp, model.w_max)
    if not wm_max > wp:
        return None
    # For detonations w_- > w_+, and therefore the scan starts just above w_+.
    # A logarithmic grid is used, as w_- / w_+ can range from close to 1 up to large values for strong transitions.
    log_wms = np.linspace(np.log(wp), np.log(wm_max), N_BRACKET_CJ)[1:]

    def deviation(log_wm: float) -> float:
        return wm_solvable_chapman_jouguet_log(np.array([log_wm]), model, wp)

    # Walk through the grid in pairs of consecutive points (log_wm_prev, log_wm).
    # The deviation is nan where v_plus has no detonation solution for the alpha_+(w_+, w_-) of that w_-.
    # A root is bracketed when the deviation is finite at both ends of an interval and changes its sign.
    dev_prev = deviation(log_wms[0])
    for log_wm_prev, log_wm in itertools.pairwise(log_wms):
        dev = deviation(log_wm)
        if np.isfinite(dev_prev) and not np.isfinite(dev):
            # The interval crosses into the region where the deviation is nan.
            # The root can be between log_wm_prev and the edge of this region,
            # for example for weak transitions with c_{s,-}^2 < 1/3.
            # The edge is located with bisection, and the interval is shortened to end there,
            # so that the deviation is finite at both ends.
            log_wm_edge = finite_edge(deviation, log_wm_prev, log_wm)
            dev_edge = deviation(log_wm_edge)
            if np.sign(dev_prev) != np.sign(dev_edge):
                return float(np.exp(brentq(deviation, log_wm_prev, log_wm_edge, xtol=1e-14)))
        elif np.isfinite(dev_prev) and np.isfinite(dev) and np.sign(dev_prev) != np.sign(dev):
            # The deviation is finite in the whole interval and changes its sign, so the root is bracketed.
            # As the grid is scanned upwards, this is the smallest root with w_- > w_+.
            return float(np.exp(brentq(deviation, log_wm_prev, log_wm, xtol=1e-14)))
        # Otherwise there is no root in this interval, or the whole interval is in the nan region.
        dev_prev = dev
    return None


def wm_solvable_chapman_jouguet_log(params: th.FloatArr1D, model: "Model", wp: float) -> float:
    r"""Deviation from the junction conditions for a Chapman-Jouguet detonation as a function of $\ln {w}_-$.

    The fluid speed behind the wall is set to $\tilde{v}_- = c_{s,-}({w}_-)$,
    and $\tilde{v}_+$ is given by :func:`pttools.bubble.v_plus.v_plus` with $\alpha_+({w}_+, {w}_-)$,
    which fulfills the combined junction conditions for any equation of state, :maki_msc:`\ ` eq. 2.66, 2.67.
    The remaining deviation is that of the first junction condition
    $${w}_- \tilde{\gamma}_-^2 \tilde{v}_- = {w}_+ \tilde{\gamma}_+^2 \tilde{v}_+,$$
    :maki_msc:`\ ` eq. 2.39, multiplied by $1 - \tilde{v}_-^2$ to avoid the division
    $$\Delta = {w}_- \tilde{v}_- - {w}_+ \tilde{\gamma}_+^2 \tilde{v}_+ (1 - \tilde{v}_-^2).$$
    The parameter is $\ln {w}_-$ instead of ${w}_-$ to ensure that ${w}_- > 0$ when solving for the root.

    :param params: $\ln {w}_-$ as a single-element array
    :param model: the equation of state
    :param wp: ${w}_+$, enthalpy in front of the wall
    :return: deviation $\Delta$ from the first junction condition
    """
    wm_param = np.exp(params[0])
    vm2 = model.cs2(wm_param, Phase.BROKEN)
    vm = np.sqrt(vm2)
    # alpha_plus can be negative for Chapman-Jouguet detonations if c_{s,-}^2 < 1/3,
    # and therefore it is not validated.
    ap = model.alpha_plus(wp=wp, wm=wm_param, error_on_invalid=False, nan_on_invalid=False, log_invalid=False)
    vp = v_plus(vm, ap, sol_type=SolutionType.DETON)
    return wm_param * vm + wp * gamma2(vp) * vp * (vm2 - 1)
