"""Solver for the fluid velocity profile of a bubble"""

import logging
import time
import typing as tp

import numpy as np

from pttools.bubble import alpha
from pttools.bubble.phase import Phase
from pttools.bubble import chapman_jouguet
from pttools.bubble import const
from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR, cs2_bag_scalar
from pttools.bubble import fluid_bag
from pttools.bubble.fluid_base import GenericSolverOutput
from pttools.bubble.fluid_detonation import sound_shell_detonation
from pttools.bubble.fluid_gksvdv import sound_shell_gksvdv
from pttools.bubble.fluid_hybrid import sound_shell_solver_hybrid
from pttools.bubble.fluid_sub_def import sound_shell_solver_deflagration, sound_shell_solver_deflagration_reverse
from pttools.bubble import fluid_reference
from pttools.bubble.integrate import DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG
from pttools.bubble import props
from pttools.bubble import relativity
from pttools.bubble.solution_type import SolutionType, cannot_be_sub_def, validate_solution_type
from pttools.bubble.solution_type_bag import identify_solution_type_bag
from pttools.speedup import NAN_ARR

if tp.TYPE_CHECKING:
    from pttools.models import Model

logger = logging.getLogger(__name__)


def sound_shell_generic(
            model: "Model",
            v_wall: float,
            alpha_n: float,
            sol_type: SolutionType | None = None,
            wn: float | None = None,
            vp_guess: float | None = None,
            wn_guess: float | None = None,
            wp_guess: float | None = None,
            wm_guess: float | None = None,
            wn_rtol: float = 1e-4,
            alpha_n_max_bag: float | None = None,
            high_alpha_n: bool | None = None,
            t_end: float = const.DEFAULT_T_END,
            n_xi: int = const.DEFAULT_N_XI,
            thin_shell_limit: int = const.THIN_SHELL_T_POINTS_MIN,
            reverse: bool = False,
            allow_failure: bool = False,
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_success: bool = True,
            log_high_alpha_n_failures: bool = False
        ) -> GenericSolverOutput:
    """Generic fluid shell solver

    In most cases you should not have to call this directly. Create a Bubble instead.
    """
    if use_giese_solver:
        return sound_shell_gksvdv(
            model=model, v_wall=v_wall, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess, wm_guess=wm_guess
        )

    start_time = time.perf_counter()
    if alpha_n_max_bag is None:
        alpha_n_max_bag = alpha.alpha_n_max_deflagration_bag(
            v_wall, df_dtau_ptr=DF_DTAU_PTR_BAG,
            ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar)
    if high_alpha_n is None:
        high_alpha_n = alpha_n > alpha_n_max_bag

    if wn is None or np.isnan(wn):
        wn = model.wn(alpha_n, wn_guess=wn_guess)
    # The shock curve hits v=0 here.
    cs_n = np.sqrt(model.cs2(wn, Phase.SYMMETRIC))

    if use_bag_solver and model.DEFAULT_NAME == "bag":
        if high_alpha_n:
            logger.info(
                "Got model=%s, v_wall=%s, alpha_n=%s, for which there is no bag model solution.",
                model.label_unicode, v_wall, alpha_n
            )
            return NAN_ARR, NAN_ARR, NAN_ARR, SolutionType.ERROR, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                True, time.perf_counter() - start_time

        logger.info(
            "Using bag solver for model=%s, v_wall=%s, alpha_n=%s",
            model.label_unicode, v_wall, alpha_n
        )
        sol_type2 = identify_solution_type_bag(
            v_wall, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG,
            ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar)
        if sol_type is not None and sol_type != sol_type2:
            raise ValueError(
                f"Bag model gave a different solution type ({sol_type2}) than what was given ({sol_type})."
            )

        v, w, xi = fluid_bag.sound_shell_bag(
            v_wall, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
            ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar)
        # The results of the old solver are scaled to wn=1
        w = w * wn
        if np.any(np.isnan(v)):
            return v, w, xi, sol_type2, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                True, time.perf_counter() - start_time

        vp, vm, vp_tilde, vm_tilde, wp, wm, wn, wm_sh = props.v_and_w_from_solution(v, w, xi, v_wall, sol_type2)

        # The wm_guess is not needed for the bag model
        v_cj: float = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn=wn, wm_guess=wm)
        return v, w, xi, sol_type2, \
            vp, vm, vp_tilde, vm_tilde, np.nan, np.nan, np.nan, wp, wm, wm_sh, v_cj, \
            False, time.perf_counter() - start_time

    sol_type = validate_solution_type(
        model,
        v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type,
        wn=wn, wm_guess=wm_guess
    )

    # Load and scale reference data
    using_ref = False
    vp_ref, vm_ref, vp_tilde_ref, vm_tilde_ref, wp_ref, wm_ref = fluid_reference.ref().get(v_wall, alpha_n, sol_type)

    if vp_guess is None or np.isnan(vp_guess):
        using_ref = True
        vp_guess = vp_ref
        vp_tilde_guess = vp_tilde_ref
    else:
        vp_tilde_guess = relativity.lorentz(v_wall, vp_guess)

    # The reference data has wn=1 and therefore has to be scaled with wn.
    if wp_guess is None or np.isnan(wp_guess):
        using_ref = True
        # Deflagrations have their own method for guessing wp, so this can be nan.
        wp_guess = wp_ref * wn
    if wm_guess is None or np.isnan(wp_guess):
        using_ref = True
        if np.isnan(wm_ref):
            logger.warning(
                "No reference data for v_wall=%s, alpha_n=%s. Using an arbitrary starting guess.",
                v_wall, alpha_n
            )
            # This is arbitrary, but seems to work OK.
            wm_guess = 0.3 * wn
        else:
            wm_guess = wm_ref * wn
    # if wn_guess is None:
    #     wn_guess = min(wp_guess, wm_guess)

    if using_ref and np.any(np.isnan((vp_ref, vm_ref, vp_tilde_ref, vm_tilde_ref, wp_ref, wm_ref))):
        logger.warning(
            "Using arbitrary starting guesses at v_wall=%s, alpha_n=%s,"
            "as all starting guesses were not provided, and the reference has nan values.",
            v_wall, alpha_n
        )

    if vp_guess < 0 or vp_guess > 1 or vp_tilde_guess < 0 or vp_tilde_guess > 1 or wm_guess < 0 or wp_guess < wn:
        raise ValueError(
            f"Got invalid guesses: vp_tilde={vp_tilde_guess}, wp={wp_guess}, wm={wm_guess}"
            f"for v_wall={v_wall}, alpha_n={alpha_n}, wn={wn_guess}"
        )

    v_cj = chapman_jouguet.v_chapman_jouguet(model, alpha_n, wn=wn, wm_guess=wm_guess)

    if log_success:
        logger.info(
            "Solving fluid shell for model=%s, v_wall=%s, alpha_n=%s " +
            (f"(alpha_n_max_bag={alpha_n_max_bag}) " if high_alpha_n and sol_type != SolutionType.DETON else "") +
            "with sol_type=%s, v_cj=%s, wn=%s "
            "and starting guesses vp=%s vp_tilde=%s, wp=%s, wm=%s, wn=%s",
            model.label_unicode, v_wall, alpha_n,
            sol_type, v_cj, wn,
            vp_guess, vp_tilde_guess, wp_guess, wm_guess, wn_guess
        )

    # Detonations are the simplest case
    if sol_type == SolutionType.DETON:
        v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
            sound_shell_detonation(
                model, v_wall, alpha_n, wn, v_cj,
                vm_tilde_guess=vm_tilde_ref, wm_guess=wm_ref, t_end=t_end, n_xi=n_xi,
            )
    elif sol_type == SolutionType.SUB_DEF:
        if cannot_be_sub_def(model, v_wall, wn):
            raise ValueError(
                f"Invalid parameters for a subsonic deflagration: model={model.name}, v_wall={v_wall}, wn={wn}. "
                "Decrease v_wall or increase csb2."
            )

        # In more advanced models,
        # the direction of the integration will probably have to be determined by trial and error.
        if reverse:
            logger.warning("Using reverse deflagration solver, which has not been properly tested.")
            v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
                sound_shell_solver_deflagration_reverse(
                    model, start_time, v_wall, alpha_n, wn,
                    t_end=t_end, n_xi=n_xi
                )
        else:
            v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
                sound_shell_solver_deflagration(
                    model, start_time,
                    v_wall, alpha_n, wn,
                    cs_n=cs_n, v_cj=v_cj,
                    high_alpha_n=high_alpha_n,
                    wm_guess=wm_guess, vp_guess=vp_guess, wp_guess=wp_guess, wn_rtol=wn_rtol, t_end=t_end, n_xi=n_xi,
                    thin_shell_limit=thin_shell_limit,
                    allow_failure=allow_failure, log_high_alpha_n_failures=log_high_alpha_n_failures
                )
    elif sol_type == SolutionType.HYBRID:
        v, w, xi, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, solution_found = \
            sound_shell_solver_hybrid(
                model, start_time,
                v_wall, alpha_n, wn,
                cs_n=cs_n,
                v_cj=v_cj,
                high_alpha_n=high_alpha_n,
                vp_tilde_guess=vp_tilde_guess,
                wp_guess=wp_guess,
                wm_guess=wm_guess,
                wn_rtol=wn_rtol,
                t_end=t_end, n_xi=n_xi,
                thin_shell_limit=thin_shell_limit,
                allow_failure=allow_failure,
                log_high_alpha_n_failures=log_high_alpha_n_failures
            )
    else:
        raise ValueError(f"Invalid solution type: {sol_type}")

    dxi = 1. / n_xi
    # Behind and ahead of the bubble the fluid is still
    xif = np.array([xi[-1] + dxi, 1])
    xib = np.array([0, xi[0] - dxi])
    vf = vb = np.zeros_like(xif)
    wf = np.ones_like(xif) * wn
    w_center = min(wm, w[0])
    wb = np.ones_like(vb) * w_center

    v = np.concatenate((vb, v, vf))
    w = np.concatenate((wb, w, wf))
    xi = np.concatenate((xib, xi, xif))
    if sol_type in (SolutionType.SUB_DEF, SolutionType.HYBRID, SolutionType.DETON):
        v[v < 0] = 0.

    elapsed = time.perf_counter() - start_time
    if solution_found and log_success:
        logger.info(
            "Solved fluid shell for model=%s, v_wall=%s, alpha_n=%s, sol_type=%s. Elapsed: %s s",
            model.label_unicode, v_wall, alpha_n, sol_type, elapsed
        )
    return v, w, xi, sol_type, vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh, v_cj, \
        not solution_found, elapsed


fluid_shell_generic = sound_shell_generic
