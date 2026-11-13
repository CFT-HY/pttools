"""A solution of the hydrodynamic equations."""

from __future__ import annotations

import logging
import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.bubble.alpha import alpha_n_max_deflagration_bag
from pttools.bubble.bubble.computed import ComputedBubble
from pttools.bubble.bubble.nucleated import BaseNucleatedBubble
from pttools.bubble.const import DEFAULT_N_XI, DEFAULT_T_END, JUNCTION_RTOL, THIN_SHELL_T_POINTS_MIN
from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR
from pttools.bubble.export import BUBBLE_FIELDS
from pttools.bubble.fluid import sound_shell_generic
from pttools.bubble.integrate import DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG
from pttools.bubble.junction import junction_condition_deviations
from pttools.bubble.junction_entropy import check_entropy_fluxes
from pttools.bubble.phase import Phase
from pttools.bubble.props import find_phase
from pttools.bubble.solution_type import SolutionType, validate_solution_type
import pttools.type_hints as th
from pttools.utils.validation import ensure_float

if tp.TYPE_CHECKING:
    from pttools.models.bag import BagModel
    from pttools.models.const_cs import ConstCSModel
    from pttools.models.model import Model

if th.TYPE_CHECKING_OR_DOCS:
    from pttools.utils.fields import Fields

__all__ = ["PSI_N_MIN_DETON_LTE", "Bubble", "BubbleArr", "BubbleArr2D"]

logger: logging.Logger = logging.getLogger(__name__)

# If alpha_n is closer than this to the bag model alpha_n_max for deflagrations, alpha_n is considered high
_HIGH_ALPHA_N_MARGIN: float = 0.05
#: Detonations with Psi_n below this may not exist according to the LTE approximation, Ai et al. (2023) p. 16
PSI_N_MIN_DETON_LTE: float = 0.75


class Bubble(ComputedBubble, BaseNucleatedBubble):
    """A solution of the hydrodynamic equations, aka. a bubble."""

    FIELDS: tp.ClassVar[Fields] = BUBBLE_FIELDS

    # Most of the statements are attribute declarations with their documentation.
    def __init__(  # noqa: PLR0915
            self,
            model: Model,
            v_wall: float,
            alpha_n: float,
            solve: bool = True,
            sol_type: SolutionType | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            label_with_model: bool = True,
            wn_guess: float | None = None,
            wm_guess: float | None = None,
            theta_bar: bool = False,
            t_end: float = DEFAULT_T_END,
            n_xi: int = DEFAULT_N_XI,
            thin_shell_t_points_min: int = THIN_SHELL_T_POINTS_MIN,
            low_v_wall_threshold: float = 0.1,
            n_xi_fix_factor: int = 10,
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_success: bool = False,
            allow_invalid: bool = False,
            log_invalid: bool = True):
        r"""Create a bubble, and optionally solve its fluid velocity profile.

        :param model: The equation of state object
        :param v_wall: Wall velocity $v_\text{wall}$
        :param alpha_n: Transition strength $\alpha_n$
        :param solve: Whether to solve the bubble immediately
        :param sol_type: Solution type (deflagration, hybrid, detonation).
            If None, it will be determined automatically.
        :param label_latex: LaTeX label for plots
        :param label_unicode: Unicode label for plots
        :param wn_guess: Initial guess for the enthalpy at the nucleation temperature $w_n$
        :param wm_guess: Initial guess for the enthalpy behind the wall $w_-$
        :param theta_bar: Whether the provided alpha_n is actually alpha_theta_bar_n
        :param t_end: The maximum value for the fluid shell ODE integration parameter
        :param n_xi: Number of points in the fluid velocity profile
        :param thin_shell_t_points_min: Limit of points for a shell to be so thin
            that it should be re-computed with more points
        :param use_bag_solver: Whether to use the bag model specific fluid shell solver
        :param use_giese_solver: Whether to use the Giese et al. solver for the constant sound speed model
        :param log_success: Whether to log successful solutions
        :param allow_invalid: Whether to allow invalid solutions
        :param log_invalid: Whether to log invalid solutions
        """
        # Validate alpha_n for computing wn
        alpha_n = ensure_float(alpha_n, "alpha_n")
        if not theta_bar:
            model.validate_alpha_n(alpha_n, allow_invalid=allow_invalid, log_invalid=log_invalid)

        super().__init__(
            model=model, v_wall=v_wall,
            w_outside=model.wn(alpha_n, wn_guess=wn_guess, theta_bar=theta_bar),
            wm_guess=wm_guess, t_end=t_end, n_xi=n_xi
        )
        # Below this line, use self.v_wall, self.wn and self.wm_guess

        # -----
        # Validate input parameters
        # -----
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese et al. solvers cannot be used at the same time.")
        self.use_bag_solver: bool = use_bag_solver
        self.use_giese_solver: bool = use_giese_solver

        if not 0 < self.v_wall <= 1:
            raise ValueError(f"Invalid v_wall={self.v_wall}. Should have 0 < v_wall <= 1.")
        if self.v_wall < low_v_wall_threshold:
            if self.n_xi == DEFAULT_N_XI:
                logger.info(
                    "Got n_xi=%s for v_wall=%s < %s. This may lead to an inaccurate solution. "
                    "Since n_xi = DEFAULT_N_XI, multiplying n_xi by %s for an automatic fix.",
                    n_xi, v_wall, low_v_wall_threshold, n_xi_fix_factor
                )
                self.n_xi *= n_xi_fix_factor
            elif self.n_xi < DEFAULT_N_XI:
                logger.warning(
                    "Got n_xi=%s for v_wall=%s < %s. This may lead to an inaccurate solution. "
                    "Please increase n_xi.",
                    n_xi, v_wall, low_v_wall_threshold
                )

        # -----
        # Set alpha_n and alpha_theta_bar_n
        # -----
        self.alpha_n: float = model.alpha_n_from_alpha_theta_bar_n(alpha_theta_bar_n=alpha_n, wn=self.wn) \
            if theta_bar else alpha_n
        self.alpha_theta_bar_n: float = alpha_n \
            if theta_bar else \
            model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=self.wn)
        if theta_bar:
            model.validate_alpha_n(self.alpha_n, allow_invalid=allow_invalid, log_invalid=log_invalid)

        # Use self.alpha_n instead of alpha_n, as the latter is alpha_theta_bar_n when theta_bar=True.
        self.sol_type: SolutionType = validate_solution_type(
            model,
            v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=sol_type,
            wn=self.wn, wm_guess=wm_guess
        )

        # -----
        # Set parameters
        # -----
        self.thin_shell_t_points_min: int = thin_shell_t_points_min
        self.log_success: bool = log_success

        # -----
        # Compute parameters
        # -----
        self.Tn: float = model.temp(self.wn, Phase.SYMMETRIC)
        if self.Tn > model.T_crit:
            msg = f"Bubbles form only when T_nuc < T_crit. Got: T_nuc={self.Tn}, T_crit={model.T_crit}"
            if log_invalid:
                logger.error(msg)
            if not allow_invalid:
                raise ValueError(msg)

        self.Psi_n: float = model.Psi_n(self.wn)
        self.alpha_theta_bar_n_min_lte: float
        self.alpha_theta_bar_n_max_lte: float
        self.alpha_theta_bar_n_min_lte, self.alpha_theta_bar_n_max_lte = self.validate_lte(log_invalid=True)

        # LaTeX labels are not supported in Plotly 3D plots.
        # https://github.com/plotly/plotly.js/issues/608
        # These require alpha_n and therefore cannot be before its validation.
        default_latex, default_unicode = self.default_labels(model, v_wall, alpha_n, label_with_model)
        self.label_latex: str = default_latex if label_latex is None else label_latex
        self.label_unicode: str = default_unicode if label_unicode is None else label_unicode

        # -----
        # Output values
        # -----
        self.alpha_plus: float = np.nan
        self.alpha_theta_bar_plus: float = np.nan

        self.entropy_flux_p_sh: float = np.nan
        self.entropy_flux_m_sh: float = np.nan
        self.entropy_flux_diff_sh: float = np.nan
        self.sn: float = np.nan
        self.sm_sh: float = np.nan
        self.Tm_sh: float = np.nan
        self.v_sh: float = np.nan
        self.vm_sh: float = np.nan
        self.vm_tilde_sh: float = np.nan
        self.v_cj: float = np.nan
        self.wm_sh: float = np.nan

        # Flags
        self.unphysical_alpha_plus: bool = False

        if solve:
            self.solve()
        elif log_success:
            logger.info(
                "Initialized a bubble with: "
                "model=%s, v_w=%s, alpha_n=%s, T_nuc=%s, w_nuc=%s",
                self.model.label_unicode, v_wall, alpha_n, self.Tn, self.wn
            )

    def lte_params_str(self, model: BagModel | ConstCSModel) -> str:
        """LTE parameters as a string for debug messages."""
        return \
            f"wn={self.wn}, sol_type={self.sol_type}, Psi_n={self.Psi_n}, " \
            f"mu_s={model.mu_s}, mu_b={model.mu_b}"

    def _set_properties(self) -> None:
        """Extract properties from the solution."""
        self.solved = True
        # If the solver failed, the failure has already been reported by the solver,
        # and the resulting invalid values would only add noise to the logs.
        log_invalid = not self.solver_failed
        self.alpha_plus = self.model.alpha_plus(
            self.wp, self.wm, vp_tilde=self.vp_tilde, sol_type=self.sol_type,
            error_on_invalid=False, nan_on_invalid=True, log_invalid=log_invalid
        )
        self.alpha_theta_bar_plus = self.model.alpha_theta_bar_plus(
            self.wp, error_on_invalid=False, nan_on_invalid=True, log_invalid=log_invalid
        )
        self.phase = find_phase(self.xi, self.v_wall, self.sol_type)

        self.sn = self.model.s(self.wn, Phase.SYMMETRIC)
        self.sm = self.model.s(self.wm, Phase.BROKEN)
        self.Tm = self.model.temp(self.wm, Phase.BROKEN)
        self.w_center = self.w[0]
        self.T_center = self.model.temp(self.w_center, Phase.BROKEN)

        # In detonations the shock and the wall have merged
        if self.sol_type == SolutionType.DETON:
            self.sp = self.sn
            self.sm_sh = self.sm
            self.Tp = self.Tn
            self.Tm_sh = self.Tm
        else:
            self.sp = self.model.s(self.wp, Phase.SYMMETRIC)
            self.sm_sh = self.model.s(self.wm_sh, Phase.SYMMETRIC)
            self.Tp = self.model.temp(self.wp, Phase.SYMMETRIC)
            self.Tm_sh = self.model.temp(self.wm_sh, Phase.SYMMETRIC)

    def solve(
            self,
            sum_rtol_warning: float = 1.5e-2,
            sum_rtol_error: float = 5e-2,
            error_prec: str = ".4f",
            use_bag_solver: bool = False,
            use_giese_solver: bool = False,
            log_high_alpha_n_failures: bool = True,
            log_negative_entropy: bool = True) -> None:
        """Simulate the fluid velocity profile of the bubble."""
        super().solve()

        use_bag_solver = self.use_bag_solver or use_bag_solver
        use_giese_solver = self.use_giese_solver or use_giese_solver
        if use_bag_solver and use_giese_solver:
            raise ValueError("Both bag and Giese et al. solvers cannot be used at the same time.")

        alpha_n_max_bag = alpha_n_max_deflagration_bag(
            self.v_wall, df_dtau_ptr=DF_DTAU_PTR_BAG,
            ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=CS2_BAG_SCALAR_PTR)
        high_alpha_n = alpha_n_max_bag - self.alpha_n < _HIGH_ALPHA_N_MARGIN

        try:
            # Todo: make the solver errors more specific
            self.v, self.w, self.xi, self.sol_type, \
                self.vp, self.vm, self.vp_tilde, self.vm_tilde, \
                self.v_sh, self.vm_sh, self.vm_tilde_sh, \
                self.wp, self.wm, self.wm_sh, self.v_cj, self.solver_failed, self.solving_duration = \
                sound_shell_generic(
                    model=self.model,
                    v_wall=self.v_wall, alpha_n=self.alpha_n, sol_type=self.sol_type,
                    wn=self.wn,
                    alpha_n_max_bag=alpha_n_max_bag,
                    high_alpha_n=high_alpha_n,
                    t_end=self.t_end,
                    n_xi=self.n_xi,
                    thin_shell_limit=self.thin_shell_t_points_min,
                    use_bag_solver=use_bag_solver,
                    use_giese_solver=use_giese_solver,
                    log_success=self.log_success,
                    log_high_alpha_n_failures=log_high_alpha_n_failures
                )
            if self.solver_failed:
                # This is already reported by the individual solvers
                msg = f"Solver failed with model={self.model.label_unicode}, " \
                      f"v_wall={self.v_wall}, alpha_n={self.alpha_n}"
                # logger.error(msg)
                self.add_note(msg)
        except (IndexError, RuntimeError) as e:
            msg = f"Solver crashed with model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}."
            logger.exception(msg, exc_info=e)
            self.add_note(msg)
            self.solving_attempted = True
            self.solver_crashed = True
            return

        self._set_properties()

        # If the solver did not produce a fluid profile, there is nothing to validate,
        # and the validations would only produce warnings about the nan values.
        if self.solver_failed and np.all(np.isnan(self.v)):
            self.failed = True
            return

        # Validity checking for the solution
        self.failed = any([
            self.solver_crashed,
            self.solver_failed,
            self.validate_xi_monotonic(),
            self.validate_junction(),
            self.validate_alpha_plus(),
            self.validate_entropy_flux(),
            self.validate_entropy_density(log_negative=log_negative_entropy),
            self.validate_thermal_energy_density(),
            self.validate_kappa_omega(
                sum_rtol_warning=sum_rtol_warning,
                sum_rtol_error=sum_rtol_error,
                error_prec=error_prec,
                high_alpha_n=high_alpha_n,
                log_high_alpha_n_failures=log_high_alpha_n_failures
            )
        ])

    # =====
    # Validation
    # =====

    def validate_alpha_plus(self) -> bool:
        r"""Validate $\alpha_+$."""
        fail = np.isnan(self.alpha_plus)
        if fail:
            self.alpha_plus = self.model.alpha_plus(
                self.wp, self.wm, vp_tilde=self.vp_tilde, sol_type=self.sol_type,
                error_on_invalid=False, nan_on_invalid=False, log_invalid=False
            )
            msg = f"Got invalid alpha_plus={self.alpha_plus} with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, " \
                  f"alpha_n={self.alpha_n}, sol_type={self.sol_type}, " \
                  f"wp={self.wp}, wm={self.wm}, vp_tilde={self.vp_tilde}."
            logger.error(msg)
            self.add_note(msg)
            self.unphysical_alpha_plus = True
        return fail

    def validate_entropy_density(self, log_negative: bool = True) -> bool:
        """Validate that the total entropy density is not decreasing."""
        fail = self.va_entropy_density_diff < 0
        if fail:
            msg = "Entropy density change should not be negative! Now entropy is decreasing. " \
                  f"Got: {self.va_entropy_density_diff} with " \
                  f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            if log_negative:
                logger.warning(msg)
            self.add_note(msg)
            self.negative_net_entropy_change = True
        return fail

    def validate_entropy_flux(self) -> bool:
        """Validate entropy fluxes at the bubble wall."""
        fail_wall, self.entropy_flux_p, self.entropy_flux_m, self.entropy_flux_diff = check_entropy_fluxes(
            self.model,
            v1_tilde=self.vp_tilde, v2_tilde=self.vm_tilde,
            w1=self.wp, w2=self.wm,
            phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN
        )
        fail_sh, self.entropy_flux_p_sh, self.entropy_flux_m_sh, self.entropy_flux_diff_sh = check_entropy_fluxes(
            self.model,
            v1_tilde=self.vp_tilde_sh, v2_tilde=self.vm_tilde_sh,
            w1=self.wn, w2=self.wm_sh,
            phase1=Phase.SYMMETRIC, phase2=Phase.BROKEN if self.sol_type == SolutionType.DETON else Phase.SYMMETRIC
        )
        fail = fail_wall or fail_sh
        if fail:
            msg = "Entropy fluxes should not be negative! " \
                f"Got entropy_flux_p={self.entropy_flux_p}, entropy_flux_m={self.entropy_flux_m}, " \
                f"entropy_flux_diff={self.entropy_flux_diff}, " \
                f"entropy_flux_p_sh={self.entropy_flux_p_sh}, entropy_flux_m_sh={self.entropy_flux_m_sh}, " \
                f"entropy_flux_diff_sh={self.entropy_flux_diff_sh} with " \
                f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            logger.error(msg)
            self.add_note(msg)
            self.negative_entropy_flux = True
        return fail

    def validate_junction(
            self,
            rtol: float = JUNCTION_RTOL,
            rtol_sh: float = 50 * JUNCTION_RTOL) -> bool:
        """Validate that the junction conditions at the bubble wall have been solved correctly."""
        devs_wall = junction_condition_deviations(
            v1=self.vp_tilde, w1=self.wp, p1=self.model.p(self.wp, Phase.SYMMETRIC),
            v2=self.vm_tilde, w2=self.wm, p2=self.model.p(self.wm, Phase.BROKEN)
        )
        devs_sh = junction_condition_deviations(
            v1=self.vp_tilde_sh, w1=self.wn, p1=self.model.p(self.wn, Phase.SYMMETRIC),
            v2=self.vm_tilde_sh, w2=self.wm_sh,
            p2=self.model.p(
                w=self.wm_sh,
                phase=Phase.BROKEN if self.sol_type == SolutionType.DETON else Phase.SYMMETRIC
            )
        )
        # This should be the same as in the junction solver,
        # as the choice there depends on the direction of the solving.
        w = self.wp if self.sol_type == SolutionType.DETON else self.wm

        devs_rel_wall = devs_wall / w
        devs_rel_sh = devs_sh / self.wm_sh

        # The shock is found by selecting a point of the curve instead of solving exactly.
        # Therefore, it may not quite fit within the desired tolerance.
        fail_wall = np.max(np.abs(devs_rel_wall)) > rtol
        fail_sh = np.max(np.abs(devs_rel_sh)) > rtol_sh
        fail = fail_wall or fail_sh
        if fail:
            msg1 = ("wall and shock deviate" if fail_wall else "shock deviates") if fail_sh else "wall deviates"

            msg = \
                f"The {msg1} too much from the junction conditions. " \
                "This indicates a numerical error or a bug. " \
                f"Got dev1={devs_wall[0]}, dev2={devs_wall[1]}, " \
                f"dev1_rel={devs_rel_wall[0]}, dev2_rel{devs_rel_wall[1]}, "\
                f"dev1_sh={devs_sh[0]}, dev2_sh={devs_sh[1]}, " \
                f"dev1_rel_sh={devs_rel_sh[0]}, dev2_rel_sh={devs_rel_sh[1]} with " \
                f"model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            logger.error(msg)
            self.add_note(msg)
        self.invalid_junction = fail
        return fail

    def validate_kappa_omega(
            self,
            sum_rtol_warning: float,
            sum_rtol_error: float,
            error_prec: str,
            high_alpha_n: bool,
            log_high_alpha_n_failures: bool = True) -> bool:
        r"""Validate that $\kappa + \omega = 1$."""
        fail = not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_warning)
        if fail:
            sum_err = not np.isclose(self.kappa + self.omega, 1, rtol=sum_rtol_error)
            if sum_err:
                self.numerical_error = True
            msg = "Got κ+ω != 1. " + \
                ("Marking the solution to have a numerical error. " if sum_err else "") + \
                f"Got: κ={self.kappa:{error_prec}}, ω={self.omega:{error_prec}}, "\
                f"κ+ω={self.kappa + self.omega:{error_prec}} " \
                f"with model={self.model.label_unicode}, v_wall={self.v_wall}, alpha_n={self.alpha_n}"
            if log_high_alpha_n_failures or (not high_alpha_n) or self.sol_type == SolutionType.DETON:
                if sum_err:
                    logger.error(msg)
                else:
                    logger.warning(msg)
            self.add_note(msg)
        return fail

    def validate_lte(self, log_invalid: bool = True) -> tuple[float, float]:
        r"""Validate whether the parameters are within the limits permitted by the LTE approximation.

        In this context, the local thermal equilibrium (LTE) approximation means no entropy generation.

        "if a deflagration or hybrid solution exists within LTE,
        it will also exist when the out-of-equilibrium effects are considered."
        :ai_2023:`\ ` p. 16
        """
        if log_invalid and self.sol_type == SolutionType.DETON and self.Psi_n < PSI_N_MIN_DETON_LTE:
            logger.info(
                "This detonation may not exist, as LTE predicts a large alpha_n_hyb_max for Psi_n=%s < 0.75. "
                "Please see Ai et al. (2023), p. 16.",
                self.Psi_n
            )

        if self.model.DEFAULT_NAME not in ("bag", "const_cs"):
            return 0., np.inf

        # The model name check above ensures that the model is a BagModel or a ConstCSModel.
        model = tp.cast("BagModel | ConstCSModel", self.model)
        alpha_theta_bar_n_min_lte: float = model.alpha_theta_bar_n_min_lte(self.wn, self.sol_type, Psi_n=self.Psi_n)
        alpha_theta_bar_n_max_lte: float = model.alpha_theta_bar_n_max_lte(self.wn, self.sol_type, Psi_n=self.Psi_n)
        if log_invalid and (alpha_theta_bar_n_max_lte < alpha_theta_bar_n_min_lte
                            or alpha_theta_bar_n_max_lte < 0):
            logger.error(
                "Got invalid limits for alpha_theta_bar_n_lte: min=%s, max=%s for %s",
                alpha_theta_bar_n_min_lte, alpha_theta_bar_n_max_lte, self.lte_params_str(model)
            )
        if log_invalid and self.alpha_theta_bar_n < alpha_theta_bar_n_min_lte:
            logger.info(
                "Got alpha_theta_bar_n=%s < lte_min=%s for %s",
                self.alpha_theta_bar_n, alpha_theta_bar_n_min_lte, self.lte_params_str(model)
            )
        if log_invalid and self.alpha_theta_bar_n > alpha_theta_bar_n_max_lte:
            logger.info(
                "alpha_theta_bar_n=%s > lte_max=%s for %s",
                self.alpha_theta_bar_n, alpha_theta_bar_n_max_lte, self.lte_params_str(model)
            )
        return alpha_theta_bar_n_min_lte, alpha_theta_bar_n_max_lte

    def validate_thermal_energy_density(self) -> bool:
        """Check that the thermal energy density does not decrease, i.e. that the bubble is not a heat engine.

        If it decreases, a warning is logged and a note is added to the bubble.

        :return: whether the validation failed, i.e. the thermal energy density change is negative
        """
        fail = self.va_thermal_energy_density_diff < 0
        if fail:
            msg = "Thermal energy density change is negative. The bubble is therefore working as a heat engine. " \
                  f"Got: {self.va_thermal_energy_density_diff}"
            logger.warning(msg)
            self.add_note(msg)
        return fail


# These are object arrays. Numpy typing has no way of expressing the element type of an object array,
# but declaring the element type here does give the correct types when the arrays are indexed.
type BubbleArr = NDArray[Bubble]  # pyrefly: ignore[bad-specialization]
type BubbleArr2D = np.ndarray[th.Shape2D, np.dtype[Bubble]]  # pyrefly: ignore[bad-specialization]
