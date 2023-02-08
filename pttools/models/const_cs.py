r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.analytic import AnalyticModel
from pttools.models.bag import BagModel

logger = logging.getLogger(__name__)


def cs2_to_mu(cs2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert speed of sound squared $c_s^2$ to $\mu$

    $$\mu = 1 + \frac{1}{c_s^2}$$
    """
    return 1 + 1 / cs2


class ConstCSModel(AnalyticModel):
    r"""$\mu, \nu$-model"""
    DEFAULT_LABEL_LATEX = "Constant $c_s$ model"
    DEFAULT_LABEL_UNICODE = "Constant cₛ model"
    DEFAULT_NAME = "const_cs"

    def __init__(
            self,
            css2: float, csb2: float,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            t_min: float = None,
            t_max: float = None,
            t_ref: float = 1,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            allow_invalid: bool = False):
        # Ensure that these descriptions correspond to those in the base class
        r"""
        :param a_s: prefactor of $p$ in the symmetric phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param a_b: prefactor of $p$ in the broken phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param css2: $c_{s,s}^2$, speed of sound squared in the symmetric phase
        :param csb2: $c_{s,b}^2$, speed of sound squared in the broken phase
        :param V_s: $V_s \equiv \epsilon_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b \equiv \epsilon_b$, the potential term of $p$ in the broken phase
        :param t_ref: reference temperature, usually 1 * unit of choice, e,g. 1 GeV
        :param name: custom name for the model

        TODO: Rename mu to mu_s and nu to mu_b
        """
        logger.debug(f"Initialising ConstCSModel with css2={css2}, csb2={csb2}.")
        self.css2 = self.validate_cs2(css2, "css2")
        self.csb2 = self.validate_cs2(csb2, "csb2")

        if np.isnan(css2) or np.isnan(csb2):
            raise ValueError(
                "c_{s,s}^2 and c_{s,b}^2 have to be 0 < c_s <= 1/3 for the solution to be physical. "
                "This is because g_eff is monotonic. "
                f"Got: c_{{s,s}}^2={css2}, c_{{s,b}}^2={csb2}."
            )

        self.css = np.sqrt(css2)
        self.csb = np.sqrt(csb2)
        self.mu = cs2_to_mu(css2)
        self.nu = cs2_to_mu(csb2)
        self.t_ref = t_ref
        self.const_cs_wn_const: float = 4 / 3 * (1 / self.nu - 1 / self.mu)

        label_prec = 3
        label_latex = f"Const. $c_s, c_{{ss}}^2={self.css2:.{label_prec}f}, c_{{sb}}^2={self.csb2:.{label_prec}f}$" \
            if not label_latex else label_latex
        label_unicode = f"Const. cₛ, css2={self.css2:.{label_prec}f}, csb2={self.csb2:.{label_prec}f}" \
            if not label_unicode else label_unicode

        super().__init__(
            V_s=V_s, V_b=V_b,
            a_s=a_s, a_b=a_b,
            g_s=g_s, g_b=g_b,
            t_min=t_min, t_max=t_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            allow_invalid=allow_invalid
        )

    @staticmethod
    def validate_cs2(cs2: float, name: str = "cs2") -> float:
        if cs2 < 0:
            return np.nan
        if cs2 > 1/3:
            if np.isclose(cs2, 1/3):
                logger.warning(f"{name} is slightly over 1/3. Changing it to 1/3.")
                return 1/3
            return np.nan
        return cs2

    def alpha_n(self, wn: th.FloatOrArr, allow_negative: bool = False, allow_no_transition: bool = False) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4}{3} \left( \frac{1}{\nu} - \frac{1}{\mu} + \frac{1}{w_n} (V_s - V_b) \right)$$

        :param wn: $w_n$, enthalpy of the symmetric phase at the nucleation temperature
        :param allow_negative: whether to allow unphysical negative output values (not checked for this model)
        :param allow_no_transition: allow $w_n$ for which there is no phase transition
        """
        self.check_w_for_alpha(wn, allow_negative)
        # self.check_p(wn, allow_fail=allow_no_transition)

        ret = 4/3 * (1/self.nu - 1/self.mu) + self.bag_wn_const/wn
        if (not allow_negative) and np.any(ret < 0):
            if np.isscalar(ret):
                info = f"Got negative alpha_n={ret} with wn={wn}, mu={self.mu}, nu={self.nu}."
            else:
                i = np.argmin(wn)
                info = f"Got negative alpha_n. Most problematic values: alpha_n={ret[i]}, wn={wn[i]}, mu={self.mu}, nu={self.nu}"
            logger.error(info)
            if not allow_negative:
                raise ValueError(info)
        return ret

    def alpha_n_bar(self, alpha_n: float) -> float:
        wn = self.w_n(alpha_n)
        tn = self.temp(wn, Phase.SYMMETRIC)
        return alpha_n + (1 - 1 / (3 * self.cs2(wn, Phase.BROKEN))) * \
            (self.p_temp(tn, Phase.SYMMETRIC) - self.p_temp(tn, Phase.BROKEN))

    def alpha_plus(self, wp: th.FloatOrArr, wm: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        r"""If $\nu=4 \Leftrightarrow c_{sb}=\frac{1}{\sqrt{3}}$, then $w_-$ does not affect the result."""

        self.check_w_for_alpha(wp, allow_negative)
        self.check_w_for_alpha(wm, allow_negative)

        ret = (1 - 4/self.mu)/3 - (1 - 4/self.nu)*wm/(3*wp) + self.bag_wn_const/wp
        if (not allow_negative) and np.any(ret < 0):
            if np.isscalar(ret):
                raise ValueError(f"Got negative alpha_+={ret}")
            raise ValueError(f"Got negative alpha_+, most problematic value: {np.min(ret)}")
        return ret

    def critical_temp_opt(self, temp: float) -> float:
        const = (self.V_b - self.V_s)*self.t_ref**4
        return self.a_s * (temp/self.t_ref)**self.mu - self.a_b * (temp/self.t_ref)**self.nu + const

    # def alpha_plus(
    #         self,
    #         wp: th.FloatOrArr, wm: th.FloatOrArr = None,
    #         allow_negative: bool = False, analytical: bool = True) -> th.FloatOrArr:
    #     r"""Transition strength parameter $\alpha_+$"""
    #     if not analytical:
    #         if wm is None:
    #             raise ValueError("wm must be provided for non-analytical alpha_plus.")
    #         return super().alpha_plus(wp, wm, allow_negative)
    #
    #
    #
    #     if wp < 0:
    #         logger.error("Got negative wp for alpha_plus")
    #         if not allow_negative:
    #             raise ValueError("Got negative wp for alpha_plus")
    #
    #     # V_s > V_b is handled by BaseModel
    #     return 4/(3*wp) * (self.V_s - self.V_b)

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        # Using the BagModel cs2 saves us from having to compile additional Numba functions
        if css2 == 1/3 and csb2 == 1/3:
            return BagModel.cs2

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            # Mathematical operations should be faster than conditional logic in compiled functions.
            return (phase*csb2 + (1 - phase)*css2) * np.ones_like(w)
        return cs2

    def cs2_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        # ConstCSModel.cs2() is independent of T and w
        return self.cs2(temp, phase)

    def delta_theta(self, wp: th.FloatOrArr, wm: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        return (1/4 - 1/self.mu)*wp/3 - (1/4 - 1/self.nu)*wm/3 + self.V_s - self.V_b

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$
        $$e_s = a_s (\mu - 1) T^\mu + V_s$$
        $$e_b = a_b (\nu - 1) T^\nu + V_b$$
        :giese_2021:`\ `, eq. 15.
        In the article there is a typo: the 4 there should be a $\mu$.
        """
        self.validate_temp(temp)
        e_s = (self.mu - 1) * self.a_s * temp**self.mu + self.V_s
        e_b = (self.nu - 1) * self.a_b * temp**self.nu + self.V_b
        return e_b * phase + e_s * (1 - phase)

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "css2": self.css2,
            "csb2": self.csb2,
            "mu": self.mu,
            "nu": self.nu
        }

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$
        $$p_s = a_s T^\mu - V_s$$
        $$p_b = a_b T^\nu - V_b$$
        :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        p_s = self.a_s * temp**self.mu - self.V_s
        p_b = self.a_b * temp**self.nu - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s=\frac{dp}{dT}$
        $$s_s = \mu a_s \left( \frac{T}{T_0} \right)^{\mu-1} T_0^3$$
        $$s_b = \nu a-b \left( \frac{T}{T_0} \right)^{\nu-1} T_0^3$$
        Derived from :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        s_s = self.mu * self.a_s * (temp/self.t_ref)**(self.mu-1) * self.t_ref**3
        s_b = self.nu * self.a_b * (temp/self.t_ref)**(self.nu-1) * self.t_ref**3
        return s_b * phase + s_s * (1 - phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$. Inverted from the equation of $w(T,\phi)$.
        $$T_s = T_0 \left( \frac{w}{\mu a_s T_0^4} \right)^\frac{1}{\mu}$$
        $$T_b = T_0 \left( \frac{w}{\nu a_s T_0^4} \right)^\frac{1}{\nu}$$
        """
        temp_s = self.t_ref * (w / (self.mu*self.a_s*self.t_ref**4))**(1/self.mu)
        temp_b = self.t_ref * (w / (self.nu*self.a_b*self.t_ref**4))**(1/self.nu)
        return temp_b * phase + temp_s * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w(T,\phi)$
        $$w_s = \mu a_s \left( \frac{T}{T_0} \right)^\mu T_0^4$$
        $$w_s = \nu a_s \left( \frac{T}{T_0} \right)^\nu T_0^4$$
        """
        self.validate_temp(temp)
        w_s = self.mu * self.a_s * (temp/self.t_ref)**self.mu * self.t_ref**4
        w_b = self.nu * self.a_b * (temp/self.t_ref)**self.nu * self.t_ref**4
        return w_b * phase + w_s * (1 - phase)

    def w_n(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = 1,
            allow_negative: bool = False,
            analytical: bool = True) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature
        $$w_n = \frac{a}{\alpha_n - b}$$
        where
        $$a = \frac{4}{3} (V_s - V_b)$$
        $$b = \frac{4}{3} \left( \frac{1}{\mu} - \frac{1}{\nu} \right)$$
        This can be derived from the equations for $\theta$ and $\alpha_n$.
        """
        diff = alpha_n - self.const_cs_wn_const
        if np.any(diff < 0):
            if np.isscalar(alpha_n):
                info = f"Got: wn={alpha_n}."
            else:
                i = np.argmin(diff)
                info = f"Most problematic values: alpha_n={alpha_n[i]}, diff={diff[i]}."
            msg = \
                f"Got too small alpha_n for the model \"{self.name}\". {info} " \
                f"The minimum with the given parameters is {self.const_cs_wn_const}."
            logger.error(msg)
            if not allow_negative:
                raise ValueError(msg)

        if not analytical:
            super().w_n(alpha_n, wn_guess)
        return self.bag_wn_const / (alpha_n - self.const_cs_wn_const)
