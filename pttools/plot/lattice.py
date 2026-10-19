r"""Plots that compare the Sound Shell Model with hydrodynamic lattice simulations.

The style follows the figures of :correia_2025:`\ `, e.g. fig. 5.
"""

import typing as tp

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pttools.analysis.lattice import suppression_series
from pttools.lattice import LatticeComparison, LatticeRun, PowGwAsymptoteFit, Quantity
from pttools.math.asymptote import AsymptoteFit
from pttools.plot.utils import FigAndAxes, create_fig_ax
import pttools.type_hints as th

#: Colours of the simulation data, as in :correia_2025:`\ `
COLOR_GW: str = "#d55e00"
COLOR_PAR: str = "#0072b2"
COLOR_PERP: str = "#cc79a7"
COLOR_V_PAR: str = "#44aa99"
COLOR_V_PERP: str = "#aa4499"
COLOR_SSM: str = "black"

Z_LABEL: str = r"$kR_*$"
GW_RATE_LABEL: str = r"$(H_n R_*)^{-1} (H_n \Delta t)^{-1} \Delta \mathcal{P}_\text{gw}(k)$"
#: Scale of the GW power in the plots of its extrapolation,
#: see :py:attr:`pttools.lattice.run.LatticeRun.gw_asymptote_scale`
GW_ASYMPTOTE_SCALE_LABEL: str = r"(\Gamma \bar{U}_\parallel^2)_*^{-2} (H_n R_*)^{-1} (H_n {t}_\text{sh})^{-1}"
T_LABEL: str = r"$t \, [T_c^{-1}]$"


def _finish(ax: Axes, xlim: tuple[float, float] | None, ylim: tuple[float, float] | None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(visible=True)


def plot_gw_rate_comparison(
        comparison: LatticeComparison,
        fig: Figure | None = None,
        ax: Axes | None = None,
        show_unsuppressed: bool = True,
        show_fit_range: bool = True,
        xlim: tuple[float, float] | None = (1, 1e4),
        ylim: tuple[float, float] | None = None) -> FigAndAxes:
    r"""Plot the GW power growth rate of a simulation and of the Sound Shell Model scaled by the suppression factor.

    This corresponds to the "Soundshell" curves of :correia_2025:`\ ` fig. 5 (bottom),
    with the suppression factor fitted to the simulation instead of taken from a suppression dataset.

    :param comparison: the comparison of the simulation and the SSM
    :param fig: figure to plot in
    :param ax: axes to plot in
    :param show_unsuppressed: whether to also plot the SSM spectrum without the suppression factor
    :param show_fit_range: whether to shade the regions that are excluded from the fit
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis, chosen automatically from the peak if None
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    spectrum = comparison.spectrum
    ssm = comparison.ssm
    fit = comparison.fit
    sup = fit.factor

    ax.plot(spectrum.y, spectrum.pow_gw_rate, color=COLOR_GW, label=r"$\dot{\mathcal{P}}_\text{gw}(k)$, simulation")
    ax.plot(ssm.y, sup * ssm.pow_gw_rate, color=COLOR_SSM, ls=":", label=rf"SSM $\times {sup:.3g}$")
    if show_unsuppressed:
        ax.plot(ssm.y, ssm.pow_gw_rate, color="grey", ls=":", lw=0.8, label="SSM, unsuppressed")
    if show_fit_range:
        shade = {"color": "grey", "alpha": 0.15, "lw": 0}
        ax.axvspan(1e-10, fit.y_min, **shade)
        ax.axvspan(fit.y_max, 1e10, label="Excluded from the fit", **shade)
        if np.isfinite(fit.y_artifact):
            ax.axvline(fit.y_artifact, color="grey", ls="--", lw=0.8)

    if ylim is None:
        peak = np.nanmax(spectrum.pow_gw_rate[(spectrum.y >= fit.y_min) & (spectrum.y <= fit.y_max)])
        ylim = (peak * 1e-6, peak * 10 * max(1., 1 / sup))
    ax.set_xlabel(Z_LABEL)
    ax.set_ylabel(GW_RATE_LABEL)
    ax.set_title(comparison.run.label_latex, fontsize=9)
    _finish(ax, xlim, ylim)
    ax.legend(fontsize=7, loc="lower left")
    return fig, ax


def plot_velocity_comparison(
        comparison: LatticeComparison,
        fig: Figure | None = None,
        ax: Axes | None = None,
        show_v: bool = True,
        xlim: tuple[float, float] | None = (1, 1e4),
        ylim: tuple[float, float] | None = (1e-7, 1e-1)) -> FigAndAxes:
    r"""Plot the velocity power spectra of a simulation and of the Sound Shell Model.

    The simulation spectra are those of the compressional and vortical components of the enthalpy-weighted
    4-velocity $U^i$ (:correia_2025:`\ ` eq. 20, fig. 3) and optionally of the 3-velocity $v^i$.
    The SSM spectrum is $\bar{U}_f^2 \mathcal{P}_{\tilde{v}}$, see
    :py:attr:`pttools.ssm.spectrum.SSMSpectrum.pow_U_par`.
    The vortical spectra are divided by two to give the power per transverse polarisation, as in
    :correia_2025:`\ ` fig. 3.

    :param comparison: the comparison of the simulation and the SSM
    :param fig: figure to plot in
    :param ax: axes to plot in
    :param show_v: whether to also plot the spectra of the 3-velocity
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    spectrum = comparison.spectrum
    ssm = comparison.ssm
    ax.plot(spectrum.y, spectrum.pow_U_par, color=COLOR_PAR, label=r"$\mathcal{P}_{U_{\parallel}}$")
    ax.plot(spectrum.y, 0.5 * spectrum.pow_U_perp, color=COLOR_PERP, label=r"$\mathcal{P}_{U_{\perp}} / 2$")
    if show_v:
        ax.plot(spectrum.y, spectrum.pow_v_par, color=COLOR_V_PAR, lw=0.8, label=r"$\mathcal{P}_{v_{\parallel}}$")
        ax.plot(
            spectrum.y, 0.5 * spectrum.pow_v_perp, color=COLOR_V_PERP, lw=0.8, label=r"$\mathcal{P}_{v_{\perp}} / 2$")
    ax.plot(
        ssm.y, ssm.pow_U_par, color=COLOR_SSM, ls=":", label=r"SSM, $\bar{U}_f^2 \mathcal{P}_{\tilde{v}}$")
    ax.set_xlabel(Z_LABEL)
    ax.set_ylabel(r"$\mathcal{P}(k)$")
    ax.set_title(f"{comparison.run.label_latex}, {spectrum.label_latex.split(', ', 1)[-1]}", fontsize=9)
    _finish(ax, xlim, ylim)
    ax.legend(fontsize=7, loc="lower left")
    return fig, ax


def plot_spectra_comparison(comparison: LatticeComparison, figsize: tuple[float, float] = (10, 4)) -> Figure:
    """Plot the velocity spectra and the GW power growth rate side by side.

    :param comparison: the comparison of the simulation and the SSM
    :param figsize: size of the figure
    :return: the figure
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plot_velocity_comparison(comparison, fig=fig, ax=axs[0])
    plot_gw_rate_comparison(comparison, fig=fig, ax=axs[1])
    fig.tight_layout()
    return fig


def _t_sh_axis(ax: Axes, t_sh: float) -> None:
    """Add a secondary x-axis in units of the shock time."""
    def to_t_sh(t: npt.ArrayLike) -> npt.ArrayLike:
        return np.asarray(t) / t_sh

    def from_t_sh(x: npt.ArrayLike) -> npt.ArrayLike:
        return np.asarray(x) * t_sh

    sec = ax.secondary_xaxis("top", functions=(to_t_sh, from_t_sh))
    sec.set_xlabel(r"$t / t_\text{sh}$")


def plot_ubar_comparison(
        comparison: LatticeComparison,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot the RMS velocities of a simulation as functions of time, and the corresponding SSM values.

    The simulation curves are those of :correia_2025:`\ ` fig. 2 (top left).
    The SSM values are constants:
    the RMS velocity of the spectrum $\bar{U}_\parallel$ (with the nucleation history)
    and the single-bubble value $\bar{U}_f$ of :py:attr:`pttools.bubble.bubble.Bubble.ubarf`.

    :param comparison: the comparison of the simulation and the SSM
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    run = comparison.run
    for quantity, color, label in (
            (Quantity.X_DIV, COLOR_PAR, r"$\bar{U}_\parallel$"),
            (Quantity.X_ROT, COLOR_PERP, r"$\bar{U}_\perp$"),
            (Quantity.VEL_DIV, COLOR_V_PAR, r"$\bar{v}_\parallel$"),
            (Quantity.VEL_ROT, COLOR_V_PERP, r"$\bar{v}_\perp$")):
        t, ubar = run.rms_series(quantity)
        ax.plot(t, ubar, color=color, label=label)
    ax.plot(run.global_times, run.ubarf_global, color="#332288", lw=0.8, label=r"$\bar{U}_f$ (global)")
    ax.axhline(comparison.ssm.ubar_par, color=COLOR_SSM, ls=":", label=r"SSM $\bar{U}_\parallel$")
    ax.axhline(comparison.ssm.bubble.ubarf, color=COLOR_SSM, ls="--", lw=0.8, label=r"Bubble $\bar{U}_f$")
    ax.axvline(comparison.spectrum.t, color="grey", lw=0.8)
    ax.set_xlabel(r"$t \, [T_c^{-1}]$")
    ax.set_ylabel(r"$\bar{U}, \bar{v}$")
    ax.set_ylim(bottom=0)
    ax.grid(visible=True)
    _t_sh_axis(ax, run.t_sh)
    ax.legend(fontsize=7)
    return fig, ax


def plot_integral_scale_comparison(
        comparison: LatticeComparison,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot the integral scales of a simulation as functions of time, and the corresponding SSM value.

    The simulation curves are those of :correia_2025:`\ ` fig. 2 (bottom left).

    :param comparison: the comparison of the simulation and the SSM
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    run = comparison.run
    for quantity, color, label in (
            (Quantity.X_DIV, COLOR_PAR, r"$\xi(U_{\parallel})$"),
            (Quantity.X_ROT, COLOR_PERP, r"$\xi(U_{\perp})$")):
        steps = run.steps(quantity)[1:]
        t, xi = run.integral_scale_series(quantity, steps)
        ax.plot(t, xi / run.R_star, color=color, label=label)
    ax.axhline(comparison.ssm.integral_scale_par, color=COLOR_SSM, ls=":", label=r"SSM $\xi(U_{\parallel})$")
    ax.axvline(comparison.spectrum.t, color="grey", lw=0.8)
    ax.set_xlabel(r"$t \, [T_c^{-1}]$")
    ax.set_ylabel(r"$\xi / R_*$")
    ax.set_ylim(bottom=0)
    ax.grid(visible=True)
    _t_sh_axis(ax, run.t_sh)
    ax.legend(fontsize=7)
    return fig, ax


def plot_kinetic_energy_comparison(
        comparison: LatticeComparison,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot the kinetic energy fractions and the mean adiabatic index of a simulation, and the SSM values.

    The simulation curves are $K_\parallel = \Gamma \bar{U}_\parallel^2$ (:correia_2025:`\ ` eq. 41),
    the kinetic energy fraction $K$ of the global quantities and $\Gamma = \bar{w} / \bar{\epsilon}_0$.
    The SSM values are $\Gamma \bar{U}_\parallel^2$ of the spectrum
    and $K$ and $\Gamma$ of the single-bubble fluid profile.

    :param comparison: the comparison of the simulation and the SSM
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    run = comparison.run
    bubble = comparison.ssm.bubble
    steps = run.steps(Quantity.X_DIV)
    t = run.time(steps)
    k_par = np.array([run.kinetic_energy_fraction_par(int(step)) for step in steps])
    ax.plot(t, k_par, color=COLOR_PAR, label=r"$K_\parallel = \Gamma \bar{U}_\parallel^2$")
    ax.plot(run.global_times, run.kinetic_energy_fraction_global, color="#332288", lw=0.8, label=r"$K$ (global)")
    ax.axhline(bubble.mean_adiabatic_index * comparison.ssm.ubar_par**2, color=COLOR_SSM, ls=":",
               label=r"SSM $\Gamma \bar{U}_\parallel^2$")
    ax.axhline(bubble.kinetic_energy_fraction, color=COLOR_SSM, ls="--", lw=0.8, label=r"Bubble $K$")
    ax.set_xlabel(r"$t \, [T_c^{-1}]$")
    ax.set_ylabel(r"$K$")
    ax.set_ylim(bottom=0)
    ax.grid(visible=True)
    _t_sh_axis(ax, run.t_sh)

    ax2 = ax.twinx()
    ax2.plot(run.global_times, run.enthalpy_global / run.epsilon_0, color="grey", lw=0.8)
    ax2.axhline(bubble.mean_adiabatic_index, color="grey", ls="--", lw=0.8)
    ax2.set_ylabel(r"$\Gamma$ (grey)")
    ax.legend(fontsize=7)
    return fig, ax


def plot_gw_rate_time_comparison(
        comparison: LatticeComparison,
        delta_steps: int | None = None,
        fig: Figure | None = None,
        axs: tp.Sequence[Axes] | None = None,
        **kwargs: tp.Any) -> tuple[Figure, tp.Sequence[Axes]]:
    r"""Plot the growth rate of $\Omega_\text{gw}$ and the fitted suppression factor as functions of time.

    The simulated flow decays, and therefore the GW power growth rate and the suppression factor
    depend on the time at which they are measured.

    :param comparison: the comparison of the simulation and the SSM
    :param delta_steps: the difference of the time steps of the finite differences,
        defaults to that of the spectrum of the comparison, see
        :py:attr:`pttools.lattice.spectrum.LatticeSpectrum.delta_steps`
    :param fig: figure to plot in
    :param axs: two axes to plot in
    :param kwargs: additional arguments for :py:func:`pttools.analysis.lattice.suppression_series`
    :return: the figure and the axes
    """
    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    elif fig is None:
        fig = tp.cast(Figure, axs[0].get_figure())
    run = comparison.run
    ssm = comparison.ssm
    if delta_steps is None:
        delta_steps = comparison.spectrum.delta_steps

    t, rate = run.omega_gw_rate_series(delta_steps)
    axs[0].plot(t, rate, color=COLOR_GW, label="Simulation")
    axs[0].axhline(ssm.omega_gw_rate, color="grey", ls=":", label="SSM, unsuppressed")
    axs[0].axhline(comparison.suppression * ssm.omega_gw_rate, color=COLOR_SSM, ls=":",
                   label=rf"SSM $\times {comparison.suppression:.3g}$")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$(H_n R_*)^{-1} \, \Delta \Omega_\text{gw} / (H_n \Delta t)$")

    t_sup, sup = suppression_series(run, ssm, delta_steps, **kwargs)
    axs[1].plot(t_sup, sup, color=COLOR_GW, label="Fitted at each time")
    axs[1].axhline(comparison.suppression, color=COLOR_SSM, ls=":", label="Fitted at the comparison time")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$\Sigma$")
    axs[1].set_xlabel(r"$t \, [T_c^{-1}]$")
    for ax in axs:
        ax.axvline(0.5 * (comparison.spectrum.t + comparison.spectrum.t_prev), color="grey", lw=0.8)
        ax.grid(visible=True)
        ax.legend(fontsize=7)
    _t_sh_axis(axs[0], run.t_sh)
    axs[0].set_title(run.label_latex, fontsize=9)
    fig.tight_layout()
    return fig, axs


# -----
# Extrapolation of the GW power to late times, :correia_2025:`\ ` fig. 6
# -----

def _mode_colors(n: int) -> npt.NDArray[np.float64]:
    r"""Colours of the wavenumbers from light to dark, as in :correia_2025:`\ ` fig. 6."""
    cmap = LinearSegmentedColormap.from_list("gw_modes", ["white", COLOR_GW])
    return cmap(np.linspace(0.33, 1, n))


def _extrapolation_times(t_start: float, t_end: float) -> npt.NDArray[np.float64]:
    """Times at which the fits are plotted, extending to twice the end of the simulation."""
    return np.linspace(t_start, 2 * t_end, 500)


def plot_pow_gw_time_fits(
        fit: PowGwAsymptoteFit,
        scale: float = 1.,
        every: int = 5,
        ylim: tuple[float, float] | None = (1e-6, 5e-2),
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot $\mathcal{P}_\text{gw}(k, t)$ of selected wavenumbers and their fits as functions of time.

    :correia_2025:`\ ` fig. 6, top row.

    :param fit: the fits of each wavenumber, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param scale: the GW power is divided by this, see :py:attr:`pttools.lattice.run.LatticeRun.gw_asymptote_scale`
    :param every: plot every n:th wavenumber
    :param ylim: limits of the y-axis, by default those of the article
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    colors = _mode_colors(fit.y.size)
    for i in range(0, fit.y.size, every):
        color = colors[-i - 1]
        ax.plot(fit.t, fit.pow_gw[:, i] / scale, color=color, lw=1)
        mode_fit = fit.fits[i]
        if mode_fit is not None:
            t = _extrapolation_times(mode_fit.t_start, mode_fit.t_end)
            ax.plot(t, mode_fit(t) / scale, color=color, ls="--", lw=1.5)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(rf"${GW_ASYMPTOTE_SCALE_LABEL} \mathcal{{P}}_\text{{gw}}(k)$")
    ax.set_title(rf"${fit.y[0]:.1f} \leq kR_* \leq {fit.y[-1]:.1f}$", fontsize=9)
    ax.grid(visible=True)
    return fig, ax


def plot_gw_growth_index(
        fit: PowGwAsymptoteFit,
        prediction: tuple[float, float] | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot the fitted indices $d(k)$ of the approach of $\mathcal{P}_\text{gw}(k, t)$ to its asymptote.

    :correia_2025:`\ ` fig. 6, second row.

    :param fit: the fits of each wavenumber, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param prediction: the predicted index $2\zeta + 2\lambda - 1$ and its uncertainty,
        see :py:meth:`pttools.lattice.run.LatticeRun.gw_growth_index_prediction`
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    ax.errorbar(fit.y, fit.index, yerr=fit.index_err, color="orange", ls="none", marker="o", ms=4)
    if prediction is not None:
        value, err = prediction
        ax.axhline(value, color="black", ls="--", label=r"$2 \zeta + 2 \lambda - 1$")
        ax.axhspan(value - err, value + err, color="silver", alpha=0.5, lw=0)
        ax.legend(fontsize=7)
    ax.set_xlim(fit.y[0] - 1, fit.y[-1] + 1)
    ax.set_ylim(-0.1, 4)
    ax.set_xlabel(Z_LABEL)
    ax.set_ylabel(r"$d(k)$")
    ax.grid(visible=True)
    return fig, ax


def plot_pow_gw_asymptote(
        fit: PowGwAsymptoteFit,
        scale: float = 1.,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot the extrapolated $\mathcal{P}_\text{gw}^\infty(k)$ and the GW power spectrum at the end of the simulation.

    :correia_2025:`\ ` fig. 6, third row.

    :param fit: the fits of each wavenumber, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param scale: the GW power is divided by this, see :py:attr:`pttools.lattice.run.LatticeRun.gw_asymptote_scale`
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    pow_inf = fit.pow_gw_inf / scale
    err = fit.pow_gw_inf_err / scale
    ax.fill_between(fit.y, pow_inf - err, pow_inf + err, color="silver", lw=0)
    ax.plot(fit.y, fit.pow_gw[-1] / scale, color=COLOR_GW, label=r"${t}_\text{end}$")
    ax.plot(fit.y, pow_inf, color="black", ls="--", label=r"$\mathcal{P}_\text{gw}^\infty(k)$")
    _finish(ax, xlim=(fit.y[0], fit.y[-1]), ylim=None)
    ax.set_xlabel(Z_LABEL)
    ax.set_ylabel(rf"${GW_ASYMPTOTE_SCALE_LABEL} \mathcal{{P}}_\text{{gw}}(k)$")
    ax.legend(fontsize=7)
    return fig, ax


def plot_omega_gw_asymptote(
        run: LatticeRun,
        omega_fit: AsymptoteFit,
        pow_fit: PowGwAsymptoteFit | None = None,
        scale: float = 1.,
        fig: Figure | None = None,
        ax: Axes | None = None) -> FigAndAxes:
    r"""Plot $\Omega_\text{gw}(t)$, its fit and the extrapolated $\Omega_\text{gw}^\infty$.

    :correia_2025:`\ ` fig. 6, bottom row.
    If the fits of each wavenumber are given, the partial GW power $\Omega_\text{par}(t)$ of the fitted wavenumbers
    and its extrapolation from the fits are also plotted.

    :param run: the simulation run
    :param omega_fit: the fit of $\Omega_\text{gw}(t)$, see :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
    :param pow_fit: the fits of each wavenumber, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param scale: the GW power is divided by this, see :py:attr:`pttools.lattice.run.LatticeRun.gw_asymptote_scale`
    :param fig: figure to plot in
    :param ax: axes to plot in
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    t, omega = run.omega_gw_series()
    t = t[t <= omega_fit.t_end]
    omega = omega[:t.size]
    ax.plot(t, omega / scale, color="black", label=r"$\Omega_\text{tot}$")
    t_extrap = _extrapolation_times(omega_fit.t_start, omega_fit.t_end)
    ax.plot(t_extrap, omega_fit(t_extrap) / scale, color="black", ls=":")
    if pow_fit is not None:
        ax.plot(pow_fit.t, pow_fit.omega_gw_partial() / scale, color=COLOR_GW, label=r"$\Omega_\text{par}$")
        t_par = _extrapolation_times(max(fit.t_start for fit in pow_fit.fits if fit is not None), pow_fit.t[-1])
        ax.plot(t_par, pow_fit.omega_gw_partial(t_par) / scale, color=COLOR_GW, ls=":")
    value, err = omega_fit.value / scale, omega_fit.value_err / scale
    ax.axhline(value, color="black", ls="--", label=r"$\Omega_\text{tot}^\infty$")
    ax.axhspan(value - err, value + err, color="silver", alpha=0.5, lw=0)
    ax.set_xlim(t[0], t_extrap[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(rf"${GW_ASYMPTOTE_SCALE_LABEL} \Omega_\text{{gw}}$")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(visible=True)
    return fig, ax


def plot_gw_asymptote(
        run: LatticeRun,
        axs: tp.Sequence[Axes] | th.AxesArr1D,
        start_fraction: float = 0.8,
        step_end: int | None = None,
        y_min: float = 8.,
        y_max: float = 81.) -> None:
    r"""Plot the extrapolation of the GW power of a run to late times, :correia_2025:`\ ` fig. 6.

    :param run: the simulation run
    :param axs: the four axes to plot in, from top to bottom in the figure of the article
    :param start_fraction: see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param step_end: last time step of the fits, defaults to the last available step
    :param y_min: smallest $z = kR_*$ to fit
    :param y_max: largest $z = kR_*$ to fit
    """
    scale = run.gw_asymptote_scale
    pow_fit = run.pow_gw_asymptote(y_min=y_min, y_max=y_max, start_fraction=start_fraction, step_end=step_end)
    plot_pow_gw_time_fits(pow_fit, scale, ax=axs[0])
    plot_gw_growth_index(pow_fit, run.gw_growth_index_prediction(step_end), ax=axs[1])
    plot_pow_gw_asymptote(pow_fit, scale, ax=axs[2])
    plot_omega_gw_asymptote(run, run.omega_gw_asymptote(step_end=step_end), pow_fit, scale, ax=axs[3])
    axs[0].set_title(f"{run.label_latex}\n{axs[0].get_title()}", fontsize=9)


def plot_pow_gw_asymptote_comparison(
        comparison: LatticeComparison,
        pow_fit: PowGwAsymptoteFit | None = None,
        step_end: int | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        xlim: tuple[float, float] | None = (1, 1e4),
        ylim: tuple[float, float] | None = None) -> FigAndAxes:
    r"""Plot the GW power spectrum of a simulation scaled to its asymptote, and that of the Sound Shell Model.

    The simulation spectrum at the time of the comparison is scaled to the asymptote with
    :py:meth:`pttools.lattice.spectrum.LatticeSpectrum.pow_gw_scaled_to_asymptote`.
    If the fits of each wavenumber are given, their asymptotes $\mathcal{P}_\text{gw}^\infty(k)$ are also plotted,
    scaled to the $r_*$ of the spectrum.
    The SSM spectrum is plotted both with the suppression factor that is fitted to the GW power growth rate
    of the simulation and without suppression.

    :param comparison: the comparison of the simulation and the SSM
    :param pow_fit: the fits of each wavenumber, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    :param step_end: last time step of the fit of $\Omega_\text{gw}(t)$, defaults to the last available step
    :param fig: figure to plot in
    :param ax: axes to plot in
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis, chosen automatically from the peak if None
    :return: the figure and the axes
    """
    fig, ax = create_fig_ax(fig, ax)
    spectrum = comparison.spectrum
    ssm = comparison.ssm
    sup = comparison.fit.factor
    ratio = spectrum.omega_gw_asymptote_ratio(step_end=step_end)
    pow_scaled = ratio * spectrum.pow_gw

    ax.plot(spectrum.y, spectrum.pow_gw, color=COLOR_GW, lw=0.8, alpha=0.5, label=f"Simulation, $t={spectrum.t:.0f}$")
    ax.plot(
        spectrum.y, pow_scaled, color=COLOR_GW,
        label=rf"Simulation $\times \Omega_\text{{gw}}^\infty / \Omega_\text{{gw}}(t) = {ratio:.3g}$")
    if pow_fit is not None:
        r_scale = (spectrum.r_star / comparison.run.r_star)**2
        pow_inf = r_scale * pow_fit.pow_gw_inf
        err = r_scale * pow_fit.pow_gw_inf_err
        ax.fill_between(pow_fit.y, pow_inf - err, pow_inf + err, color="silver", lw=0)
        ax.plot(pow_fit.y, pow_inf, color="black", ls="--", label=r"$\mathcal{P}_\text{gw}^\infty(k)$, fits")
    ax.plot(ssm.y, sup * ssm.pow_gw, color=COLOR_SSM, ls=":", label=rf"SSM $\times {sup:.3g}$")
    ax.plot(ssm.y, ssm.pow_gw, color="grey", ls=":", lw=0.8, label="SSM, unsuppressed")

    if ylim is None:
        peak = max(np.nanmax(pow_scaled), np.nanmax(ssm.pow_gw))
        ylim = (peak * 1e-6, peak * 3)
    ax.set_xlabel(Z_LABEL)
    ax.set_ylabel(r"$\mathcal{P}_\text{gw}(k)$")
    ax.set_title(f"{comparison.run.label_latex}, $r_*={spectrum.r_star:.3g}$", fontsize=9)
    _finish(ax, xlim, ylim)
    ax.legend(fontsize=7, loc="lower left")
    return fig, ax
