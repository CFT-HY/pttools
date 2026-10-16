r"""Plots that compare the Sound Shell Model with hydrodynamic lattice simulations.

The style follows the figures of :correia_2025:`\ `, e.g. fig. 5.
"""

import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pttools.analysis.lattice import LatticeComparison, gw_rate_series, suppression_series
from pttools.analysis.utils import FigAndAxes, create_fig_ax
from pttools.lattice import Quantity

#: Colours of the simulation data, as in :correia_2025:`\ `
COLOR_GW: str = "#d55e00"
COLOR_PAR: str = "#0072b2"
COLOR_PERP: str = "#cc79a7"
COLOR_V_PAR: str = "#44aa99"
COLOR_V_PERP: str = "#aa4499"
COLOR_SSM: str = "black"

Z_LABEL: str = r"$kR_*$"
GW_RATE_LABEL: str = r"$(H_n R_*)^{-1} (H_n \Delta t)^{-1} \Delta \mathcal{P}_\text{gw}(k)$"


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
        delta_steps: int,
        fig: Figure | None = None,
        axs: tp.Sequence[Axes] | None = None,
        **kwargs: tp.Any) -> tuple[Figure, tp.Sequence[Axes]]:
    r"""Plot the growth rate of $\Omega_\text{gw}$ and the fitted suppression factor as functions of time.

    The simulated flow decays, and therefore the GW power growth rate and the suppression factor
    depend on the time at which they are measured.

    :param comparison: the comparison of the simulation and the SSM
    :param delta_steps: the difference of the time steps of the finite differences
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

    t, rate = gw_rate_series(run, delta_steps)
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
