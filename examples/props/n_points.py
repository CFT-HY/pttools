r"""
Numbers of points
=================

Investigate how small the numbers of points of the numerical computations can be
while still producing accurate gravitational wave spectra.

The accuracy settings are

- ``n_xi``, the number of points $n_\xi$ of the fluid profile of :py:class:`pttools.bubble.bubble.Bubble`,
- ``n_z_lookup``, the number of points of the lookup arrays of :py:class:`pttools.omgw0.spectrum.Spectrum`,
- ``nx_P_tilde_gw``, the number of points of the integration of $\tilde{P}_\text{gw}$
  (by default the same as ``n_z_lookup``),
- ``nT``, the number of points of the bubble lifetime distribution integration.

Each setting is varied separately,
while the others are kept at the base values :py:data:`~examples.props.n_points.BASE`.
The resulting spectra $\Omega_{\text{gw},0} h^2(f)$ are compared with a reference spectrum,
which differs from the others only by having the reference value of the varied setting.
The relative error is shown both near the peak, i.e. where $\Omega_{\text{gw},0} \geq 10^{-3} \Omega_\text{peak}$,
and over the entire frequency range, whose tails can be many orders of magnitude below the peak.
The computation time of a spectrum is measured on a single CPU core, which corresponds to the computation of a dataset
with one worker process per core, as in :ref:`sphx_glr_auto_examples_const_cs_dataset.py`.
Too small values of $n_\xi$ also cause the bubbles to be marked as failed,
as the numerical integrals of the energy budget no longer fulfill $\kappa + \omega = 1$
within the tolerance of the validation.

The computation takes several minutes of CPU time,
and is therefore skipped when this example is run by Sphinx-Gallery for building the documentation.
Run it from the root directory of the repository with ``python -m examples.props.n_points``.

Finally, the spectra computed with the accuracy settings :py:data:`~examples.const_cs.dataset.ACCURACY`
of :ref:`sphx_glr_auto_examples_const_cs_dataset.py`
are compared with spectra computed with the reference values of all the settings.

The results show that ``nx_P_tilde_gw`` and ``nT`` can be reduced far below their defaults
with a negligible effect on the accuracy.
The accuracy and the computation time of the spectra are determined mostly by ``n_z_lookup``.
With the settings of the dataset, the error is about 1 % or less from the lowest frequencies
up to about ten times the peak frequency.
The steep high-frequency tail several orders of magnitude below the peak converges slowly with ``n_z_lookup``,
and its relative error can be tens of percent even with the default settings.
The number of points in the fluid profile does not always grow with $n_\xi$,
and therefore the error can also jump up when $n_\xi$ is increased.
"""

from concurrent.futures import Future
import dataclasses
import logging
import time
import typing as tp

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from examples.const_cs.dataset import ACCURACY, MODEL_KWARGS, Accuracy, F, warm_up
from examples.utils import save_and_show_figs
from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble
from pttools.docs.examples import is_sphinx_gallery
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.speedup.parallel import FakeFuture, get_process_pool
import pttools.type_hints as th
from pttools.utils.system import IS_READ_THE_DOCS

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Case:
    """Parameters of a test case, i.e. a model and a bubble."""

    css2: float
    csb2: float
    v_wall: float
    alpha_n: float

    @property
    def label(self) -> str:
        """LaTeX label of the case."""
        return (
            rf"$c_{{s,s}}^2={self.css2:.3f}, c_{{s,b}}^2={self.csb2:.3f}, "
            rf"v_\text{{w}}={self.v_wall}, \alpha_n={self.alpha_n}$"
        )


#: The test cases, which cover the solution types and the extremes of the parameter ranges of the dataset
CASES: tuple[Case, ...] = (
    Case(1/3, 1/4, 0.05, 0.1),
    Case(1/3, 1/3, 0.35, 0.1),
    Case(1/4, 1/3, 0.5, 0.5),
    Case(1/3, 1/4, 0.65, 0.1),
    Case(1/4, 1/4, 0.75, 1.),
    Case(1/3, 1/4, 0.9, 0.1),
    Case(1/4, 1/3, 0.95, 0.3),
)

#: Parameters of the spectra
SPECTRUM_KWARGS: dict[str, tp.Any] = {"beta_tilde": 100, "T_star": 200, "g_star": 100, "f": F}

#: Base values of the accuracy settings, which are used when the other settings are varied
BASE: Accuracy = Accuracy(n_xi=10000, n_z_lookup=5000, nx_P_tilde_gw=1000, nT=1000)
#: Reference values of the accuracy settings
REFERENCE: Accuracy = Accuracy(n_xi=20000, n_z_lookup=20000, nx_P_tilde_gw=None, nT=10000)
#: The tested values of the accuracy settings
VALUES: dict[str, tuple[int, ...]] = {
    "n_xi": (500, 1000, 2000, 3000, 5000, 7000, 10000),
    "n_z_lookup": (500, 1000, 2000, 3000, 5000, 10000),
    "nx_P_tilde_gw": (50, 100, 200, 500, 1000, 2000),
    "nT": (100, 300, 1000, 3000),
}
#: The relative error is computed near the peak for the frequencies where $\Omega \geq$ this times the peak value
PEAK_REGION: float = 1e-3


@dataclasses.dataclass(frozen=True)
class Result:
    """Result of a single computation."""

    omgw0_h2: th.FloatArr1D
    #: $\kappa + \omega - 1$
    kappa_omega_err: float
    xi_size: int
    failed: bool
    time_bubble: float
    time_spectrum: float


def compute(case: Case, model: ConstCSModel, accuracy: Accuracy) -> Result:
    """Compute a spectrum for the given case with the given accuracy settings on a single CPU core."""
    warm_up(model)
    start_time = time.perf_counter()
    bubble = Bubble(model, v_wall=case.v_wall, alpha_n=case.alpha_n, n_xi=accuracy.n_xi, log_success=False)
    bubble_time = time.perf_counter()
    spectrum = Spectrum(bubble, **SPECTRUM_KWARGS, **accuracy.spectrum_kwargs(), parallel=False)
    omgw0_h2 = spectrum.omgw0_h2()
    end_time = time.perf_counter()
    return Result(
        omgw0_h2=omgw0_h2,
        kappa_omega_err=bubble.kappa + bubble.omega - 1,
        xi_size=bubble.xi.size,
        failed=bubble.failed,
        time_bubble=bubble_time - start_time,
        time_spectrum=end_time - bubble_time,
    )


def rel_errors(omgw0_h2: th.FloatArr1D, ref: th.FloatArr1D) -> tuple[float, float]:
    """Maximum relative errors near the peak and over the entire frequency range."""
    err = np.abs(omgw0_h2 / ref - 1)
    peak = ref >= PEAK_REGION * np.nanmax(ref)
    return float(np.nanmax(err[peak])), float(np.nanmax(err))


def plot_errors(
        cases: tuple[Case, ...],
        results: dict[tuple[int, str, int], Result],
        refs: dict[tuple[int, str], Result]) -> Figure:
    """Plot the errors and the computation times as a function of the accuracy settings."""
    fig: Figure = plt.figure(figsize=(A4_PAPER_SIZE[0] * 1.5, A4_PAPER_SIZE[0] * 1.2))
    axs = fig.subplots(3, len(VALUES), sharex="col", sharey="row")
    for i_case, case in enumerate(cases):
        color = f"C{i_case}"
        for i_param, (param, values) in enumerate(VALUES.items()):
            ref = refs[(i_case, param)].omgw0_h2
            errs = np.array([rel_errors(results[(i_case, param, value)].omgw0_h2, ref) for value in values])
            times = [results[(i_case, param, value)].time_spectrum for value in values]
            axs[0, i_param].plot(values, errs[:, 0], color=color, marker=".", label=case.label)
            axs[1, i_param].plot(values, errs[:, 1], color=color, marker=".")
            axs[2, i_param].plot(values, times, color=color, marker=".")
    for i_param, param in enumerate(VALUES):
        for i_row in range(3):
            ax = axs[i_row, i_param]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid()
        chosen = getattr(ACCURACY, param)
        for ax in axs[:, i_param]:
            if chosen is not None:
                ax.axvline(chosen, color="k", ls="--")
        for ax in axs[:2, i_param]:
            ax.axhline(1e-2, color="k", ls=":")
        axs[2, i_param].set_xlabel(rf"\texttt{{{param}}}" if plt.rcParams["text.usetex"] else param)
    axs[0, 0].set_ylabel("Max. rel. error near the peak")
    axs[1, 0].set_ylabel("Max. rel. error over all $f$")
    axs[2, 0].set_ylabel("Spectrum computation time (s)")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncols=2, fontsize="small")
    fig.suptitle(
        "Varied accuracy settings compared with the reference value of the setting\n"
        "(dashed: the value used for the dataset, dotted: 1 % error)",
        y=0.9, fontsize="medium"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_bubbles(cases: tuple[Case, ...], results: dict[tuple[int, str, int], Result]) -> Figure:
    r"""Plot $\kappa + \omega - 1$ and the size of the fluid profile as a function of $n_\xi$."""
    fig: Figure = plt.figure(figsize=(A4_PAPER_SIZE[0] * 1.5, A4_PAPER_SIZE[0] * 0.6))
    ax1, ax2 = fig.subplots(1, 2)
    values = VALUES["n_xi"]
    for i_case, case in enumerate(cases):
        res = [results[(i_case, "n_xi", value)] for value in values]
        err = np.array([abs(r.kappa_omega_err) for r in res])
        size = np.array([r.xi_size for r in res])
        failed = np.array([r.failed for r in res])
        color = f"C{i_case}"
        for ax, data in ((ax1, err), (ax2, size)):
            ax.plot(values, data, color=color, label=case.label)
            ax.scatter(np.array(values)[failed], data[failed], color=color, marker="x")
            ax.scatter(np.array(values)[~failed], data[~failed], color=color, marker="o")
    ax1.axhline(1.5e-2, color="k", ls=":", label="Tolerance of the validation")
    ax1.set_ylabel(r"$|\kappa + \omega - 1|$")
    ax2.set_ylabel(r"Number of points in the fluid profile")
    for ax in (ax1, ax2):
        ax.axvline(ACCURACY.n_xi, color="k", ls="--")
        ax.set_xlabel(r"$n_\xi$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
    ax1.set_title("x: bubble marked as failed, o: valid bubble", fontsize="medium")
    ax2.legend(fontsize="x-small")
    fig.tight_layout()
    return fig


def plot_spectra(cases: tuple[Case, ...], dataset: list[Result], best: list[Result]) -> Figure:
    """Plot the spectra computed with the accuracy settings of the dataset and with the reference settings."""
    fig: Figure = plt.figure(figsize=(A4_PAPER_SIZE[0] * 1.5, A4_PAPER_SIZE[0] * 0.6))
    ax1, ax2 = fig.subplots(1, 2, sharex=True)
    for i_case, case in enumerate(cases):
        color = f"C{i_case}"
        ax1.plot(F, best[i_case].omgw0_h2, color=color, label=case.label)
        ax1.plot(F, dataset[i_case].omgw0_h2, color=color, ls="--")
        ax2.plot(F, np.abs(dataset[i_case].omgw0_h2 / best[i_case].omgw0_h2 - 1), color=color)
    ax1.set_ylabel(r"$\Omega_{\text{gw},0} h^2$")
    ax1.set_ylim(bottom=1e-25)
    ax1.set_title("Solid: reference settings, dashed: dataset settings", fontsize="medium")
    ax2.set_ylabel("Relative error of the dataset settings")
    ax2.axhline(1e-2, color="k", ls=":")
    for ax in (ax1, ax2):
        ax.set_xlabel("$f$ (Hz)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
    ax1.legend(fontsize="x-small")
    fig.tight_layout()
    return fig


def main(cases: tuple[Case, ...] = CASES) -> tuple[Figure, Figure, Figure]:
    """Compute the spectra with the various accuracy settings and plot the errors."""
    start_time = time.perf_counter()
    models = {
        (case.css2, case.csb2): ConstCSModel(css2=case.css2, csb2=case.csb2, **MODEL_KWARGS)
        for case in cases
    }
    # The futures are created first, so that all the computations run in parallel.
    with get_process_pool(single_thread=IS_READ_THE_DOCS) as pool:
        futs: dict[tuple[int, str, int], FakeFuture | Future[Result]] = {}
        ref_futs: dict[tuple[int, str], FakeFuture | Future[Result]] = {}
        for i_case, case in enumerate(cases):
            model = models[(case.css2, case.csb2)]
            for param, values in VALUES.items():
                ref_futs[(i_case, param)] = pool.submit(
                    compute, case, model, dataclasses.replace(BASE, **{param: getattr(REFERENCE, param)}))
                for value in values:
                    futs[(i_case, param, value)] = pool.submit(
                        compute, case, model, dataclasses.replace(BASE, **{param: value}))
            futs[(i_case, "dataset", 0)] = pool.submit(compute, case, model, ACCURACY)
            futs[(i_case, "reference", 0)] = pool.submit(compute, case, model, REFERENCE)
        results = {key: fut.result() for key, fut in futs.items()}
        refs = {key: fut.result() for key, fut in ref_futs.items()}

    cpu_time = sum(res.time_bubble + res.time_spectrum for res in (*results.values(), *refs.values()))
    logger.info("The computations took %.1f s of CPU time excluding the compilation", cpu_time)
    for i_case, case in enumerate(cases):
        dataset = results[(i_case, "dataset", 0)]
        err_peak, err_all = rel_errors(dataset.omgw0_h2, results[(i_case, "reference", 0)].omgw0_h2)
        logger.info(
            "%s: max. rel. error of the dataset settings near the peak: %.2e, over all f: %.2e, "
            "bubble failed: %s, bubble time: %.3f s, spectrum time: %.3f s",
            case.label, err_peak, err_all, dataset.failed, dataset.time_bubble, dataset.time_spectrum
        )

    fig_errors = plot_errors(cases, results, refs)
    fig_bubbles = plot_bubbles(cases, results)
    fig_spectra = plot_spectra(
        cases,
        [results[(i_case, "dataset", 0)] for i_case in range(len(cases))],
        [results[(i_case, "reference", 0)] for i_case in range(len(cases))]
    )
    logger.info("Computing the accuracy comparison took %.1f s", time.perf_counter() - start_time)
    return fig_errors, fig_bubbles, fig_spectra


# The computation is skipped in the documentation, as it would take too long.
if __name__ == "__main__" and not is_sphinx_gallery():
    _fig_errors, _fig_bubbles, _fig_spectra = main()
    save_and_show_figs({
        "n_points_errors": _fig_errors,
        "n_points_bubbles": _fig_bubbles,
        "n_points_spectra": _fig_spectra,
    })
