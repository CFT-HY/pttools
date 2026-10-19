r"""Compute the suppression factors of the Sound Shell Model for the simulations of Correia et al.

The suppression factors are fitted to the GW power growth rates of the $4096^3$ simulations of :correia_2025:`\ `
with :py:func:`pttools.analysis.lattice.compare_lattice_run`, excluding the lattice artifact at high $k$.
The time steps of the growth rates are those of :correia_2025:`\ ` fig. 5.
The SSM spectra are computed with simultaneous nucleation as in the simulations, without suppression,
and with the default accuracy settings of :py:class:`pttools.ssm.spectrum.SSMSpectrum`,
so that the suppression factors apply to the spectra computed with the default settings.

For reference, the factors are also computed with exponential nucleation and with the log-space fit,
but the dataset (``sup_ssm``) uses simultaneous nucleation and the integral fit.
The suppression datasets of :gowling_2021:`\ ` are instead computed with exponential nucleation,
see :py:mod:`pttools.ssm.suppression.suppression_ssm_data.suppression_ssm_calculator`.

This requires the data of the simulations, see :py:mod:`pttools.lattice.correia_2025`.
Run with ``python -m pttools.ssm.suppression.suppression_ssm_data.suppression_lattice_calculator``.
"""

import logging
from pathlib import Path

import numpy as np

from pttools.analysis.lattice import compare_lattice_run, ssm_spectrum_for_run
from pttools.lattice import CORREIA_2025_RUNS, load_correia_2025_run
from pttools.ssm.nucleation import NucType
from pttools.ssm.suppression.fit import SuppressionFitMethod
from pttools.ssm.suppression.suppression_ssm_data import SUPPRESSION_FOLDER

logger: logging.Logger = logging.getLogger(__name__)

#: Name of the output files
OUTPUT_NAME: str = "suppression_correia_2025"
#: Smallest $z = kR_*$ included in the fits. This excludes the lowest bin, which contains only one mode.
Y_MIN: float = 2.


def calc_sup_lattice(save: bool = True, path: Path = SUPPRESSION_FOLDER) -> dict[str, np.ndarray]:
    r"""Compute the suppression factors for the simulations of :correia_2025:`\ `.

    :param save: whether to save the results as ``suppression_correia_2025.txt``,
        which is loaded by :py:data:`pttools.ssm.suppression.suppression.CORREIA_2025`
    :param path: directory of the output files
    :return: the results, where ``vw_sim``, ``alpha_sim`` and ``sup_ssm`` form the suppression dataset
    """
    columns: dict[str, list[float]] = {
        "vw_sim": [], "alpha_sim": [], "sup_ssm": [],
        "sup_ssm_log_lsq": [], "sup_ssm_exponential": [],
        "step": [], "step_prev": [], "y_min": [], "y_max": [], "y_artifact": [],
        "ubar_par_sim": [], "ubar_par_ssm": [], "omega_gw_rate_sim": [], "omega_gw_rate_ssm": [],
    }
    for name, meta in CORREIA_2025_RUNS.items():
        run = load_correia_2025_run(name)
        ssm = ssm_spectrum_for_run(run, nuc_type=NucType.SIMULTANEOUS)
        comparison = compare_lattice_run(
            run, step=meta.fig_step, step_prev=meta.fig_step_prev, ssm=ssm,
            method=SuppressionFitMethod.INTEGRAL, y_min=Y_MIN)
        log_lsq = compare_lattice_run(
            run, step=meta.fig_step, step_prev=meta.fig_step_prev, ssm=ssm,
            method=SuppressionFitMethod.LOG_LSQ, y_min=Y_MIN)
        exponential = compare_lattice_run(
            run, step=meta.fig_step, step_prev=meta.fig_step_prev, nuc_type=NucType.EXPONENTIAL,
            method=SuppressionFitMethod.INTEGRAL, y_min=Y_MIN)
        fit = comparison.fit
        values = {
            "vw_sim": run.v_wall, "alpha_sim": run.alpha_n, "sup_ssm": fit.factor,
            "sup_ssm_log_lsq": log_lsq.fit.factor, "sup_ssm_exponential": exponential.fit.factor,
            "step": meta.fig_step, "step_prev": meta.fig_step_prev,
            "y_min": fit.y_min, "y_max": fit.y_max, "y_artifact": fit.y_artifact,
            "ubar_par_sim": comparison.spectrum.ubar_par, "ubar_par_ssm": ssm.ubar_par,
            "omega_gw_rate_sim": comparison.spectrum.omega_gw_rate, "omega_gw_rate_ssm": ssm.omega_gw_rate,
        }
        for key, value in values.items():
            columns[key].append(value)
        logger.info("%s: %s", name, values)

    data = {key: np.array(value) for key, value in columns.items()}
    if save:
        np.savetxt(
            path / f"{OUTPUT_NAME}.txt",
            np.column_stack(list(data.values())),
            fmt="%.8g",
            header=" ".join(data.keys()),
            comments=""
        )
    return data


if __name__ == "__main__":
    from pttools.logging import setup_logging
    setup_logging()
    calc_sup_lattice()
