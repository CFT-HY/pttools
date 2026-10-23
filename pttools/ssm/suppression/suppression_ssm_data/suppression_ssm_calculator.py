"""Compute the kinetic energy suppression factor for a given set of simulation data."""

import logging
import os
from pathlib import Path
import typing as tp

import numpy as np

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm.nucleation import NucType
from pttools.ssm.spectrum import SSMSpectrum
from pttools.ssm.ssm import DEFAULT_LAMBDA_CORRECTION, LambdaCorrection
from pttools.ssm.suppression.suppression import SuppressionMethod
from pttools.ssm.suppression.suppression_ssm_data import SUPPRESSION_FOLDER
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: $z = kR_*$ points over which the GW power growth rate is integrated
DEFAULT_Z: th.FloatArr1D = np.logspace(0, 3, 100)
#: $r_*$ for the spectra. :py:attr:`pttools.ssm.spectrum.SSMSpectrum.pow_gw_rate` does not depend on it,
#: but a value below 1 avoids the warning of :py:meth:`pttools.ssm.spectrum.SSMSpectrum.validate_nucleation`.
R_STAR: float = 0.1
#: Columns of the output files
OUTPUT_COLUMNS: tuple[str, ...] = ("vw_sim", "alpha_sim", "sup_ssm", "Ubarf_2_ssm", "ssm_tot")


def save_columns(path: str | os.PathLike[str], data: dict[str, th.FloatArr1DOrList]) -> None:
    """Save the columns :py:data:`OUTPUT_COLUMNS` of the data as a text file with a header row.

    A text file is used instead of a binary format so that the changes in the values
    are visible in the version history.
    The values are written with :py:func:`repr`, which gives the shortest representation
    that preserves the float64 value exactly.

    :param path: path of the output file
    :param data: the data, which must contain the keys of :py:data:`OUTPUT_COLUMNS`
    """
    columns = [np.asarray(data[key], dtype=np.float64) for key in OUTPUT_COLUMNS]
    lines = [" ".join(OUTPUT_COLUMNS)]
    lines += [" ".join(repr(float(value)) for value in row) for row in zip(*columns, strict=True)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def calc_sup_ssm(
        path: str | os.PathLike[str],
        save: bool = True,
        nuc_type: NucType = NucType.EXPONENTIAL,
        lambda_correction: LambdaCorrection = DEFAULT_LAMBDA_CORRECTION,
        z: th.FloatArr1D = DEFAULT_Z,
        **kwargs: tp.Any) -> dict[str, th.FloatArr1DOrList]:
    r"""Compute the kinetic energy suppression factors of the SSM by comparing to the simulation data in a file.

    The file must be a text file with a header row and the columns
    ``vw alpha suppression_sim sim_omgw exp_omgw exp_ubarf``, where
    ``sim_omgw`` is the integrated GW power growth rate of the simulation
    $\Omega_\text{gw,sim} / [(H_n R_*)(H_n t)]$,
    ``exp_omgw`` is the same quantity as expected from the SSM of the simulation article,
    and ``exp_ubarf`` is the expected RMS fluid velocity $\bar{U}_\text{exp}$.

    The suppression factor is
    $$\Sigma = \left( \frac{\bar{U}_f^2}{\bar{U}_\text{exp}^2} \right)^2
    \frac{\Omega_\text{gw,sim}}{\Omega_\text{gw,SSM}}, \quad
    \Omega_\text{gw,SSM} = \int d\ln z \, \frac{1}{H_n R_*} \frac{d\mathcal{P}_\text{gw}}{d(H_n t)},$$
    where the integrand is :py:attr:`pttools.ssm.spectrum.SSMSpectrum.pow_gw_rate` computed without suppression,
    i.e. with the mean adiabatic index $\Gamma$ of the bubble,
    and $\bar{U}_f^2$ is :py:attr:`pttools.bubble.bubble.Bubble.ubarf2`.
    The suppression factor therefore matches the spectra that :py:class:`pttools.ssm.spectrum.SSMSpectrum`
    computes with the same settings to the simulations.

    :param path: path of the simulation data file, relative to the folder of the suppression data
    :param save: whether to save the results as ``<name>_ssm.txt`` next to the simulation data file
    :param nuc_type: nucleation type of the SSM spectra
    :param lambda_correction: kinetic energy term of $\lambda$, see :py:class:`pttools.ssm.ssm.LambdaCorrection`
        and :py:meth:`pttools.ssm.spectrum.SSMSpectrum.compute`
    :param z: $z = kR_*$ points over which the GW power growth rate is integrated
    :param kwargs: additional arguments for :py:class:`pttools.ssm.spectrum.SSMSpectrum`
    :return: the results, where ``vw_sim``, ``alpha_sim`` and ``sup_ssm`` form the suppression dataset
    """
    # If the path is absolute, the joining operator returns it as is.
    path = SUPPRESSION_FOLDER / path
    sim_data = np.loadtxt(path, skiprows=1)
    model = BagModel(alpha_n_min=min(0.001, float(sim_data[:, 1].min()) / 2))
    ln_z = np.log(z)

    out_ssm_tot = []
    Ubarf_2_ssm = []
    sup_ssm_all = []
    for vw, alpha, _, sim_omgw, _, exp_ubarf in sim_data:
        bubble = Bubble(model, v_wall=vw, alpha_n=alpha)
        spectrum = SSMSpectrum(
            bubble, y=z, r_star=R_STAR, nuc_type=nuc_type,
            suppression_method=SuppressionMethod.NONE, compute=False, **kwargs)
        spectrum.compute(lambda_correction=lambda_correction)
        ssm_tot = float(np.trapezoid(spectrum.pow_gw_rate, ln_z))
        ubarf2 = bubble.ubarf2

        out_ssm_tot.append(ssm_tot)
        Ubarf_2_ssm.append(ubarf2)
        sup_ssm_all.append((ubarf2 / exp_ubarf**2)**2 * sim_omgw / ssm_tot)

    ssm_sup_data = {
        "vw_sim": sim_data[:, 0],
        "alpha_sim": sim_data[:, 1],
        "sup_ssm": sup_ssm_all,
        "Ubarf_2_ssm": Ubarf_2_ssm,
        "ssm_tot": out_ssm_tot
    }

    if save:
        save_columns(path.with_name(f"{path.stem}_ssm.txt"), ssm_sup_data)

    return ssm_sup_data


if __name__ == "__main__":
    calc_sup_ssm("suppression_2.txt")
    calc_sup_ssm("suppression_no_hybrids.txt")
