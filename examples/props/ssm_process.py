"""
Sound Shell Model process
=========================

Plot the intermediate steps of the Sound Shell Model.
"""

from __future__ import annotations

import numpy as np

from examples.utils import save_and_show_figs
from pttools.bubble import Bubble
from pttools.math import as_latex
from pttools.models import ConstCSModel
from pttools.omgw0 import DEFAULT_T_STAR, Spectrum
from pttools.ssm import NucType
import pttools.type_hints as th
from pttools.utils.time import log_duration

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure


@log_duration
def main() -> tuple[Figure, Figure]:
    """Plot the intermediate steps of the Sound Shell Model for a single bubble

    :return: the figure with all the LISA noise sources,
        and the figure with only the LISA instrument noise
    """
    model = ConstCSModel.cached(css2=1/4, csb2=1/4, alpha_n_min=0.1)

    # Create and simulate the fluid profile of a bubble.
    bubble = Bubble.cached(model, v_wall=0.6, alpha_n=0.1, label_with_model=False)

    # Compute the gravitational wave spectrum for the bubble.
    spectrum = Spectrum.cached(
        bubble, nuc_type=NucType.EXPONENTIAL, r_star=0.1, low_k=False,
        y=np.logspace(-0.5, 3, 1000),
        label_latex=rf"$T_*={as_latex(DEFAULT_T_STAR)} \ \mathrm{{GeV}}$"
    )
    fig, _axs = spectrum.plot_multi_flat(legend=True)
    fig_ins, _axs_ins = spectrum.plot_multi_flat(legend=True, eb=False, gb=False)
    return fig, fig_ins


if __name__ == "__main__":
    _fig, _fig_ins = main()
    save_and_show_figs({
        "ssm_process": _fig,
        "ssm_process_ins": _fig_ins,
    })
