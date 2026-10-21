"""
Plot multiple spectra
=====================

Plot the velocity and GW spectra of multiple bubbles
"""

from __future__ import annotations

from examples.utils import save_and_show_fig
from pttools.bubble import Bubble
from pttools.models import BagModel, ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.plot import plot_spectra_multi
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure


def main() -> Figure:
    """Plot the velocity and GW spectra of multiple bubbles"""
    # Create the equations of state.
    # Model.cached(), Bubble.cached() and Spectrum.cached() take the same arguments as the constructors,
    # but return an existing object if one has already been created with equal arguments in this process.
    # The cached objects are shared with the other examples, and should therefore not be modified.
    model1 = BagModel.cached(alpha_n_min=0.1)
    # Instead of giving alpha_n_min you can give a_s, a_b and V_s to specify a model.
    model2 = ConstCSModel.cached(
        css2=1/4,  # speed of sound in the symmetric phase
        csb2=1/3,  # speed of sound in the broken phase
        a_s=1.5,   # degrees of freedom coefficient for the equation of state in the symmetric phase
        a_b=1,     # degrees of freedom coefficient for the equation of state in the broken phase
        V_s=1      # potential in the symmetric phase
    )
    # Create the bubbles
    bubbles = [
        Bubble.cached(model1, v_wall=0.5, alpha_n=0.2),
        Bubble.cached(model2, v_wall=0.5, alpha_n=0.2)
    ]
    # Convert the bubbles to GW spectra
    spectra = [Spectrum.cached(bubble, r_star=0.1) for bubble in bubbles]

    fig, _axs = plot_spectra_multi(spectra)
    return fig


if __name__ == "__main__":
    _fig = main()
    save_and_show_fig(_fig, "spectra")
