r"""
Exact sine transforms for large z
=================================

Compare $\lvert A(z) \rvert^2$ computed with the exact sine transforms of the piecewise-linear fluid profiles
for large $z$ (``exact_large_z=True``, the default) to that computed with the approximate sine transform
and finite differences for $f'(z)$ (``exact_large_z=False``),
see :py:func:`pttools.ssm.ssm.f_fp_l`.
The $z$ array has the same density as the default lookup arrays of :py:class:`pttools.ssm.spectrum.SSMSpectrum`,
so that the effect of the finite differences is the same as in the spectra.
As $\lvert A(z) \rvert^2$ oscillates, the ratio is shown for the values averaged over a window in $\ln z$.
"""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save_and_show_fig
from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import const
from pttools.ssm.ssm import A2_e_conserving
import pttools.type_hints as th

#: $(v_{\text{wall}}, \alpha_n)$ of the fluid shells
PARAMS: tuple[tuple[float, float], ...] = ((0.5, 0.1), (0.7, 0.1), (0.95, 0.5), (0.95, 1.45))
#: $z = kR_*$, with about the same density as the default lookup arrays of SSMSpectrum
Z: th.FloatArr1D = np.logspace(0, 4, 6001)
#: Width of the averaging window in $\ln z$
WINDOW_LN_Z: float = 0.1


def window_mean(arr: th.FloatArr1D, z: th.FloatArr1D, width: float) -> th.FloatArr1D:
    r"""Moving average over a window of the given width in $\ln z$, for a logarithmically spaced $z$."""
    n = max(1, round(width / np.log(z[1] / z[0])))
    return np.convolve(arr, np.ones(n) / n, mode="same")


def main() -> Figure:
    r"""Plot $\lvert A(z) \rvert^2$ with and without the exact sine transforms for large $z$."""
    model = BagModel.cached(alpha_n_min=0.001)
    fig, axs = plt.subplots(2, len(PARAMS), figsize=(4 * len(PARAMS), 7), sharex=True, sharey="row")
    for i, (v_wall, alpha_n) in enumerate(PARAMS):
        bubble = Bubble.cached(model, v_wall=v_wall, alpha_n=alpha_n)
        kwargs = {
            "v": bubble.v, "w": bubble.w, "xi": bubble.xi, "e": bubble.e, "z": Z,
            "v_wall": bubble.v_wall, "v_sh": bubble.v_sh, "cs": const.CS0
        }
        a2_exact = A2_e_conserving(**kwargs, exact_large_z=True)[0]
        a2_approx = A2_e_conserving(**kwargs, exact_large_z=False)[0]

        ax1 = axs[0, i]
        ax1.plot(Z, Z**4 * a2_approx, label="exact_large_z=False", lw=0.5)
        ax1.plot(Z, Z**4 * a2_exact, label="exact_large_z=True", lw=0.5)
        ax1.set_title(
            rf"{bubble.sol_type.value}, $v_\mathrm{{wall}}={v_wall}$, $\alpha_n={alpha_n}$")
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        ax2 = axs[1, i]
        ratio = window_mean(a2_exact, Z, WINDOW_LN_Z) / window_mean(a2_approx, Z, WINDOW_LN_Z)
        # The edges of the window would be biased
        n_edge = round(WINDOW_LN_Z / np.log(Z[1] / Z[0]))
        ax2.plot(Z[n_edge:-n_edge], ratio[n_edge:-n_edge])
        ax2.axhline(1, color="k", lw=0.5)
        ax2.set_xlabel("$z = kR_*$")

        for ax in (ax1, ax2):
            ax.axvline(const.Z_ST_THRESH, color="gray", ls=":", lw=1)

    axs[0, 0].set_ylabel(r"$z^4 |A(z)|^2$")
    axs[0, 0].legend()
    axs[1, 0].set_ylabel(
        r"$\langle |A|^2 \rangle_\mathrm{exact} / \langle |A|^2 \rangle_\mathrm{approx}$"
        rf" ($\Delta \ln z = {WINDOW_LN_Z}$)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _fig = main()
    save_and_show_fig(_fig, "exact_large_z")
