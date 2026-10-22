r"""Parameters of hydrodynamic lattice simulation runs."""

from dataclasses import dataclass

from pttools.ssm.nucleation import NucType


@dataclass(frozen=True, slots=True)
class LatticeRunParams:
    r"""Parameters and metadata of a simulation run, e.g. of :correia_2025:`\ `.

    The time steps of the figures are not physically special.
    They are the times at which the spectra are compared with the Sound Shell Model in the figures of the article,
    e.g. in :correia_2025:`\ ` fig. 5, where they are after the maximum of the kinetic energy,
    at about one shock formation time.
    """

    #: Name of the run
    name: str
    #: Directory of the run relative to the data directory
    subdir: str
    #: ${v}_\text{wall}$, nominal wall speed
    v_wall: float
    #: $\alpha_n$, transition strength
    alpha_n: float
    #: Nucleation type of the bubbles
    nuc_type: NucType
    #: The time step of the spectra of the figures of the article, e.g. the GW power growth rate of
    #: :correia_2025:`\ ` fig. 5
    fig_step: int
    #: The earlier time step of the GW power growth rate of the figures of the article,
    #: e.g. :correia_2025:`\ ` fig. 5
    fig_step_prev: int
    #: The fraction of the final value of each wavenumber at which the fits of the extrapolation of the GW power
    #: start in the article, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    fit_start_fraction: float = 0.8
    #: The last time step of the fits of the extrapolation of the GW power in the article,
    #: or None if they use all the time steps, see :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
    fit_step_end: int | None = None

    @property
    def label_latex(self) -> str:
        """Label of the run in LaTeX format."""
        return rf"{self.name.capitalize()}, $v_w={self.v_wall}, \alpha_n={self.alpha_n}$"

    @property
    def label_unicode(self) -> str:
        """Label of the run in Unicode format."""
        return f"{self.name.capitalize()}, vw={self.v_wall}, αₙ={self.alpha_n}"
