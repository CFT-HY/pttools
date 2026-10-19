r"""Parameters of hydrodynamic lattice simulation runs."""

from dataclasses import dataclass


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
    #: The time step of the spectra of the figures of the article, e.g. the GW power growth rate of
    #: :correia_2025:`\ ` fig. 5
    fig_step: int
    #: The earlier time step of the GW power growth rate of the figures of the article,
    #: e.g. :correia_2025:`\ ` fig. 5
    fig_step_prev: int

    @property
    def label_latex(self) -> str:
        """Label of the run in LaTeX format."""
        return rf"{self.name.capitalize()}, $v_w={self.v_wall}, \alpha_n={self.alpha_n}$"

    @property
    def label_unicode(self) -> str:
        """Label of the run in Unicode format."""
        return f"{self.name.capitalize()}, vw={self.v_wall}, αₙ={self.alpha_n}"
