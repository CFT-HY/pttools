r"""Parameters of hydrodynamic lattice simulation runs."""

from dataclasses import dataclass

from pttools.ssm.nucleation import NucType

__all__ = ["LatticeRunParams"]


@dataclass(frozen=True, slots=True)
class LatticeRunParams:
    r"""Parameters and metadata of a simulation run, e.g. of :correia_2025:`\ ` or :cutting_2019:`\ `.

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
    #: :correia_2025:`\ ` fig. 5, or None if the article does not have one, see :py:attr:`fig_steps`
    fig_step: int | None = None
    #: The earlier time step of the GW power growth rate of the figures of the article,
    #: e.g. :correia_2025:`\ ` fig. 5, or None if the article does not have one, see :py:attr:`fig_steps`
    fig_step_prev: int | None = None
    #: The fraction of the final value of each wavenumber at which the fits of the extrapolation of the GW power
    #: start in the article, see :py:meth:`pttools.lattice.run.LatticeRun.pow_gw_asymptote`
    fit_start_fraction: float = 0.8
    #: The last time step of the fits of the extrapolation of the GW power in the article,
    #: or None if they use all the time steps, see :py:meth:`pttools.lattice.run.LatticeRun.omega_gw_asymptote`
    fit_step_end: int | None = None
    #: $N_b$, number of bubbles, or None to read it from the input file of the run
    n_bubbles: int | None = None
    #: $L$, side length of the lattice in lattice spacings, or None to read it from the input file of the run
    L: int | None = None
    #: $\delta x$, lattice spacing, or None to read it from the input file of the run
    dx: float | None = None
    #: Short label of the run for the plots, defaults to the capitalised name
    label: str | None = None

    @property
    def fig_steps(self) -> tuple[int, int]:
        """The time steps :py:attr:`fig_step` and :py:attr:`fig_step_prev`.

        :raises ValueError: if they are not given
        """
        if self.fig_step is None or self.fig_step_prev is None:
            raise ValueError(f"The time steps of the figures of the article are not given for the run {self.name}.")
        return self.fig_step, self.fig_step_prev

    @property
    def label_latex(self) -> str:
        """Label of the run in LaTeX format."""
        lattice = "" if self.L is None or self.dx is None else rf", $L={self.L}, \delta x={self.dx}$"
        return rf"{self._label}, $v_w={self.v_wall}, \alpha_n={self.alpha_n}${lattice}"

    @property
    def label_unicode(self) -> str:
        """Label of the run in Unicode format."""
        lattice = "" if self.L is None or self.dx is None else f", L={self.L}, δx={self.dx}"
        return f"{self._label}, vw={self.v_wall}, αₙ={self.alpha_n}{lattice}"

    @property
    def _label(self) -> str:
        return self.name.capitalize() if self.label is None else self.label
