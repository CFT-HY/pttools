"""Repositories of projects that use PTtools, such as PTtools itself and PTPlot."""

import dataclasses
from pathlib import Path
import re

__all__ = ["Repo"]


@dataclasses.dataclass(frozen=True, slots=True)
class Repo:
    """A repository of a project, and the paths of its standard directories."""

    #: The root directory of the repository
    path: Path

    @property
    def docs_dir(self) -> Path:
        """The documentation directory."""
        return self.path / "docs"

    @property
    def fig_dir_env(self) -> str:
        """The environment variable with which the figure directory of the examples is set.

        It's ``NAME_FIG_DIR``, where ``NAME`` is the name of the repository in upper case,
        e.g. ``PTTOOLS_FIG_DIR`` or ``PTPLOT_FIG_DIR``. See ``examples/utils.py`` of the repository.
        """
        return re.sub(r"\W", "_", self.name).upper() + "_FIG_DIR"

    @property
    def log_dir(self) -> Path:
        """The directory of the logs."""
        return self.path / "logs"

    @property
    def name(self) -> str:
        """The name of the repository directory."""
        return self.path.name
