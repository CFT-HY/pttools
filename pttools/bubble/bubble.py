import typing as tp

import numpy as np

from pttools.models.model import Model


class Bubble:
    """A solution of the hydrodynamic equations

    TODO: Work in progress

    (Defined in the meeting with Hindmarsh on 2022-07-06.)
    """
    def __init__(self, model: Model, tn: float, alpha_n: float, vw: float):
        self.model: Model = model

        self.xi: tp.Optional[np.ndarray] = None
        self.v: tp.Optional[np.ndarray] = None
        self.w: tp.Optional[np.ndarray] = None

        self.gw_power_spectrum = None
        self.notes: tp.List[str] = []

    def add_note(self, note: str):
        self.notes.append(note)

    def export(self):
        # TODO: export as e.g. JSON or HDF5
        pass

    def solve(self):
        raise NotImplementedError
