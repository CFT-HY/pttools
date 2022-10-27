from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel


def model_phase_label(model: BaseModel, phase: Phase) -> str:
    if phase == Phase.SYMMETRIC:
        phase_str = "s"
    elif phase == Phase.BROKEN:
        phase_str = "b"
    else:
        phase_str = f"{phase:.2f}"
    return rf"{model.label}, $\phi$={phase_str}"
