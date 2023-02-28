import os.path
import typing as tp

import h5py
import numpy as np

from pttools.bubble import boundary
from pttools.bubble.fluid_bag import fluid_shell
from pttools.bubble.relativity import lorentz
from pttools.speedup.parallel import run_parallel


def compute(v_wall: float, alpha_n: float) -> tp.Tuple[float, float, float, float, float, float]:
    v, w, xi = fluid_shell(v_wall, alpha_n)
    i_wall = np.argmax(v)
    i_wall_w = np.argmax(w)
    if i_wall != i_wall_w:
        raise ValueError("The wall is not at the same index in v and w")
    v_w = xi[i_wall]
    if not np.isclose(v_w, v_wall):
        raise ValueError(f"v_wall={v_wall}, computed v_wall={v_w}")
    v_p = v[i_wall]
    v_m = v[i_wall-1]
    # Todo: check that these conversions are correct
    v_p_tilde = lorentz(v_wall, v_p)
    if v_p_tilde < v_p or v_p_tilde < 0:
        raise ValueError(f"v_p={v_p}, v_p_tilde={v_p_tilde}")
    v_m_tilde = -lorentz(-v_wall, v_m)
    if v_m_tilde < 0:
        raise ValueError(f"v_m={v_m}, v_m_tilde={v_m_tilde}")
    w_p = w[i_wall]
    w_m = w[i_wall-1]

    dev = boundary.junction_condition_deviation1(v_p_tilde, w_p, v_m_tilde, w_m)
    if not np.isclose(dev, 0):
        print(dev)
    return v_p, v_m, v_p_tilde, v_m_tilde, w_p, w_m


class FluidReference:
    def __init__(
            self,
            path: str,
            v_wall_min: float = 0.05,
            v_wall_max: float = 0.95,
            alpha_n_min: float = 0.05,
            alpha_n_max: float = 0.05,
            n_v_wall: int = 20,
            n_alpha_n: int = 20):
        self.path = path
        self.v_wall = np.linspace(v_wall_min, v_wall_max, n_v_wall, endpoint=True)
        self.alpha_n = np.linspace(alpha_n_min, alpha_n_max, n_alpha_n, endpoint=True)

        if not os.path.exists(path):
            self.create()

        file = h5py.File(path, "r")
        self.v_p = file["v_p"]
        self.v_m = file["v_m"]
        self.v_p_tilde = file["v_p_tilde"]
        self.v_m_tilde = file["v_m_tilde"]
        self.w_p = file["w_p"]
        self.w_m = file["w_m"]

    def create(self):
        file = h5py.File(self.path, "w")

        params = np.empty((self.v_wall.size, self.alpha_n.size, 2))
        params[:, :, 0], params[:, :, 1] = np.meshgrid(self.v_wall, self.alpha_n)

        v_p, v_m, v_p_tilde, v_m_tilde, w_p, w_m = run_parallel(
            compute,
            params,
            multiple_params=True,
            unpack_params=True,
            output_dtypes=(np.float_, np.float_, np.float_, np.float_, np.float_, np.float_),
            log_progress=True
        )
        file.create_dataset("v_p", data=v_p)
        file.create_dataset("v_m", data=v_m)
        file.create_dataset("v_p_tilde", data=v_p_tilde)
        file.create_dataset("v_m_tilde", data=v_m_tilde)
        file.create_dataset("w_p", data=w_p)
        file.create_dataset("w_m", data=w_m)
        file.close()


ref = FluidReference(path=os.path.join(os.path.dirname(__file__), "fluid_reference.hdf5"))
