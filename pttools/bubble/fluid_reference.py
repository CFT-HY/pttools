import functools
import logging
import multiprocessing
import os.path
import typing as tp

import h5py
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from pttools.bubble import boundary
from pttools.bubble.alpha import alpha_n_max_bag
from pttools.bubble.fluid_bag import fluid_shell
from pttools.bubble.relativity import lorentz
from pttools.speedup.parallel import run_parallel

logger = logging.getLogger(__name__)


class FluidReference:
    def __init__(
            self,
            path: str,
            v_wall_min: float = 0.05,
            v_wall_max: float = 0.95,
            alpha_n_min: float = 0.05,
            alpha_n_max: float = 0.95,
            n_v_wall: int = 20,
            n_alpha_n: int = 20):
        self.path = path
        self.v_wall = np.linspace(v_wall_min, v_wall_max, n_v_wall, endpoint=True)
        self.alpha_n = np.linspace(alpha_n_min, alpha_n_max, n_alpha_n, endpoint=True)

        if not os.path.exists(path):
            self.create()

        self.data = np.empty((n_alpha_n, n_v_wall, 6))
        with h5py.File(path, "r") as file:
            self.data[:, :, 0] = file["vp"]
            self.data[:, :, 1] = file["vm"]
            self.data[:, :, 2] = file["vp_tilde"]
            self.data[:, :, 3] = file["vm_tilde"]
            self.data[:, :, 4] = file["wp"]
            self.data[:, :, 5] = file["wm"]

            self.interp = NearestNDInterpolator(x=file["coords"], y=file["inds"])

        if np.any(self.data < 0):
            raise ValueError

        self.vp = self.data[:, :, 0]
        self.vm = self.data[:, :, 1]
        self.vp_tilde = self.data[:, :, 2]
        self.vm_tilde = self.data[:, :, 3]
        self.wp = self.data[:, :, 4]
        self.wm = self.data[:, :, 5]

    def create(self):
        logger.info("Generating fluid reference")
        if os.path.exists(self.path):
            os.remove(self.path)
        try:
            with h5py.File(self.path, "w") as file:
                alpha_n_max = alpha_n_max_bag(self.v_wall)

                params = np.empty((self.alpha_n.size, self.v_wall.size, 3))
                params[:, :, 0], params[:, :, 1] = np.meshgrid(self.v_wall, self.alpha_n)
                params[:, :, 2], _ = np.meshgrid(alpha_n_max, self.alpha_n)

                vp, vm, vp_tilde, vm_tilde, wp, wm = run_parallel(
                    compute,
                    params,
                    multiple_params=True,
                    unpack_params=True,
                    output_dtypes=(np.float_, np.float_, np.float_, np.float_, np.float_, np.float_),
                    log_progress=True
                )
                file.create_dataset("vp", data=vp)
                file.create_dataset("vm", data=vm)
                file.create_dataset("vp_tilde", data=vp_tilde)
                file.create_dataset("vm_tilde", data=vm_tilde)
                file.create_dataset("wp", data=wp)
                file.create_dataset("wm", data=wm)
                # file.create_dataset("wn", data=wn)

                data = np.empty((self.alpha_n.size, self.v_wall.size, 6))
                data[:, :, 0] = vp
                data[:, :, 1] = vm
                data[:, :, 2] = vp_tilde
                data[:, :, 3] = vm_tilde
                data[:, :, 4] = wp
                data[:, :, 5] = wm

                # Nearest neighbour interpolator set-up
                valids = np.logical_not(np.any(np.isnan(data), axis=2))
                valid_count = np.sum(valids)
                coords = np.empty((valid_count, 2))
                inds = np.empty((valid_count,), dtype=np.uint)
                i = 0
                for i_alpha_n, alpha_n in enumerate(self.alpha_n):
                    for i_v_wall, v_wall in enumerate(self.v_wall):
                        if valids[i_alpha_n, i_v_wall]:
                            if np.any(np.isnan(data[i_alpha_n, i_v_wall, :])):
                                raise RuntimeError(
                                    "nan values should not be picked up for the nearest neighbour set-up"
                                )

                            coords[i, 0] = self.v_wall[i_v_wall]
                            coords[i, 1] = self.alpha_n[i_alpha_n]
                            inds[i] = i_alpha_n * self.v_wall.size + i_v_wall
                            i += 1

                file.create_dataset("coords", data=coords)
                file.create_dataset("inds", data=inds)
        except Exception as e:
            # Remove broken file
            os.remove(self.path)
            raise e
        logger.info("Fluid reference ready")

    def get(self, v_wall: float, alpha_n: float, allow_nan: bool = False) -> np.ndarray:
        if allow_nan:
            i_v_wall = (np.abs(self.v_wall - v_wall)).argmin()
            i_alpha_n = (np.abs(self.alpha_n - alpha_n)).argmin()
            return self.data[i_alpha_n, i_v_wall, :]

        ind = int(self.interp(v_wall, alpha_n))
        return self.data[ind // self.v_wall.size, ind % self.v_wall.size]


def compute(v_wall: float, alpha_n: float, alpha_n_max: float) -> tp.Tuple[float, float, float, float, float, float]:
    if alpha_n > alpha_n_max:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    v, w, xi = fluid_shell(v_wall, alpha_n)
    if np.any(np.isnan(v)) or np.any(np.isnan(w)) or np.any(np.isnan(xi)):
        logger.error("Got nan values from the integration at v_wall=%s, alpha_n=%s", v_wall, alpha_n)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    i_wall = np.argmax(v)
    i_wall_w = np.argmax(w)
    if i_wall != i_wall_w:
        raise ValueError("The wall is not at the same index in v and w")
    vw = xi[i_wall]
    if not np.isclose(vw, v_wall):
        raise ValueError(f"v_wall={v_wall}, computed v_wall={vw}")
    vp = v[i_wall]
    vm = v[i_wall-1]
    vp_tilde = lorentz(v_wall, vp)
    if np.isnan(vp_tilde) or vp_tilde < 0:
        raise ValueError(f"vp={vp}, vp_tilde={vp_tilde}")
    vm_tilde = lorentz(v_wall, vm)
    if np.isnan(vm_tilde) or vm_tilde < 0:
        raise ValueError(f"vm={vm}, vm_tilde={vm_tilde}")
    wp = w[i_wall]
    wm = w[i_wall-1]

    wn = w[-1]
    if not np.isclose(wn, 1):
        raise ValueError(f"The old solver should always have wn=1, got wn={wn}")

    dev = boundary.junction_condition_deviation1(vp_tilde, wp, vm_tilde, wm)
    if not np.isclose(dev, 0, atol=0.025):
        logger.warning(f"Deviation from boundary conditions: %s at v_wall=%s, alpha_n=%s", dev, v_wall, alpha_n)
        if not np.isclose(dev, 0, atol=0.025):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return vp, vm, vp_tilde, vm_tilde, wp, wm


@functools.cache
def ref():
    if multiprocessing.parent_process() is not None:
        raise RuntimeError(
            "The reference data should be loaded in the main process "
            "to ensure that each process doesn't have to load it separately. "
            "Call this function once before creating sub-processes.")
    return FluidReference(path=os.path.join(os.path.dirname(__file__), "fluid_reference.hdf5"))


if __name__ == "__main__":
    from pttools.logging import setup_logging
    setup_logging()
    ref()
