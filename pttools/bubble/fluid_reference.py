import functools
import logging
import multiprocessing
import os.path
import time
import typing as tp

import h5py
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from pttools.bubble import boundary
from pttools.bubble.alpha import alpha_n_max_bag
from pttools.bubble.boundary import SolutionType
from pttools.bubble.fluid_bag import sound_shell_bag
from pttools.bubble import props
from pttools.bubble import transition
from pttools.speedup import IS_OSX

from pttools.speedup.parallel import run_parallel

logger = logging.getLogger(__name__)


class FluidReference:
    def __init__(
            self,
            path: str,
            v_wall_min: float = 0.05,
            v_wall_max: float = 0.95,
            alpha_n_min: float = 0.01,
            alpha_n_max: float = 0.99,
            n_v_wall: int = 100,
            n_alpha_n: int = 100):
        self.path = path

        if not os.path.exists(path):
            self.create(v_wall_min, v_wall_max, alpha_n_min, alpha_n_max, n_v_wall, n_alpha_n)

        try:
            file = h5py.File(path, "r")
        except (KeyError, OSError) as e:
            logger.exception(
                "Could not open the fluid reference file at \"%s\". Generating a new one.",
                path, exc_info=e
            )
            os.remove(self.path)
            self.create(v_wall_min, v_wall_max, alpha_n_min, alpha_n_max, n_v_wall, n_alpha_n)
            file = h5py.File(path, "r")

        self.v_wall = file["v_wall"][...]
        self.alpha_n = file["alpha_n"][...]
        self.data = np.empty((self.alpha_n.size, self.v_wall.size, 6))
        self.data[:, :, 0] = file["vp"]
        self.data[:, :, 1] = file["vm"]
        self.data[:, :, 2] = file["vp_tilde"]
        self.data[:, :, 3] = file["vm_tilde"]
        self.data[:, :, 4] = file["wp"]
        self.data[:, :, 5] = file["wm"]

        self.interp_sub_def = NearestNDInterpolator(x=file["coords_sub_def"][...], y=file["inds_sub_def"][...])
        self.interp_hybrid = NearestNDInterpolator(x=file["coords_hybrid"][...], y=file["inds_hybrid"][...])
        self.interp_detonation = NearestNDInterpolator(x=file["coords_detonation"][...], y=file["inds_detonation"][...])
        file.close()

        if np.any(self.data < 0):
            raise ValueError

        self.vp = self.data[:, :, 0]
        self.vm = self.data[:, :, 1]
        self.vp_tilde = self.data[:, :, 2]
        self.vm_tilde = self.data[:, :, 3]
        self.wp = self.data[:, :, 4]
        self.wm = self.data[:, :, 5]

        # There is no need to add the PID number here, as that is done automatically by the logging system.
        logger.info("Loaded fluid reference with n_alpha_n=%s, n_v_wall=%s", self.data.shape[0], self.data.shape[1])

    def create(
            self,
            v_wall_min: float, v_wall_max: float,
            alpha_n_min: float, alpha_n_max: float,
            n_v_wall: int, n_alpha_n: int):
        logger.info("Generating fluid reference")
        start_time = time.perf_counter()
        if os.path.exists(self.path):
            os.remove(self.path)
        try:
            with h5py.File(self.path, "w") as file:
                v_walls = np.linspace(v_wall_min, v_wall_max, n_v_wall, endpoint=True)
                alpha_ns = np.linspace(alpha_n_min, alpha_n_max, n_alpha_n, endpoint=True)
                alpha_n_max = alpha_n_max_bag(v_walls)

                params = np.empty((alpha_ns.size, v_walls.size, 3))
                params[:, :, 0], params[:, :, 1] = np.meshgrid(v_walls, alpha_ns)
                params[:, :, 2], _ = np.meshgrid(alpha_n_max, alpha_ns)

                sol_type, vp, vm, vp_tilde, vm_tilde, wp, wm = run_parallel(
                    compute,
                    params,
                    multiple_params=True,
                    unpack_params=True,
                    output_dtypes=(np.int_, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64),
                    log_progress_percentage=10
                )
                file.create_dataset("v_wall", data=v_walls)
                file.create_dataset("alpha_n", data=alpha_ns)
                file.create_dataset("vp", data=vp)
                file.create_dataset("vm", data=vm)
                file.create_dataset("vp_tilde", data=vp_tilde)
                file.create_dataset("vm_tilde", data=vm_tilde)
                file.create_dataset("wp", data=wp)
                file.create_dataset("wm", data=wm)
                # file.create_dataset("wn", data=wn)

                data = np.empty((alpha_ns.size, v_walls.size, 6))
                data[:, :, 0] = vp
                data[:, :, 1] = vm
                data[:, :, 2] = vp_tilde
                data[:, :, 3] = vm_tilde
                data[:, :, 4] = wp
                data[:, :, 5] = wm

                # Nearest neighbour interpolator set-up
                valids = np.logical_not(np.any(np.isnan(data), axis=2))
                coords = [[], [], []]
                inds = [[], [], []]
                i = 0
                for i_alpha_n, alpha_n in enumerate(alpha_ns):
                    for i_v_wall, v_wall in enumerate(v_walls):
                        if valids[i_alpha_n, i_v_wall]:
                            if np.any(np.isnan(data[i_alpha_n, i_v_wall, :])):
                                raise RuntimeError(
                                    "nan values should not be picked up for the nearest neighbour set-up"
                                )
                            sol_tp = sol_type[i_alpha_n, i_v_wall]
                            coords[sol_tp].append([v_walls[i_v_wall], alpha_ns[i_alpha_n]])
                            inds[sol_tp].append(i_alpha_n * v_walls.size + i_v_wall)
                            i += 1

                file.create_dataset("coords_sub_def", data=np.array(coords[0], dtype=np.float64))
                file.create_dataset("coords_hybrid", data=np.array(coords[1], dtype=np.float64))
                file.create_dataset("coords_detonation", data=np.array(coords[2], dtype=np.float64))
                file.create_dataset("inds_sub_def", data=np.array(inds[0], dtype=np.int_))
                file.create_dataset("inds_hybrid", data=np.array(inds[1], dtype=np.int_))
                file.create_dataset("inds_detonation", data=np.array(inds[2], dtype=np.int_))
        except Exception as e:
            # Remove broken file
            os.remove(self.path)
            raise e
        logger.info("Fluid reference ready, took: %s s", time.perf_counter() - start_time)

    def get(self, v_wall: float, alpha_n: float, sol_type: SolutionType) -> np.ndarray:
        if sol_type == SolutionType.SUB_DEF:
            ind = int(self.interp_sub_def(v_wall, alpha_n))
        elif sol_type == SolutionType.HYBRID:
            ind = int(self.interp_hybrid(v_wall, alpha_n))
        elif sol_type == SolutionType.DETON:
            ind = int(self.interp_detonation(v_wall, alpha_n))
        else:
            raise ValueError(f"Invalid solution type: {sol_type}")
        i_alpha_n = ind // self.v_wall.size
        i_v_wall = ind % self.v_wall.size
        return self.data[i_alpha_n, i_v_wall]


def compute(v_wall: float, alpha_n: float, alpha_n_max: float) -> tp.Tuple[int, float, float, float, float, float, float]:
    if alpha_n > alpha_n_max:
        return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    v, w, xi = sound_shell_bag(v_wall, alpha_n)
    sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)

    if np.any(np.isnan(v)) or np.any(np.isnan(w)) or np.any(np.isnan(xi)):
        logger.error("Got nan values from the integration at v_wall=%s, alpha_n=%s", v_wall, alpha_n)
        return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    vp, vm, vp_tilde, vm_tilde, wp, wm, wn, wm_sh = props.v_and_w_from_solution(v, w, xi, v_wall, sol_type)

    if not np.isclose(wn, 1):
        raise ValueError(f"The old solver should always have wn=1, got wn={wn}")

    dev = boundary.junction_condition_deviation1(vp_tilde, wp, vm_tilde, wm)
    if not np.isclose(dev, 0, atol=0.025):
        logger.warning(f"Deviation from boundary conditions: %s at v_wall=%s, alpha_n=%s", dev, v_wall, alpha_n)
        if not np.isclose(dev, 0, atol=0.025):
            return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    sol_type_int = -1
    if sol_type == SolutionType.SUB_DEF:
        sol_type_int = 0
    elif sol_type == SolutionType.HYBRID:
        sol_type_int = 1
    elif sol_type == SolutionType.DETON:
        sol_type_int = 2

    return sol_type_int, vp, vm, vp_tilde, vm_tilde, wp, wm


@functools.cache
def ref():
    if multiprocessing.parent_process() is not None:
        msg = \
            "The reference data was attempted to be loaded in a subprocess. " \
            "The reference data should be loaded in the main process before creating subprocesses " \
            "to ensure that each process doesn't have to load it separately. " \
            "Call this function once before creating subprocesses."
        if not IS_OSX:
            raise RuntimeError(msg)
        logger.warning(msg + " (This may not work on macOS.)")
    return FluidReference(path=os.path.join(os.path.dirname(__file__), "fluid_reference.hdf5"))


if __name__ == "__main__":
    from pttools.logging import setup_logging
    setup_logging()
    ref()
