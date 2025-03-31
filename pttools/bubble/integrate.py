import logging
import typing as tp

import numba
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from pttools import speedup
from pttools.bubble.boundary import Phase
from pttools.speedup.numba_wrapper import numbalsoda
from pttools.speedup.options import NUMBA_DISABLE_JIT
from . import bag
from . import const


logger = logging.getLogger(__name__)

DEFAULT_DF_DTAU: str = "bag"
# ODEINT_LOCK = threading.Lock()

#: Cache for the differential equations.
#: New differential equations have to be added here before usage so that they can be found by
#: :func:`scipy.integrate.odeint` and :func:`scipy.integrate.solve_ivp`.
differentials = speedup.DifferentialCache()


def add_df_dtau(name: str, cs2_fun: th.CS2Fun) -> speedup.DifferentialPointer:
    """Add a new differential equation to the cache based on the given sound speed function.

    :param name: the name of the function
    :param cs2_fun: function, which gives the speed of sound squared $c_s^2$.
    :return:
    """
    func = gen_df_dtau(cs2_fun)
    return differentials.add(name, func)


def gen_df_dtau(cs2_fun: th.CS2Fun) -> speedup.DifferentialCFunc:
    r"""Generate a function for the differentials of fluid variables $(v, w, \xi)$ in parametric form.
    The parametrised differential equation is as in :gw_pt_ssm:`\ ` eq. B.14-16:

    - $\frac{dv}{dt} = 2v c_s^2 (1-v^2) (1 - \xi v)$
    - $\frac{dw}{dt} = \frac{w}{1-v^2} \frac{\xi - v}{1 - \xi v} (\frac{1}{c_s^2}+1) \frac{dv}{dt}$
    - $\frac{d\xi}{dt} = \xi \left( (\xi - v)^2 - c_s^2 (1 - \xi v)^2 \right)$

    :param cs2_fun: function, which gives the speed of sound squared $c_s^2$.
    :return: function for the differential equation
    """
    cs2_fun_numba = cs2_fun \
        if isinstance(cs2_fun, (speedup.CFunc, speedup.Dispatcher)) or NUMBA_DISABLE_JIT \
        else numba.cfunc("float64(float64, float64)")(cs2_fun)

    def df_dtau(t: float, u: np.ndarray, du: np.ndarray, args: np.ndarray = None) -> None:
        r"""Computes the differentials of the variables $(v, w, \xi)$ for a given $c_s^2$ function

        :param t: "time"
        :param u: point
        :param du: derivatives
        :param args: extra arguments: [phase]
        :return: $\frac{dv}{d\tau}, \frac{dw}{d\tau}, \frac{d\xi}{d\tau}$
        """
        v = u[0]
        w = u[1]
        xi = u[2]
        phase = args[0]
        cs2 = cs2_fun_numba(w, phase)
        xiXv = xi * v
        xi_v = xi - v
        v2 = v * v

        du[0] = 2 * v * cs2 * (1 - v2) * (1 - xiXv)  # dv/dt
        du[1] = (w / (1 - v2)) * (xi_v / (1 - xiXv)) * (1 / cs2 + 1) * du[0]  # dw_dt
        du[2] = xi * (xi_v ** 2 - cs2 * (1 - xiXv) ** 2)  # dxi/dt
    return df_dtau


#: Pointer to the differential equation of the bag model
DF_DTAU_BAG_PTR = add_df_dtau("bag", bag.cs2_bag_scalar_cfunc if speedup.NUMBA_DISABLE_JIT else bag.cs2_bag)


@numba.njit
def fluid_integrate_param(
        v0: float,
        w0: float,
        xi0: float,
        phase: float = -1.,
        t_end: float = const.T_END_DEFAULT,
        n_xi: int = const.N_XI_DEFAULT,
        df_dtau_ptr: speedup.DifferentialPointer = DF_DTAU_BAG_PTR,
        method: str = "odeint") -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Integrates parametric fluid equations in df_dtau from an initial condition.
    Positive t_end integrates along curves from $(v,w) = (0,c_{s,0})$ to $(1,1)$.
    Negative t_end integrates towards $(0,c_s{s,0})$.

    :param v0: $v_0$
    :param w0: $w_0$
    :param xi0: $\xi_0$
    :param phase: phase $\phi$
    :param t_end: $t_\text{end}$
    :param n_xi: number of $\xi$ points
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :return: $v, w, \xi, t$
    """
    if phase < 0.:
        print("The phase has not been set! Assuming symmetric phase.")
        phase = 0.

    t = np.linspace(0., t_end, n_xi)
    y0 = np.array([v0, w0, xi0])
    # The second value ensures that the Numba typing is correct.
    data = np.array([phase, 0.])
    if method == "numba_lsoda" or speedup.NUMBA_INTEGRATE:
        if numbalsoda is None:
            raise ImportError("NumbaLSODA is not loaded")
        v, w, xi, success = fluid_integrate_param_numba(t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr)

    # This lock prevents a SystemError when running multiple threads
    # with ODEINT_LOCK:

    # SciPy differential equation solvers are not supported by Numba.
    # Putting these within numba.objmode can also be challenging, as function-type arguments are not supported.
    # For better performance, the "df_dtau" should be already fully Numba-compiled at this point instead
    # of taking functions as its arguments.
    else:
        with numba.objmode(v="float64[:]", w="float64[:]", xi="float64[:]", success="boolean"):
            if method == "odeint":
                v, w, xi, success = fluid_integrate_param_odeint(t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr)
            else:
                v, w, xi, success = fluid_integrate_param_solve_ivp(
                    t=t, y0=y0, data=data, df_dtau_ptr=df_dtau_ptr, method=method)
    if not success:
        raise RuntimeError("Integration failed")
    return v, w, xi, t


@numba.njit
def fluid_integrate_param_numba(t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    r"""Integrate a differential equation using NumbaLSODA.

    :param t: time
    :param y0: starting point
    :param data: constants
    :param df_dtau_ptr: pointer to the differential equation function
    :return: $v, w, \xi$, success status
    """
    if speedup.NUMBA_DISABLE_JIT:
        raise NotImplementedError("NumbaLSODA is supported only when jitting is enabled")

    backwards = t[-1] < 0
    t_numba = -t if backwards else t
    data_numba = np.zeros((data.size + 1))
    data_numba[:-1] = data
    # Numba does not support float(bool)
    data_numba[-1] = int(backwards)
    usol, success = numbalsoda.lsoda(df_dtau_ptr, u0=y0, t_eval=t_numba, data=data_numba)
    if not success:
        with numba.objmode:
            logger.error(f"NumbaLSODA failed for %s integration", "backwards" if backwards else "forwards")
    v = usol[:, 0]
    w = usol[:, 1]
    xi = usol[:, 2]
    return v, w, xi, success


def fluid_integrate_param_odeint(t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer):
    r"""Integrate a differential equation using :func:`scipy.integrate.odeint`.

    :param t: time
    :param y0: starting point
    :param df_dtau_ptr: pointer to the differential equation function, which is already in the cache
    :return: $v, w, \xi$, success status
    """
    try:
        func = differentials.get_odeint(df_dtau_ptr)
        # This function call that does nothing seems to be necessary to avoid a segfault within SciPy.
        # Probably it has something to do with the lifetime of the function object.
        func(y0, t[0], data)
        soln: np.ndarray = spi.odeint(func, y0=y0, t=t, args=(data,))
        v = soln[:, 0]
        w = soln[:, 1]
        xi = soln[:, 2]
        success = True
    except Exception as e:
        logger.exception("Integrating fluid shell with odeint failed", exc_info=e)
        v = w = xi = np.zeros_like(t)
        success = False
    return v, w, xi, success


def fluid_integrate_param_solve_ivp(
        t: np.ndarray, y0: np.ndarray, data: np.ndarray, df_dtau_ptr: speedup.DifferentialPointer, method: str):
    """Integrate a differential equation using :func:`scipy.integrate.solve_ivp`.

    :param t: time
    :param y0: starting point
    :param df_dtau_ptr: pointer to the differential equation function, which is already in the cache
    :param method: name of the integrator to be used. See the :func:`scipy.integrate.solve_ivp` documentation.
    """
    try:
        func = differentials.get_solve_ivp(df_dtau_ptr)
        soln: spi._ivp.ivp.OdeResult = spi.solve_ivp(
            func, t_span=(t[0], t[-1]), y0=y0, method=method, t_eval=t, args=(data,))
        v = soln.y[0, :]
        w = soln.y[1, :]
        xi = soln.y[2, :]
        success = True
    except Exception as e:
        logger.exception("Integrating fluid shell with solve_ivp failed", exc_info=e)
        v = w = xi = np.zeros_like(t)
        success = False
    return v, w, xi, success


def precompile():
    fluid_integrate_param(v0=0.5, w0=0.5, xi0=0.5, phase=0., n_xi=2)
