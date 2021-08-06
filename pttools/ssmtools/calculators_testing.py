import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import sympy as sp
# import time

# from pttools import speedup
import pttools.type_hints as th
from . import const
from . import calculators


# @profile
def gen_piecewise(x, points: np.ndarray):
    funcs = []
    lims = []
    for p1, p2 in zip(points[:-1], points[1:]):
        # Linear
        fit = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], deg=1)
        lims.append((float(p1[0]) < x) & (x < float(p2[0])))
        funcs.append(fit[0]*x + fit[1])
    lims.append(True)
    funcs.append(0)

    args = list(zip(funcs, lims))
    # print(args)
    return sp.Piecewise(*args)


def sin_transform(
        z: th.FloatOrArr, xi: np.ndarray, f: np.ndarray, z_st_thresh: float = const.Z_ST_THRESH) -> th.FloatOrArr:

    # Ensure that xi is monotonically increasing
    if np.any(np.diff(xi) <= 0):
        raise ValueError

    if np.any(np.diff(z) <= 0):
        raise ValueError

    sin_transform_debug(z, xi, f, z_st_thresh)

    # start_time = time.perf_counter()
    integral = calculators.sin_transform(z, xi, f, z_st_thresh)
    # end_time = time.perf_counter()
    # print("Numeric:", end_time - start_time)
    return integral

    # if np.max(xi) > 1 or np.min(xi) < 0:
    #     raise ValueError

    # f_even = np.interp()
    # arr = scipy.fft.dst(f)
    # # print(f.shape, z.shape, arr.shape)
    # x = np.linspace(0, 1, num=arr.size+1)[1:]
    #
    # integral = np.zeros_like(z)
    # for i, z_val in enumerate(z):
    #     arr_xi = np.interp(xi, x, arr)
    #     integral[i] = np.trapz(arr_xi, xi * z_val)
    #
    # integral2 = sin_transform2(z, xi, f, z_st_thresh)
    # print("Test")
    # print("xi", xi)
    # print("f", f)
    # print("z", z)
    # print("I", integral)
    # print("I2", integral2)
    # return integral2


def sin_transform_debug(z: th.FloatOrArr, xi: np.ndarray, f: np.ndarray, z_st_thresh: float = const.Z_ST_THRESH):
    fig: plt.Figure
    axs: np.ndarray
    fig, axs = plt.subplots(2, 3, figsize=(11.7, 8.3))
    ax1: plt.Axes = axs[0, 0]
    ax2: plt.Axes = axs[0, 1]
    ax3: plt.Axes = axs[1, 0]
    ax4: plt.Axes = axs[1, 1]
    ax5: plt.Axes = axs[1, 2]

    # f
    ax1.plot(xi, f, label="f")
    ax1.set_xlabel("xi")
    ax1.set_ylabel("f")
    ax1.set_title("f")

    # z
    dz_blend = const.DZ_ST_BLEND
    ax2.plot(z)
    ax2.set_yscale("log")
    ax2.set_ylabel("z")
    ax2.axhline(z_st_thresh, label=f"z_st_tresh ({z_st_thresh})", color="g")
    ax2.axhline(z_st_thresh - dz_blend, label=f"z_st_thresh - dz_blend ({z_st_thresh - dz_blend})", color="r")
    ax2.legend()
    ax2.set_title("z")

    # Old sine transform
    integrand = f * np.sin(np.outer(z, xi))
    integral = np.trapz(integrand, xi)
    ax3.plot(z, np.abs(integral)**2)
    ax3.set_xscale("log")
    # ax3.set_yscale("log")
    ax3.set_title("Old DST")
    ax3.set_xlabel("z")
    ax3.set_ylabel("integral")

    # New sine transform
    x_uniform = np.linspace(0, 1, num=10000)
    f_uniform = np.interp(x=x_uniform, xp=xi, fp=f)
    ax1.plot(x_uniform, np.abs(f_uniform)**2, label="f_uniform")
    repeats = 100
    f_uniform_repeated = np.repeat(f_uniform, repeats)
    arr2 = scipy.fft.dst(f_uniform_repeated)
    # z_uniform = np.linspace()
    ax4.plot(np.linspace(start=0, stop=arr2.size/repeats, num=arr2.size), arr2)
    ax4.set_xscale("log")
    # ax4.set_yscale("log")
    ax4.set_title("SciPy")
    ax4.set_ylabel("integral")

    # Symbolic
    # start_time = time.perf_counter()
    inds = np.array([0, 1, -3, -2, -1])
    points = np.array([xi[inds], f[inds]]).T
    x_sym, z_sym = sp.symbols("x z")
    f_sym = gen_piecewise(x_sym, points)
    f_lam = sp.lambdify(x_sym, f_sym, "numpy")

    trans = sp.integrals.transforms.sine_transform(f_sym, x_sym, z_sym)
    trans_lam = sp.lambdify(z_sym, trans, "numpy")
    # end_time = time.perf_counter()
    # print("Symbolic:", end_time - start_time)

    ax1.plot(points[:, 0], points[:, 1], label="piecewise")
    ax1.plot(xi, f_lam(xi))

    ax5.plot(z, np.abs(trans_lam(z))**2)
    ax5.set_title("SymPy")
    ax5.set_xscale("log")
    ax5.set_xlabel("z")
    ax5.set_ylabel("integral")

    ax1.legend()
    plt.show()
