import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble, models


def compute_model(alpha_n: np.ndarray, model: models.Model, analytical: bool = True):
    v_cj = np.empty_like(alpha_n)
    for i in range(alpha_n.size):
        v_cj[i] = bubble.v_chapman_jouguet(alpha_n=alpha_n[i], model=model, analytical=analytical)
    return v_cj


def debug_plot(ax: plt.Axes, model: models.Model):
    # alpha_n
    ap = 0.1
    vp = bubble.v_chapman_jouguet_bag(ap)
    wp = model.w_n(ap)
    vm_test = bubble.v_minus(vp, ap, debug=True)
    print(f"ap={ap}, vp={vp}, wp={wp}, vm={vm_test}")
    # exit()
    vm = bubble.CS0
    wm_bag = bubble.wm_junction(vp, wp, vm)
    dev = bubble.wm_vw_solvable(np.array([wm_bag]), model, vp, wp)
    print(f"vp={vp}, wp={wp}, wm={wm_bag}, dev={dev}")
    wm_arr = np.linspace(0, 25, 20)
    dev_arr = np.zeros_like(wm_arr)
    for i, wm in enumerate(wm_arr):
        dev_arr[i] = bubble.wm_vw_solvable(np.array([wm]), model, vp, wp)
    ax.axhline(0)
    ax.axvline(wm_bag)
    ax.plot(wm_arr, dev_arr)
    ax.set_xlabel("wm")
    ax.set_ylabel("wm_vw_solvable")


def main():
    fig: plt.Figure = plt.figure()
    ax1, ax2 = fig.subplots(2)
    ax1: plt.Axes
    ax2: plt.Axes

    n_points = 100
    # Plot opacity
    alpha = 0.5
    # Transition strength
    # alpha_n = np.linspace(0.1, 0.3, n_points)
    alpha_n = np.linspace(0, 2, n_points)

    model_bag = models.BagModel(a_s=1.1, a_b=1, V_s=1)
    model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    model_const_cs = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=0.25, V_s=1)
    # print("w_n:", model_const_cs.w_n(alpha_n=0.1))
    # print("alpha_n:", model_const_cs.alpha_n(wn=1e6))

    # alpha_n_wn = 0.1
    # wn = np.linspace(30, 50, 100)
    # wn = np.linspace(0, 20, 100)
    # wn_sol_bag = bubble.gen_wn_solvable(model_bag, alpha_n_wn)
    # wn_sol_const_cs = bubble.gen_wn_solvable(model_const_cs, alpha_n_wn)
    # wn_sol_const_cs_like_bag = bubble.gen_wn_solvable(model_const_cs_like_bag, alpha_n_wn)
    # ax1.plot(wn, wn_sol_bag([wn]), label="Bag model", alpha=alpha)
    # ax1.plot(wn, wn_sol_const_cs([wn]), label="Constant $c_s$ model", alpha=alpha)
    # ax1.plot(wn, wn_sol_const_cs_like_bag([wn]), label="Constant $c_s$ model with bag coeff.", alpha=alpha)
    # ax1.axvline(model_bag.w_n(alpha_n_wn), ls=":")
    # ax1.axvline(model_const_cs_like_bag.w_n(alpha_n_wn), c="k", alpha=alpha)
    # ax1.axhline(0, c="k")
    # ax1.set_xlabel("$w_n$")
    # ax1.set_ylabel("solvable (look for zeroes)")
    # ax1.legend()

    # ax1.plot(wn, model_bag.alpha_n(wn), label="bag")
    # ax1.plot(wn, model_const_cs.alpha_n(wn), label="Const CS")
    # ax1.plot(wn, model_const_cs_like_bag.alpha_n(wn), label="Const CS (bag)")
    # ax1.axhline(alpha_n_wn, c="k")
    # ax1.legend()

    # alpha_n_test = 0.05
    # wm = np.linspace(0, 20, 50)
    # wn = model_bag.w_n(alpha_n_test)
    # print("wn:", wn)
    # bag_wm_vals = bubble.wm_solvable(np.array([wm]), model_bag, wn)
    # print(bag_wm_vals)
    # ax1.plot(wm, bag_wm_vals)
    # ax1.axhline(0)
    # ax1.set_yscale("log")
    # ax1.set_xscale("log")

    debug_plot(ax1, model_bag)

    v_cj_bag_analytical = compute_model(alpha_n, model_bag)
    v_cj_bag = compute_model(alpha_n, model_bag, analytical=False)
    v_cj_const_cs_like_bag = compute_model(alpha_n, model_const_cs_like_bag)
    v_cj_const_cs = compute_model(alpha_n, model_const_cs)

    ax2.plot(alpha_n, v_cj_bag_analytical, label="Bag model (analytical)", alpha=alpha)
    ax2.plot(alpha_n, v_cj_bag, label="Bag model", alpha=alpha)
    ax2.plot(alpha_n, v_cj_const_cs, label=r"Constant $c_s$ model", alpha=alpha)
    ax2.plot(alpha_n, v_cj_const_cs_like_bag, label=r"Constant $c_s$ model with bag coeff.", alpha=alpha)
    ax2.set_xlabel(r"$\alpha_n$")
    ax2.set_ylabel("$v_{CJ}$")
    ax2.legend()

    # Checking
    # ax2.axvline(1/3)
    # ax2.axhline(np.sqrt(3)/2)
    # ax2.set_xlim(0, 0.5)
    #
    # wn = model_const_cs.w_n(alpha_n)
    # wm = 4*wn
    # alpha_plus = model_const_cs.alpha_plus(wn, wm)
    # ax2.axvline(alpha_plus)

    fig.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()
