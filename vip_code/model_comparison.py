from __future__ import absolute_import, division, print_function

import new_general_shooting_solution as gss
import EIKR_Toolbox as Eikr
import Bag_Toolbox as Bag
import matplotlib.pyplot as plt
import numpy as np


def bagify():
    eps_plus = Eikr.epsilon(Eikr.Tn, 0)
    eps_minus = Eikr.epsilon(Eikr.Tn)
    a_plus = Eikr.a(Eikr.Tn, 0)
    a_minus = Eikr.a(Eikr.Tn)
    Bag.set_params('aplus', a_plus)
    Bag.set_params('aminus', a_minus)
    Bag.set_params('epsilonplus', eps_plus)
    Bag.set_params('epsilonminus', eps_minus)
    return


def plot_diff(xi, true_w, true_v, e_w, e_v, j, xi_wall):
    diff_w = true_w - e_w
    diff_v = true_v - e_v
    plt.figure(2*j)
    plt.plot(xi, diff_w)
    plt.title(r'$\xi_w=${}'.format(xi_wall))
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$w_{true}-w_{eikr}$')
    plt.figure(2*j+1)
    plt.plot(xi, diff_v)
    plt.title(r'$\xi_w=${}'.format(xi_wall))
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$v_{true}-v_{eikr}$')
    return


def plot_over(true_xi, true_w, true_v, e_xi, e_w, e_v, j, xi_wall):
    plt.figure(2*j)
    # print(max(w_plot))
    if max(true_w) > max(e_w):
        plt.axis([0, 1, 0, max(true_w)])
    else:
        plt.axis([0, 1, 0, max(e_w)])
    plt.title(r'$\xi_w=${}'.format(xi_wall))
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$w(\xi)$')
    plt.plot(true_xi, true_w, color='b', label='True Bag')
    plt.plot(e_xi, e_w, color='g', label='Bagified EIKR')
    plt.legend()

    plt.figure(2*j+1)
    plt.title(r'$\xi_w=${}'.format(xi_wall))
    plt.axis([0, 1, 0, 1])
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$v(\xi)$')
    plt.plot(true_xi, true_v, color='b', label='True Bag')
    plt.plot(e_xi, e_v, color='g', label='Bagified EIKR')
    plt.legend()
    return


def merge(true_xi, true_w, true_v, e_xi, e_w, e_v):
    # Since the xi values of the two models do not line up, this function interpolates the data sets to make 'complete'
    # ones which can be more easily compared
    full_xi = np.append(true_xi, e_xi)
    full_xi = np.sort(full_xi)
    # print(full_xi)
    full_true_w = np.interp(full_xi, true_xi[::-1], true_w[::-1])
    full_true_v = np.interp(full_xi, true_xi[::-1], true_v[::-1])
    full_e_w = np.interp(full_xi, e_xi[::-1], e_w[::-1])
    full_e_v = np.interp(full_xi, e_xi[::-1], e_v[::-1])
    return full_xi, full_true_w, full_true_v, full_e_w, full_e_v


Bag.print_params()
true_bag_xi_1, true_bag_w_1, true_bag_v_1 = gss.plot_graph_module(0.4, Bag)
# true_bag_xi_2, true_bag_w_2, true_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# true_bag_xi_3, true_bag_w_3, true_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# true_bag_xi_4, true_bag_w_4, true_bag_v_4 = gss.plot_graph_module(0.55, Bag)
# Bag.print_params()
# print('')
bagify()
print('')
Bag.print_params()
# Bag.print_params()
e_bag_xi_1, e_bag_w_1, e_bag_v_1 = gss.plot_graph_module(0.4, Bag)
# e_bag_xi_2, e_bag_w_2, e_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# e_bag_xi_3, e_bag_w_3, e_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# e_bag_xi_4, e_bag_w_4, e_bag_v_4 = gss.plot_graph_module(0.55, Bag)


plot_over(true_bag_xi_1, true_bag_w_1, true_bag_v_1, e_bag_xi_1, e_bag_w_1, e_bag_v_1, 1, 0.4)
m_xi_1, m_true_w_1, m_true_v_1, m_e_w_1, m_e_v_1 = merge(true_bag_xi_1, true_bag_w_1, true_bag_v_1, e_bag_xi_1,
                                                         e_bag_w_1, e_bag_v_1)
plot_diff(m_xi_1, m_true_w_1, m_true_v_1, m_e_w_1, m_e_v_1, 10, 0.4)
plt.show()

# plot_over(true_bag_xi_2, true_bag_w_2, true_bag_v_2, e_bag_xi_2, e_bag_w_2, e_bag_v_2, 2, 0.45)
