from __future__ import absolute_import, division, print_function

import new_general_shooting_solution as gss
import EIKR_Toolbox as Eikr
import Bag_Toolbox as Bag
import matplotlib.pyplot as plt


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


# def plot_diff(true_xi, true_w, true_v, e_xi, e_w, e_v):

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


# def line_extend(xi, w, v):


true_bag_xi_1, true_bag_w_1, true_bag_v_1 = gss.plot_graph_module(0.4, Bag)
true_bag_xi_2, true_bag_w_2, true_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# true_bag_xi_3, true_bag_w_3, true_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# true_bag_xi_4, true_bag_w_4, true_bag_v_4 = gss.plot_graph_module(0.55, Bag)
# Bag.print_params()
# print('')
bagify()
print('')
# Bag.print_params()
e_bag_xi_1, e_bag_w_1, e_bag_v_1 = gss.plot_graph_module(0.4, Bag)
e_bag_xi_2, e_bag_w_2, e_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# e_bag_xi_3, e_bag_w_3, e_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# e_bag_xi_4, e_bag_w_4, e_bag_v_4 = gss.plot_graph_module(0.55, Bag)


plot_over(true_bag_xi_1, true_bag_w_1, true_bag_v_1, e_bag_xi_1, e_bag_w_1, e_bag_v_1, 1, 0.4)
plot_over(true_bag_xi_2, true_bag_w_2, true_bag_v_2, e_bag_xi_2, e_bag_w_2, e_bag_v_2, 2, 0.45)
plt.show()
