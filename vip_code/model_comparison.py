from __future__ import absolute_import, division, print_function

import new_general_shooting_solution as gss
import EIKR_Toolbox as Eikr
import Bag_Toolbox as Bag


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


#def plot_diff(true_xi, true_w, true_v, e_xi, e_w, e_v):



true_bag_xi_1, true_bag_w_1, true_bag_v_1 = gss.plot_graph_module(0.4, Bag)
# true_bag_xi_2, true_bag_w_2, true_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# true_bag_xi_3, true_bag_w_3, true_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# true_bag_xi_4, true_bag_w_4, true_bag_v_4 = gss.plot_graph_module(0.55, Bag)
# Bag.print_params()
# print('')
bagify()
# Bag.print_params()
e_bag_xi_1, e_bag_w_1, e_bag_v_1 = gss.plot_graph_module(0.4, Bag)
# e_bag_xi_2, e_bag_w_2, e_bag_v_2 = gss.plot_graph_module(0.45, Bag)
# e_bag_xi_3, e_bag_w_3, e_bag_v_3 = gss.plot_graph_module(0.5, Bag)
# e_bag_xi_4, e_bag_w_4, e_bag_v_4 = gss.plot_graph_module(0.55, Bag)

print('number of bag xi points', len(true_bag_xi_1))
print('number of bagified eikr xi points', len(e_bag_xi_1))


