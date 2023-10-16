"""
ConstCSModel testing
====================

Simple testing for the constant sound speed model
"""

from pttools.models.const_cs import ConstCSModel


model = ConstCSModel(1/3, 1/4, V_s=5, a_s=10, a_b=5)
print(model.t_crit)
# print(model.alpha_n(552))
