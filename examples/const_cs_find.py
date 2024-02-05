"""
Testing for finding $\alpha_{n,\text{min}}$ for ConstCSModel
============================================================
"""

from pttools.models import ConstCSModel


model = ConstCSModel(css2=1/3, csb2=1/4, a_s=5, alpha_n_min=0.02)
print(f"alpha_n_min={model.alpha_n_min}, a_s={model.a_s}, a_b={model.a_b}, V_s={model.V_s}, V_b={model.V_b}")

model2 = ConstCSModel(css2=1/3, csb2=1/4, a_s=2, a_b=1, V_s=0.1)
print(f"alpha_n_min={model.alpha_n_min}, alpha_n={model.alpha_n(model.w_crit)}")
