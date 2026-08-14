r"""
Finding alpha_n_min for ConstCSModel
====================================
"""

from pttools.models import ConstCSModel


def main() -> tuple[ConstCSModel, ConstCSModel]:
    model1 = ConstCSModel(css2=1/3, csb2=1/4, a_s=5, alpha_n_min=0.02)
    print(f"alpha_n_min={model1.alpha_n_min}, a_s={model1.a_s}, a_b={model1.a_b}, V_s={model1.V_s}, V_b={model1.V_b}")

    model2 = ConstCSModel(css2=1/3, csb2=1/4, a_s=2, a_b=1, V_s=0.1)
    print(f"alpha_n_min={model2.alpha_n_min}, alpha_n={model2.alpha_n(model2.w_crit)}")

    return model1, model2


if __name__ == "__main__":
    main()
