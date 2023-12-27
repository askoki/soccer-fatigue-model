import matplotlib.pyplot as plt
import scienceplots


def load_plt_style(is_default=False) -> None:
    if not is_default:
        plt.style.use(['science', 'grid'])
    else:
        plt.style.use('default')