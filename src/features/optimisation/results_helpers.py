import pandas as pd
import numpy as np
from indago import Optimizer
from matplotlib import pyplot as plt
from src.features.typing import RunDict
from src.features.typing import MuscleFatigueRecovery


class SubIterationResult:
    evaluation_count = 0
    cost_fun_list = None

    def __init__(self):
        self.cost_fun_list = []
        self.evaluation_count = 0

    def add_sub_iteration_result(self, cost_function: float):
        self.evaluation_count += 1
        self.cost_fun_list.append(cost_function)

    def get_func_list(self):
        return self.cost_fun_list


def create_opt_var_df(run_dict: RunDict, columns: list) -> pd.DataFrame:
    opt_variable_vectors_df = pd.DataFrame(run_dict['X'], columns=columns)
    opt_variable_vectors_df.loc[:, 'cost_function'] = run_dict['f']
    return opt_variable_vectors_df


def create_run_results_df(
        run_count: int,
        init_mfr: MuscleFatigueRecovery,
        num_steps: int,
        num_eval: int,
        mfr_results: MuscleFatigueRecovery,
        fin_cost_fun: int
) -> pd.DataFrame:
    return pd.DataFrame({
        'run_count': run_count,
        'init_m0': init_mfr.M0,
        'init_alpha': init_mfr.alpha,
        'init_beta': init_mfr.beta,
        'init_f': init_mfr.F,
        'init_r': init_mfr.R,
        'num_steps': num_steps,
        'number_of_evaluations': num_eval,
        'M0': mfr_results.M0,
        'alpha': mfr_results.alpha,
        'beta': mfr_results.beta,
        'F': mfr_results.F,
        'R': mfr_results.R,
        'gamma': mfr_results.gamma,
        'final_cost_function': fin_cost_fun
    }, index=[run_count])


class IndagoResults:

    def __init__(self, x: np.array, fun: float, allvecs: list):
        self.x = x
        self.fun = fun
        self.allvecs = allvecs


def post_iteration_processing(it, candidates, best, iteration_dict: {}):
    if candidates[0] <= best:
        iteration_dict['X'].append(candidates[0].X)
        iteration_dict['f'].append(candidates[0].f)
    return


def save_run_history(opt: Optimizer, save_path: str):
    opt.plot_history()
    plt.savefig(save_path, dpi=300)
    return
