import os
import sys

import pandas as pd
from sklearn.metrics import r2_score

from settings import REPORTS_DIR, TABLES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import set_save_path_arg_parser
from src.reports.helpers import format_number

folder_path = set_save_path_arg_parser()
load_name = 'pso'

sprint_data_holder = DataHolder()

create_dir(TABLES_DIR)
all_runs_df = pd.read_csv(os.path.join(REPORTS_DIR, load_name, folder_path, f'{load_name}_all_runs_df.csv'))
one_player = [sprint_data_holder.get_players()[0]]
all_runs_df = all_runs_df[all_runs_df.player.isin(one_player)]
all_runs_df.reset_index(drop=True, inplace=True)

one_player_results_df = pd.DataFrame()
for idx, best_row in all_runs_df.iterrows():
    player_dh = PlayerDataProcessor(best_row.player, sprint_data_holder.get_player_data(best_row.player))
    mfr_best = MuscleFatigueRecovery(
        M0=best_row.M0,
        alpha=best_row.alpha,
        beta=best_row.beta,
        F=best_row.F,
        R=best_row.R
    )
    player_dh.calculate_euler_values(mfr_best)
    measured_values = player_dh.data['test_values']['real_values']
    model_values = player_dh.data['test_values']['calculated_values']

    player_error = player_dh.get_error()
    test_results = player_dh.data['test_values']
    r2 = r2_score(measured_values, model_values)
    res_df = pd.DataFrame({
        'run_cnt': best_row.run_count,
        'n_evals': best_row.number_of_evaluations,
        'alpha': best_row.alpha,
        'beta': best_row.beta,
        'F': best_row.F,
        'R': best_row.R,
        'cost_f': player_error,
        'r2': r2,
    }, index=[0])
    one_player_results_df = pd.concat([one_player_results_df, res_df])

with open(os.path.join(TABLES_DIR, 'variation_through_runs_one_player_table.tex'), 'w') as f:
    sys.stdout = f

    for idx, tex_df in one_player_results_df.iterrows():
        tex_df = tex_df.round(5)
        print('\midrule')
        print(
            f'${int(tex_df.run_cnt+1)}$ & ${int(tex_df.n_evals)}$ & ${format_number(tex_df.alpha)}$ & ${format_number(tex_df.beta)}$ & ${format_number(tex_df.F)}$ & ${format_number(tex_df.R)}$ & ${tex_df.cost_f:.0f}$ & ${tex_df.r2:.2f}$\\\\'
        )
sys.stdout = sys.__stdout__
