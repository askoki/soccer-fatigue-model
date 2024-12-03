import os
import sys

import numpy as np
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
best_run_df = pd.read_csv(os.path.join(REPORTS_DIR, load_name, folder_path, f'{load_name}_best_run_df.csv'))
best_run_df = best_run_df[best_run_df.player.isin(sprint_data_holder.get_players())]
best_run_df.reset_index(drop=True, inplace=True)

all_players_results_df = pd.DataFrame()
for idx, best_row in best_run_df.iterrows():
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
        'player': best_row.player,
        'n_evals': best_row.number_of_evaluations,
        'm0': best_row.M0,
        'alpha': best_row.alpha,
        'beta': best_row.beta,
        'F': best_row.F,
        'R': best_row.R,
        'cost_f': player_error,
        'r2': r2,
    }, index=[0])
    all_players_results_df = pd.concat([all_players_results_df, res_df])

tmp = all_players_results_df.select_dtypes(include=[np.number])
all_players_results_df.loc[:, tmp.columns] = np.round(tmp, 5)
all_players_results_df.loc[:, 'player'] = all_players_results_df.player.str.capitalize()
with open(os.path.join(TABLES_DIR, 'best_runs_table.tex'), 'w') as f:
    sys.stdout = f

    for idx, tex_df in all_players_results_df.iterrows():
        print('\midrule')
        print(
            f'{tex_df.player} & ${int(tex_df.n_evals)}$ & ${int(tex_df.m0)}$ & ${format_number(tex_df.alpha)}$ & ${format_number(tex_df.beta)}$ & ${format_number(tex_df.F)}$ & ${format_number(tex_df.R)}$ & ${format_number(tex_df.cost_f)}$ & ${tex_df.r2:.2f}$\\\\'
        )
sys.stdout = sys.__stdout__
