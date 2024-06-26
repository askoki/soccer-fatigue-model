import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from settings import REPORTS_DIR, TABLES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import set_save_path_arg_parser

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

    mae = mean_absolute_error(measured_values, model_values)
    mse = mean_squared_error(measured_values, model_values)
    rmse = mean_squared_error(measured_values, model_values, squared=False)
    r2 = r2_score(measured_values, model_values)

    res_df = pd.DataFrame({
        'player': best_row.player,
        'alpha': best_row.alpha,
        'beta': best_row.beta,
        'F': best_row.F,
        'R': best_row.R,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }, index=[0])
    all_players_results_df = pd.concat([all_players_results_df, res_df])

tmp = all_players_results_df.select_dtypes(include=[np.number])
all_players_results_df.loc[:, tmp.columns] = np.round(tmp, 5)
all_players_results_df.loc[:, 'player'] = all_players_results_df.player.str.capitalize()
with open(os.path.join(TABLES_DIR, 'best_runs_performance_table.tex'), 'w') as f:
    sys.stdout = f

    for idx, tex_df in all_players_results_df.iterrows():
        print('\midrule')
        print(
            f'{tex_df.player} & ${tex_df.mae:.5f}$ & ${tex_df.mse:.5f}$ & ${tex_df.rmse:.5f}$ & ${tex_df.r2:.5f}$ \\\\'
        )
sys.stdout = sys.__stdout__
