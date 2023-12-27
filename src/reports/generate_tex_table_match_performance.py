import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from settings import REPORTS_DIR, TABLES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.math_utils import euler_method
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

    res_df = pd.DataFrame({
        'player': best_row.player,
        'alpha': best_row.alpha,
        'beta': best_row.beta,
        'F': best_row.F,
        'R': best_row.R,
    }, index=[0])

    match_results = player_dh.data['match_values']
    error_matches = 0
    energy_matches = 0
    num_min = 0
    for i, match_result in enumerate(match_results):
        energy_matches += match_result['real_values'].sum()
        num_min += match_result['num_minutes']
        y0 = np.array([1, 0]) * match_result['real_values'][0]
        y_euler = euler_method(
            y0,
            match_result['seconds'],
            mfr_best.M0,
            match_result['m_ad'],
            mfr_best.alpha,
            mfr_best.beta,
            mfr_best.F,
            mfr_best.R,
        )
        match_result['calculated_values'] = y_euler[:, 0]
        match_result['calculated_fatigue_values'] = y_euler[:, 1]

        measured_values = match_result['real_values']
        model_values = match_result['calculated_values']

        error_matches += np.sum((match_result['calculated_values'] - match_result['real_values']) ** 2)
        r2_match = r2_score(measured_values, model_values)
        res_df.loc[:, f'r2_match{i + 1}'] = r2_match
    squared_err_per_min = error_matches / num_min
    res_df.loc[:, 'num_min'] = num_min
    res_df.loc[:, 'cost_f'] = squared_err_per_min
    res_df.loc[:, 'r2_mean'] = res_df.apply(lambda r: np.mean([r.r2_match1, r.r2_match2, r.r2_match3, r.r2_match4]), axis=1)
    all_players_results_df = pd.concat([all_players_results_df, res_df])

tmp = all_players_results_df.select_dtypes(include=[np.number])
all_players_results_df.loc[:, tmp.columns] = np.round(tmp, 5)
all_players_results_df.loc[:, 'player'] = all_players_results_df.player.str.capitalize()
with open(os.path.join(TABLES_DIR, 'match_performance_table.tex'), 'w') as f:
    sys.stdout = f

    for idx, tex_df in all_players_results_df.iterrows():
        print('\midrule')
        print(
            f'{tex_df.player} & ${int(tex_df.num_min)}$ & ${format_number(tex_df.cost_f)}$ & ${tex_df.r2_match1:.5f}$ & ${tex_df.r2_match2:.5f}$ & ${tex_df.r2_match3}$ & ${tex_df.r2_match4}$& ${tex_df.r2_mean}$\\\\'
        )
sys.stdout = sys.__stdout__
