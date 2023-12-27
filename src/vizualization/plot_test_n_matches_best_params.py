import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from settings import REPORTS_DIR, FIGURES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.math_utils import euler_method
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import log, set_nm_pso_collect_path_arg_parser

is_pso, folder_path = set_nm_pso_collect_path_arg_parser()
load_name = 'pso' if is_pso else 'nm'

sprint_data_holder = DataHolder()

best_runs_df = pd.read_csv(os.path.join(REPORTS_DIR, load_name, folder_path, f'{load_name}_best_run_df.csv'))
best_runs_df = best_runs_df[best_runs_df.player.isin(sprint_data_holder.get_players())]
best_runs_df.reset_index(drop=True, inplace=True)
for idx, best_row in best_runs_df.iterrows():
    log(f'Processing player: {best_row.player} {idx + 1}/{best_runs_df.index.max() + 1}')
    player_dh = PlayerDataProcessor(best_row.player, sprint_data_holder.get_player_data(best_row.player))
    mfr_best = MuscleFatigueRecovery(
        M0=best_row.M0,
        alpha=best_row.alpha,
        beta=best_row.beta,
        F=best_row.F,
        gamma=best_row.gamma
    )
    player_dh.calculate_euler_values(mfr_best)
    test_results = player_dh.data['test_values']
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f'{best_row.player} M0={mfr_best.M0}, $\\alpha$={mfr_best.alpha} $\\beta$={mfr_best.beta} F={mfr_best.F}, R={mfr_best.R}'
    )

    t = test_results['seconds']
    m_p = mfr_best.M0 - test_results['calculated_values'] - test_results['calculated_fatigue_values']
    try:
        ax.set_ylim(-0.05, mfr_best.M0 * 1.05)
    except TypeError:
        print(best_row.player)
        # Only one match for that player
        continue
    x_t = t / 60
    ax.plot(x_t, test_results['m_ad'](t), label='$M_{AD}$', linestyle='--')
    ax.plot(x_t, test_results['calculated_values'], label='$M_{A}$', linestyle=':', linewidth=2)
    ax.plot(x_t, m_p, label='$M_{P}$', linestyle='-')
    ax.plot(x_t, test_results['calculated_fatigue_values'], label='$M_{F}$', linestyle='-.')
    ax.plot(x_t, test_results['real_values'], label='$v_{true}$', color='purple', alpha=0.5)

    plt.legend(loc='upper right', ncol=2)
    plt.xlabel('t [min]')
    plt.ylabel('J/kg')
    save_name = f'{best_row.player.replace(" ", "_").replace(".", "")}_{folder_path}.png'
    save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
    create_dir(save_path)
    plt.savefig(os.path.join(save_path, save_name), dpi=300)
    plt.close()

    # Plot matches
    log('Plotting matches...')
    player_matches = player_dh.data['match_values']
    fig, ax = plt.subplots(figsize=(12, 4 * len(player_matches)))
    fig.suptitle(
        f'{best_row.player} M0={mfr_best.M0}, $\\alpha$={mfr_best.alpha} $\\beta$={mfr_best.beta} F={mfr_best.F}, R={mfr_best.R}'
    )
    for match_id, p_match in enumerate(player_matches):
        y0 = np.array([1, 0]) * p_match['real_values'][0]
        t = p_match['seconds']
        y_euler = euler_method(
            y0,
            t,
            mfr_best.M0,
            p_match['m_ad'],
            mfr_best.alpha,
            mfr_best.beta,
            mfr_best.F,
            mfr_best.R,
        )
        p_match['calculated_values'] = y_euler[:, 0]
        p_match['calculated_fatigue_values'] = y_euler[:, 1]

        m_p = mfr_best.M0 - p_match['calculated_values'] - p_match['calculated_fatigue_values']
        try:
            ax.set_ylim(-0.05, mfr_best.M0 * 1.05)
        except TypeError:
            print(best_row.player)
            # Only one match for that player
            continue
        x_t = t / 60
        ax.plot(x_t, p_match['m_ad'](t), label='$M_{AD}$', linestyle='--')
        ax.plot(x_t, p_match['calculated_values'], label='$M_{A}$', linestyle=':', linewidth=2)
        ax.plot(x_t, m_p, label='$M_{P}$', linestyle='-')
        ax.plot(x_t, p_match['calculated_fatigue_values'], label='$M_{F}$', linestyle='-.')
        ax.plot(x_t, p_match['real_values'], label='$v_{true}$', color='purple', alpha=0.5)

        plt.legend(loc='upper right', ncol=2)
        plt.xlabel('t [min]')
        plt.ylabel('J/kg')
        save_name = f'{best_row.player.replace(" ", "_").replace(".", "")}_{folder_path}_match{match_id + 1}.png'
        save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
        create_dir(save_path)
        plt.savefig(os.path.join(save_path, save_name), dpi=300)
