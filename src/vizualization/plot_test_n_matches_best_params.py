import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from settings import REPORTS_DIR, FIGURES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.math_utils import euler_method
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import log, set_save_path_arg_parser
from src.vizualization.helpers import load_plt_style

is_pso, folder_path = set_save_path_arg_parser()
load_name = 'pso'

load_plt_style()
sprint_data_holder = DataHolder()
best_runs_df = pd.read_csv(os.path.join(REPORTS_DIR, load_name, folder_path, f'{load_name}_best_run_df.csv'))
best_runs_df = best_runs_df[best_runs_df.player.isin(sprint_data_holder.get_players())]
best_runs_df.reset_index(drop=True, inplace=True)

figsize = (9, 3)
for idx, best_row in best_runs_df.iterrows():
    log(f'Processing player: {best_row.player} {idx + 1}/{best_runs_df.index.max() + 1}')
    player_dh = PlayerDataProcessor(best_row.player, sprint_data_holder.get_player_data(best_row.player))
    mfr_best = MuscleFatigueRecovery(
        M0=best_row.M0,
        alpha=best_row.alpha,
        beta=best_row.beta,
        F=best_row.F,
        R=best_row.R
    )
    player_dh.calculate_euler_values(mfr_best)
    test_results = player_dh.data['test_values']
    fig, ax = plt.subplots(figsize=figsize)
    mfr_s = mfr_best.get_series()
    title = f'M0={int(mfr_s.M0)}, $\\alpha$={mfr_s.alpha:.5f} $\\beta$={mfr_s.beta:.5f} F={mfr_s.F:.5f}, R={mfr_s.R:.5f}'
    # fig.suptitle(title)

    t = test_results['seconds']
    k_divider = 1000
    m_p_test = mfr_best.M0 - test_results['calculated_values'] - test_results['calculated_fatigue_values']
    try:
        ax.set_ylim(-0.05, (mfr_best.M0 / k_divider) * 1.05)
    except TypeError:
        print(best_row.player)
        # Only one match for that player
        continue
    x_t = t / 60
    m_ad_test = test_results['m_ad'](t) / k_divider
    m_a_test = test_results['calculated_values'] / k_divider
    m_p_test = m_p_test / k_divider
    m_f_test = test_results['calculated_fatigue_values'] / k_divider
    m_true_test = test_results['real_values'] / k_divider

    ax.plot(x_t, m_ad_test, label='$M_{AD}$', linestyle='--')
    ax.plot(x_t, m_a_test, label='$M_{A}$', linestyle=':', linewidth=2)
    ax.plot(x_t, m_p_test, label='$M_{P}$', linestyle='-')
    ax.plot(x_t, m_f_test, label='$M_{F}$', linestyle='-.')
    ax.plot(x_t, m_true_test, label='$E_{spent}$', color='purple', alpha=0.5)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend(loc='upper right', ncol=2)
    xlabel = 't [min]'
    ylabel = 'E [kJ]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_name = f'{best_row.player}_{folder_path}.png'
    save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
    create_dir(save_path)
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot matches
    log('Plotting matches...')
    player_matches = player_dh.get_player_matches()
    fig, ax = plt.subplots(nrows=len(player_matches), ncols=1, figsize=(figsize[0], figsize[1] * len(player_matches)))
    # fig.suptitle(title)
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
            ax[match_id].set_ylim(-0.05, (mfr_best.M0 / k_divider) * 1.05)
        except TypeError:
            print('Should no be here')
            # Only one match for that player
            continue

        x_t = t / 60
        m_ad = p_match['m_ad'](t) / k_divider
        m_a = p_match['calculated_values'] / k_divider
        m_p = m_p / k_divider
        m_f = p_match['calculated_fatigue_values'] / k_divider
        m_true = p_match['real_values'] / k_divider

        ax[match_id].plot(x_t, m_ad, label='$M_{AD}$', linestyle='--')
        ax[match_id].plot(x_t, m_a, label='$M_{A}$', linestyle=':', linewidth=2)
        ax[match_id].plot(x_t, m_p, label='$M_{P}$', linestyle='-')
        ax[match_id].plot(x_t, m_f, label='$M_{F}$', linestyle='-.')
        ax[match_id].plot(x_t, m_true, label='$E_{spent}$', color='purple', alpha=0.5)

        # add zoom plot in the first match
        snippet_t = int(x_t[-1] * 0.5)
        if match_id == 0:
            x1, x2, y1, y2 = snippet_t, snippet_t + 3, 0, np.max(m_p)  # subregion of the original image
            axins = ax[match_id].inset_axes(
                [0.1, 1.0, 0.8, 0.5],
                xlim=(x1, x2), ylim=(y1, y2)
            )
            axins.plot(x_t, m_ad, label='$M_{AD}$', linestyle='--')
            axins.plot(x_t, m_a, label='$M_{A}$', linestyle=':', linewidth=2)
            axins.plot(x_t, m_p, label='$M_{P}$', linestyle='-')
            axins.plot(x_t, m_f, label='$M_{F}$', linestyle='-.')
            axins.plot(x_t, m_true, label='$E_{spent}$', color='purple', alpha=0.5)
            axins.set_ylabel(ylabel)
            axins.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax[match_id].indicate_inset_zoom(axins, edgecolor='brown', linewidth=3, alpha=1)
        ax[match_id].set_xlabel(xlabel)
        ax[match_id].set_ylabel(ylabel)
        ax[match_id].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend(loc='upper right', ncol=2)
    save_name = f'{best_row.player}_{folder_path}_matches.png'
    save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
    create_dir(save_path)
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
