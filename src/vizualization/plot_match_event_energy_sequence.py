import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from matplotlib import pyplot as plt
from settings import REPORTS_DIR, FIGURES_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.math_utils import euler_method
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import log, set_save_path_arg_parser
from src.vizualization.helpers import load_plt_style

folder_path = set_save_path_arg_parser()
load_name = 'pso'

load_plt_style()
sprint_data_holder = DataHolder()
best_runs_df = pd.read_csv(os.path.join(REPORTS_DIR, load_name, folder_path, f'{load_name}_best_run_df.csv'))
best_runs_df = best_runs_df[best_runs_df.player.isin(sprint_data_holder.get_players())]
best_runs_df.reset_index(drop=True, inplace=True)

athlete1_best_df = best_runs_df.query('player == "athlete1"').squeeze()
log(f'Processing player: {athlete1_best_df.player}')

player_dh = PlayerDataProcessor(athlete1_best_df.player, sprint_data_holder.get_player_data(athlete1_best_df.player))
mfr_best = MuscleFatigueRecovery(
    M0=athlete1_best_df.M0,
    alpha=athlete1_best_df.alpha,
    beta=athlete1_best_df.beta,
    F=athlete1_best_df.F,
    R=athlete1_best_df.R
)
player_match = player_dh.get_player_matches()[0]
figsize = (9, 4)
k_divider = 1000
xlabel = 't [min]'
ylabel = 'E [kJ]'

y0 = np.array([1, 0]) * player_match['real_values'][0]
t = player_match['seconds']
y_euler = euler_method(
    y0,
    t,
    mfr_best.M0,
    player_match['m_ad'],
    mfr_best.alpha,
    mfr_best.beta,
    mfr_best.F,
    mfr_best.R,
)
player_match['calculated_values'] = y_euler[:, 0]
player_match['calculated_fatigue_values'] = y_euler[:, 1]

m_p = mfr_best.M0 - player_match['calculated_values'] - player_match['calculated_fatigue_values']

warnings.simplefilter('default')
# ----------- plotting -----------
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(figsize[0], figsize[1]), sharex=True,
    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.04}
)
ax[0].set_ylim(-0.05, (mfr_best.M0 / k_divider) * 1.05)

x_t = t / 60
m_ad = player_match['m_ad'](t) / k_divider
m_a = player_match['calculated_values'] / k_divider
m_p = m_p / k_divider
m_f = player_match['calculated_fatigue_values'] / k_divider
m_true = player_match['real_values'] / k_divider

ax[0].plot(x_t, m_ad, label='$M_{AD}$', linestyle='--')
ax[0].plot(x_t, m_a, label='$M_{A}$', linestyle=':', linewidth=2)
ax[0].plot(x_t, m_p, label='$M_{P}$', linestyle='-', linewidth=2)
ax[0].plot(x_t, m_f, label='$M_{F}$', linestyle='-.', linewidth=2)

ax[0].set_ylabel(ylabel)
ax[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax[0].set_xlim((76, 82))
ax[1].set_xlim((76, 82))

annotate_font_size = 12
annotate_lw = 2
ax[0].annotate(
    'Long ball\n(Unsuccessful)',
    xy=(76.4, m_ad[int(76.4 * 60)]), xycoords='data',
    xytext=(76.1, 2.9), textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=annotate_lw),
    fontsize=annotate_font_size, color='black', weight='bold'
)
ax[0].axvspan(76.5, 78, color='orangered', alpha=0.3, label='Out-of-possession')
ax[0].axvspan(78, 79.8, color='lightgreen', alpha=0.3, label='Possession')
ax[0].axvspan(79.8, 79.9, color='orangered', alpha=0.3)
ax[0].axvspan(79.9, 80.15, color='lightgreen', alpha=0.3)
ax[0].axvspan(80.15, 82, color='orangered', alpha=0.3)

ax[0].annotate(
    'Lost ball',
    xy=(79.8, m_ad[int(79.8 * 60)]), xycoords='data',
    xytext=(79, 3), textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=annotate_lw),
    fontsize=annotate_font_size, color='black'
)
ax[0].annotate(
    'Ball recovery',
    xy=(79.9, m_ad[int(79.9 * 60)]), xycoords='data',
    xytext=(79.6, 3), textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=annotate_lw),
    fontsize=annotate_font_size, color='black'
)
ax[0].annotate(
    'Dribble',
    xy=(80.25, m_ad[int(80.25 * 60)]), xycoords='data',
    xytext=(80.6, 3), textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=annotate_lw),
    fontsize=annotate_font_size, color='black'
)
ax[0].legend(loc='upper left', bbox_to_anchor=(0.025, 1.15), ncol=6)

norm = mcolors.Normalize(vmin=0, vmax=100)
colors = ['red', 'orange', 'yellow', 'green']
cmap = mcolors.LinearSegmentedColormap.from_list('battery_color_scheme', colors)

capacity_percentage = m_p / (athlete1_best_df.M0 / 1000)
capacity_percentage *= 100

ax[1].grid(False)
ax[1].plot(x_t, capacity_percentage, color='black')
for i in range(1, len(x_t)):
    ax[1].fill_between(
        x_t[i - 1:i + 1], capacity_percentage[i - 1:i + 1],
        color=cmap(norm(capacity_percentage[i]))
    )

ax[1].set_ylim(0, 100)
ax[1].set_xlabel(xlabel)
ax[1].set_ylabel('E (\%)')
cbar_ax = fig.add_axes([0.15, 0.04, 0.3, 0.03])  # [left, bottom, width, height]
colorbar = plt.colorbar(
    mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cbar_ax,
    orientation='horizontal',
)
ax[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

save_name = f'{athlete1_best_df.player}_{folder_path}_match_sequence.png'
save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
create_dir(save_path)
plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
