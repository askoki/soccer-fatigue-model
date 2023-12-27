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
# Plot matches
log('Plotting matches...')
player_match = player_dh.get_player_matches()[0]
figsize = (9, 3)
k_divider = 1000
xlabel = 't [min]'
ylabel = 'E [kJ]'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figsize[0], figsize[1]))
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
try:
    ax.set_ylim(-0.05, (mfr_best.M0 / k_divider) * 1.05)
except TypeError:
    print('Should not be here')
    # Only one match for that player

x_t = t / 60
m_ad = player_match['m_ad'](t) / k_divider
m_a = player_match['calculated_values'] / k_divider
m_p = m_p / k_divider
m_f = player_match['calculated_fatigue_values'] / k_divider
m_true = player_match['real_values'] / k_divider

ax.plot(x_t, m_ad, label='$M_{AD}$', linestyle='--')
ax.plot(x_t, m_a, label='$M_{A}$', linestyle=':', linewidth=2)
ax.plot(x_t, m_p, label='$M_{P}$', linestyle='-')
ax.plot(x_t, m_f, label='$M_{F}$', linestyle='-.')
ax.plot(x_t, m_true, label='$E_{spent}$', color='purple', alpha=0.5)

# ---------- add zoom plot in the first match ----------
snippet_t = int(x_t[-1] * 0.5)

x1, x2, y1, y2 = snippet_t, snippet_t + 3, 0, np.max(m_p)  # subregion of the original image
axins = ax.inset_axes(
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
ax.indicate_inset_zoom(axins, edgecolor='brown', linewidth=3, alpha=1)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.legend(loc='center right', ncol=2)
save_name = f'{athlete1_best_df.player}_{folder_path}_match1.png'
save_path = os.path.join(FIGURES_DIR, load_name, folder_path)
create_dir(save_path)
plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
