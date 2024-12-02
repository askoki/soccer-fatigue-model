import os

from matplotlib import pyplot as plt

from settings import FIGURES_DIR
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.utils import log
from src.vizualization.helpers import load_plt_style

sprint_data_holder = DataHolder()
player_name = sprint_data_holder.get_players()[2]

load_plt_style()
player_dh = PlayerDataProcessor(player_name, sprint_data_holder.get_player_data(player_name))
test_results = player_dh.data['test_values']
fig, ax = plt.subplots(figsize=(6, 2))
t = test_results['seconds']

x_t = t
k_divider = 1000
ax.plot(x_t, test_results['m_ad'](t) / k_divider, label='$M_{AD}$ (brain demand)', linestyle='--')
ax.plot(x_t, test_results['real_values'] / k_divider, label='$E_{spent}$', color='purple', alpha=0.5)

plt.legend(loc=(-0.12, -0.28), ncol=2)
plt.xlabel('t [s]')
plt.ylabel('E [kJ]')
plt.savefig(os.path.join(FIGURES_DIR, f'example_sprint_80m.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot matches
log('Plotting matches...')
p_match = player_dh.get_player_matches()[0]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2))
t = p_match['seconds']
x_t = t
ax.plot(x_t, p_match['m_ad'](t) / k_divider, label='$M_{AD}$ (brain demand)', linestyle='--')
ax.plot(x_t, p_match['real_values'] / k_divider, label='$E_{spent}$', color='purple', alpha=0.5)
ax.set_xlim(400, 500)
ax.set_ylim(0, 2.2)

plt.legend(loc='upper right', ncol=2)
plt.xlabel('t [s]')
plt.ylabel('E [kJ]')
plt.savefig(os.path.join(FIGURES_DIR, f'example_match.png'), dpi=300, bbox_inches='tight')
