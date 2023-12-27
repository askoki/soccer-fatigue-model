import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from settings import FIGURES_DIR, RAW_DATA_DIR
from src.features.file_helpers import create_dir
from src.vizualization.helpers import load_plt_style
from src.features.optimisation.math_utils import dy_dt
from src.vizualization.related_work.calculate_helpers import sns_dy_dt, get_least_square_results
from src.vizualization.related_work.processing_helpers import create_custom_m_ad

test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, '3_soccer_specific_intermittent_protocol_15min.csv'))
test_df = test_df.rename(columns={'t (min)': 't', 'velocity (km/h)': 'v'})
test_df = test_df.drop_duplicates('t')
test_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)
create_custom_m_ad(df=test_df, y_param='v', use_min_grouping=True)
m_ad = interp1d(test_df.t, test_df.m_ad, kind='linear', fill_value='extrapolate')

T = np.linspace(0, test_df.t.iloc[-1], int(test_df.shape[0]) * 10)

# ------------------- optimize for sns --------------------
initial_params = [27, 0.05, 0.05]
lb = [25, 1e-2, 1e-2]
ub = [30, 1e-1, 1e-1]

optimized_params = get_least_square_results(
    df=test_df,
    initial_params=initial_params,
    m_ad=m_ad, dy_dt=sns_dy_dt, lb=lb, ub=ub, y_param='v'
)
# Extract optimized parameters
m0_opt, f_opt, r_opt = optimized_params
print(f'M0={m0_opt}, F={f_opt}, R={r_opt}')

Y0 = np.array([1, 0]) * test_df.m_ad.iloc[0]
Y = odeint(sns_dy_dt, Y0, T, args=(m0_opt, m_ad, f_opt, r_opt), h0=1)
m_a, m_f = Y[:, 0], Y[:, 1]
m_p = m0_opt - m_a - m_f

# ------------------- optimize for model --------------------
initial_params_model = [27, 10, 10, 0.05, 0.05]
lb_model = [25, 5, 5, 1e-2, 1e-2]
ub_model = [30, 15, 15, 1e-1, 1e-1]

optimized_params = get_least_square_results(
    df=test_df,
    initial_params=initial_params_model,
    m_ad=m_ad, dy_dt=dy_dt, lb=lb_model, ub=ub_model, y_param='v'
)
# Extract optimized parameters
m0_opt_model, alpha_opt_model, beta_opt_model, f_opt_model, r_opt_model = optimized_params
print(f'M0={m0_opt_model}, alpha={alpha_opt_model}, beta={beta_opt_model}, F={f_opt_model}, R={r_opt_model}')

imp_Y = odeint(dy_dt, Y0, T, args=(m0_opt_model, m_ad, alpha_opt_model, beta_opt_model, f_opt_model, r_opt_model), h0=1)
imp_m_a, imp_m_f = imp_Y[:, 0], imp_Y[:, 1]
imp_m_p = m0_opt_model - imp_m_a - imp_m_f

load_plt_style()

dt = T[1] - T[0]

fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2))
p1, = ax.plot(T, m_ad(T), label='$M_{AD}$', linestyle='--')
p2, = ax.plot(T, m_a, label='SNS $M_{A}$', linestyle=':', linewidth=2)
p3, = ax.plot(test_df.t, test_df.v, label='Measured', color='purple')

ax.legend(ncol=3, loc=(0.18, 0.9))
# ax.set_xlim(0, m_ad_data['x'][-1])
ax.set_xlabel('t [min]')
ax.set_ylabel('v [km/h]')

save_path = os.path.join(FIGURES_DIR, 'related_manuscript')
create_dir(save_path)

fig.savefig(os.path.join(save_path, f'sns_problems_soccer_specific_intermittent_protocol_15min.png'), dpi=300)

p4, = ax.plot(T, imp_m_a, label='$M_{A}$', linestyle=':', linewidth=2, marker='^', markevery=30, markersize=4)
l1 = ax.legend([p1, p3], ['$M_{AD}$', 'Measured'], loc=(0.4, 0.14))
l2 = ax.legend([p2, p4], ['SNS $M_{A}$', '$M_{A}$'], loc=(0.78, 0.14))
fig.gca().add_artist(l1)
fig.savefig(os.path.join(save_path, f'compare_with_sns_soccer_specific_intermittent_protocol_15min.png'), dpi=300)


