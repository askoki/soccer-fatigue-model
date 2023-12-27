import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

from src.features.typing import MReqDict
from src.features.file_helpers import create_dir
from src.features.optimisation.math_utils import dy_dt
from src.vizualization.helpers import load_plt_style
from src.vizualization.related_work.calculate_helpers import get_least_square_results, sns_dy_dt
from settings import FIGURES_DIR, RAW_DATA_DIR

handgrip_df = pd.read_csv(os.path.join(RAW_DATA_DIR, '1_liu_max_handgrip.csv'))
handgrip_df = handgrip_df.rename(columns={'t (s)': 't', 'F (N)': 'F'})
m_ad_data = MReqDict(x=[0, 60, 120, 180], y=[435, 435, 435, 435], label='model_vs_sns_max_handgrip')
m_ad = interp1d(m_ad_data['x'], m_ad_data['y'], kind='nearest', fill_value='extrapolate')
T = np.linspace(0, m_ad_data['x'][-1], handgrip_df.t.shape[0])

initial_params = [400, 0.02, 0.02, 0.03, 0.01]
lb = [300, 1e-2, 1e-2, 1e-2, 1e-2]
ub = [450, 1e-1, 1e-1, 1e-1, 1e-1]

optimized_params = get_least_square_results(
    df=handgrip_df,
    initial_params=initial_params,
    m_ad=m_ad, dy_dt=dy_dt, lb=lb, ub=ub, y_param='F'
)
# Extract optimized parameters
m0_opt, alpha_opt, beta_opt, f_opt, r_opt = optimized_params
print(f'M0={m0_opt}, alpha={alpha_opt}, beta={beta_opt}, F={f_opt}, R={r_opt}')

load_plt_style()

dt = T[1] - T[0]

Y0 = np.array([0, 0]) * m0_opt
Y = odeint(dy_dt, Y0, T, args=(m0_opt, m_ad, alpha_opt, beta_opt, f_opt, r_opt), h0=1)
m_a, m_f = Y[:, 0], Y[:, 1]
m_p = m0_opt - m_a - m_f

F_sns = 0.0245
R_sns = 0.0115
m0_sns = 435
Y0 = np.array([0, 0])
Y_sns = odeint(sns_dy_dt, Y0, T, args=(m0_sns, m_ad, F_sns, R_sns), h0=1)
m_a_sns, m_f_sns = Y_sns[:, 0], Y_sns[:, 1]

r2_fit = r2_score(handgrip_df.F, m_a)
r2_fit_sns = r2_score(handgrip_df.F, m_a_sns)
print(f'R^2={r2_fit}')
print(f'R^2={r2_fit_sns}')

fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 2))
ax.scatter(handgrip_df.t, handgrip_df.F, color='black', s=8, label='Measured (Liu)')
ax.plot(T, m_ad(T), label='$M_{AD}$', linestyle='--')
ax.plot(T, m_a_sns, label='SNS $M_{A}$', linestyle='-', linewidth=2)

ax.legend(ncol=2, loc=(0.2, 0.6))
ax.set_xlim(0, m_ad_data['x'][-1])
ax.set_xlabel('t [s]')
ax.set_ylabel('Force [N]')

save_path = os.path.join(FIGURES_DIR, 'related_manuscript')
create_dir(save_path)

fig.savefig(os.path.join(save_path, f'sns_max_handgrip.png'), dpi=300)

ax.plot(T, m_a, label='Model $M_{A}$', linestyle=':', linewidth=2)
ax.legend(ncol=2, loc=(1/3, 0.55))
fig.savefig(os.path.join(save_path, f'model_vs_sns_max_handgrip.png'), dpi=300)
