import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from src.features.typing import MReqDict
from settings import FIGURES_DIR, RAW_DATA_DIR
from src.features.file_helpers import create_dir
from src.vizualization.helpers import load_plt_style
from src.features.optimisation.math_utils import dy_dt
from src.vizualization.related_work.calculate_helpers import sns_dy_dt, get_least_square_results

test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, '2_soccer_specific_drill_1min.csv'))
test_df = test_df.rename(columns={'t (s)': 't', 'velocity (km/h)': 'v'})
m_ad_data = MReqDict(
    x=[0, 4, 10, 18, 27, 34, 43, 47, 58, 60],
    y=[3, 3, 20, 1, 17, 2, 15, 2, 16, 2],
    label='soccer_specific_1min_drill'
)
m_ad = interp1d(m_ad_data['x'], m_ad_data['y'], kind='next', fill_value='extrapolate')
T = np.linspace(0, m_ad_data['x'][-1], int(m_ad_data['x'][-1]) * 10)

# ------------------- optimize for sns --------------------
initial_params = [20, 0.05, 0.05]
lb = [18, 1e-2, 1e-2]
ub = [22, 1e-1, 1e-1]

optimized_params = get_least_square_results(
    df=test_df,
    initial_params=initial_params,
    m_ad=m_ad, dy_dt=sns_dy_dt, lb=lb, ub=ub, y_param='v'
)
# Extract optimized parameters
m0_opt, f_opt, r_opt = optimized_params
print(f'M0={m0_opt}, F={f_opt}, R={r_opt}')

Y0 = np.array([1, 0]) * m_ad_data['y'][1]
Y = odeint(sns_dy_dt, Y0, T, args=(m0_opt, m_ad, f_opt, r_opt), h0=1)
m_a, m_f = Y[:, 0], Y[:, 1]
m_p = m0_opt - m_a - m_f

# ------------------- optimize for model --------------------
initial_params_model = [20, 0.02, 0.02, 0.05, 0.05]
lb_model = [18, 1e-2, 1e-2, 1e-2, 1e-2]
ub_model = [22, 1e-1, 1e-1, 1e-1, 1e-1]

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
ax.plot(T, m_ad(T), label='$M_{AD}$', linestyle='--')
ax.plot(T, m_a, label='SNS $M_{A}$', linestyle=':', linewidth=2)
ax.plot(test_df.t, test_df.v, label='Measured', color='purple')

horizontal_ellipse = Ellipse((30, 4.5), 11, 2, color='red', angle=0, fill=False, linewidth=2)
ax.add_patch(horizontal_ellipse)

horizontal_ellipse2 = Ellipse((45, 8), 12, 8, color='red', angle=70, fill=False, linewidth=2)
ax.add_patch(horizontal_ellipse2)

vertical_ellipse = Ellipse((18, 9), 11, 2, color='red', angle=90, fill=False, linewidth=2)
ax.add_patch(vertical_ellipse)

ax.legend(ncol=3, loc=(0.4, 0.83))
ax.set_xlim(0, m_ad_data['x'][-1])
ax.set_xlabel('t [s]')
ax.set_ylabel('v [km/h]')

save_path = os.path.join(FIGURES_DIR, 'related_manuscript')
create_dir(save_path)
fig.savefig(os.path.join(save_path, f'sns_problems_soccer_specific_1min_drill.png'), dpi=300)

horizontal_ellipse.remove()
horizontal_ellipse2.remove()
vertical_ellipse.remove()

ax.plot(T, imp_m_a, label='$M_{A}$', linestyle=':', linewidth=2, marker='^', markevery=5)
ax.legend(ncol=2, loc=(0.6, 0.78))
fig.savefig(os.path.join(save_path, f'compare_with_sns_soccer_specific_1min_drill.png'), dpi=300)
