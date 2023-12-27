import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from sklearn.metrics import r2_score

from settings import FIGURES_DIR
from src.features.file_helpers import create_dir
from src.features.typing import MReqDict
from src.vizualization.helpers import load_plt_style
from src.features.optimisation.math_utils import dy_dt as imp_dy_dt

m_ad_data = MReqDict(
    x=[0, 60, 150, 160, 170, 170.1, 180, 190, 200, 200, 220, 225, 230],
    y=[435, 435, 200, 50, 10, 300, 300, 200, 10, 250, 250, 200, 100],
    label='max_handgrip_with_extension'
)


def dy_dt(y: Tuple[int, int], t: float, m0: float, m_ad: interp1d, R: float, F: float):
    m_a, m_f = y

    m_uc = m0 - m_a - m_f
    if m_ad(t) < m_a + m_uc:
        d_m_a = (m_ad(t + 1e-4) - m_ad(t - 1e-4)) / 2e-4
    else:
        d_m_a = R * m_f - F * m_a + np.max([m_uc, 0])
    d_m_f = F * m_a - R * m_f
    dy = d_m_a, d_m_f
    return dy


F = 0.0245
R = 0.0115
m0 = 435
alpha = 0.005
beta = 0.05

load_plt_style()
T = np.linspace(0, m_ad_data['x'][-1], m_ad_data['x'][-1] * 10)
m_ad = interp1d(m_ad_data['x'], m_ad_data['y'], kind='linear', fill_value='extrapolate')

Y0 = np.array([0, 0]) * m0
Y = odeint(dy_dt, Y0, T, args=(m0, m_ad, R, F), h0=1)
m_a, m_f = Y[:, 0], Y[:, 1]

imp_Y = odeint(imp_dy_dt, Y0, T, args=(m0, m_ad, alpha, beta, F, R), h0=1)
imp_m_a, imp_m_f = imp_Y[:, 0], imp_Y[:, 1]

fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
ax.plot(T, m_ad(T), label='$M_{AD}$', linestyle='--', linewidth=3)
ax.plot(T, m_a, label='$M_{A}$', linestyle=':', linewidth=3)
ax.legend()
ax.set_ylim(0, 450)
ax.set_xlim(0, m_ad_data['x'][-1])
ax.set_xlabel('t [s]')
ax.set_ylabel('N')

circle_xy = (4, 395)
circle_f = Circle(circle_xy, 3, color='red', fill=False, linewidth=2)
ax.add_patch(circle_f)
ax.annotate(
    'F', xy=circle_xy, xytext=(circle_xy[0] + 20, circle_xy[1] + 15),
    fontsize=16, fontweight='bold',
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=6),
)

circle_r_xy = (100, 150)
circle_r = Ellipse(circle_r_xy, 80, 20, color='red', angle=-20, fill=False, linewidth=2)
ax.add_patch(circle_r)
ax.annotate(
    'R', xy=circle_r_xy, xytext=(circle_r_xy[0] + 10, circle_r_xy[1] + 50),
    fontsize=16, fontweight='bold',
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=6),
)
plt.axvline(x=150, color='black', linestyle='--', linewidth=2)

# question
elipse_question = (170, 160)
elipse_q = Ellipse(elipse_question, 250, 5, color='red', angle=90, fill=False, linewidth=2)
ax.add_patch(elipse_q)
ax.annotate(
    '?', xy=elipse_question, xytext=(elipse_question[0] + 10, circle_r_xy[1] - 80),
    fontsize=30, fontweight='bold',
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=6),
)
save_path = os.path.join(FIGURES_DIR, 'related_manuscript')
create_dir(save_path)
fig.savefig(os.path.join(save_path, m_ad_data['label']), dpi=300)
plt.close()

# Comparison with our approach
fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
ax.plot(T, m_ad(T), label='$M_{AD}$', linestyle='--', linewidth=3)
ax.plot(T, m_a, label='Original $M_{A}$', linestyle=':', linewidth=3)
ax.plot(T, imp_m_a, label='$M_{A}$', linestyle=':', linewidth=3, marker='^', markevery=40)
ax.legend()
ax.set_ylim(0, 450)
ax.set_xlim(0, m_ad_data['x'][-1])
ax.set_xlabel('t [s]')
ax.set_ylabel('N')
fig.savefig(os.path.join(save_path, f'comparison_{m_ad_data["label"]}'), dpi=300)
plt.close()

# Keep data relevant to max handgrip test
sns_m_a = m_a[:1500]
imp_m_a = imp_m_a[:1500]
r2 = r2_score(sns_m_a, imp_m_a)
print(f'$R^2$ score is {r2}')
