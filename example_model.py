import numpy as np
import matplotlib.pyplot as plt

from src.features.optimisation.data_loaders import DataHolder

"""
Nogomet 96 min
"""
sprint_data_holder = DataHolder()
player_name = 'athlete1'
player_dh = sprint_data_holder.get_player_data(player_name)

# test_data = player_dh['test_values']
test_data = player_dh['match_values'][0]

time = np.linspace(0, test_data['seconds'].shape[0], test_data['seconds'].shape[0] * 10)
W = test_data['m_ad']

# alpha = 5e-4  # Određeno iz 100m sprinta
# beta = 1e-1
# gamma = 5e-4  # Određeno iz 100m sprinta
# delta = 6e-4
# alpha = 1e-3
alpha = 9.97e-5
# beta = 0.019
beta = 0.049
# gamma = 0.001
gamma = 0.023267
# delta = 1e-3
delta = gamma * 0.41
# delta = 0.000966421

Tot = player_dh['total_w']
dt = time[1] - time[0]


def dY_dt(Y, t):
    A, F = Y
    P = Tot - A - F

    dW = (W(t) - A)
    dA_dt = np.max([0, dW]) * P * alpha + np.min([0, dW]) * A * beta
    dF_dt = A * gamma - F * delta

    dF_dt = np.array([dA_dt, dF_dt])
    return dF_dt


Y = np.full([time.size, 2], np.nan)
Y0 = np.array([0, 0])
Y[0, :] = Y0

for i in range(time.size - 1):
    Y[i + 1, :] = Y[i, :] + dt * dY_dt(Y[i, :], time[i])
A = Y[:, 0]
F = Y[:, 1]
P = Tot - A - F
print(f'{A=}')

fig, ax = plt.subplots(figsize=(16, 8))
t = time / 60
ax.plot(t, W(time), ls='--', label='W (target)')
ax.plot(t, A, label='A (active)')
ax.plot(t, P, label='P (passive)')
ax.plot(t, F, label='F (fatigued)')
# import pdb; pdb.set_trace()
ax.plot(
    test_data['seconds'] / 60,
    test_data['real_values'],
    label='real values',
    ls=':'
)
ax.set_xlabel('t [min]')
# ax.set_xlim(0, 5)
ax.set_ylabel('E [W]')
ax.set_ylim(-0.05, Tot * 1.1)
# A_lim = delta / (gamma + delta)
# ax.axhline(A_lim, label=f'{A_lim=}')
ax.legend()

# print(sum((A[::10] - game_data['real_values'])**2) / game_data['num_minutes'])
plt.show()
