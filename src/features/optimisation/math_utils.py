import numpy as np
import warnings
from scipy.interpolate import interp1d
from src.features.utils import ComputationError


def dy_dt(y: np.array, t: np.array,
          m0: float, m_ad: interp1d, alpha: float, beta: float, fatigue_rate: float, recovery_rate: float):
    m_a, m_f = y
    m_p = m0 - m_a - m_f
    d_w = (m_ad(t) - m_a)
    warnings.filterwarnings('error')
    try:
        d_m_a_dt = np.max([0, d_w]) * m_p * alpha + np.min([0, d_w]) * m_a * beta  # * (m_a_max - m_a) * (m_a - m_a_min)
    except RuntimeWarning:
        raise ComputationError(
            message='Diff. equation broke due to out-of-bounds parameters',
            euler_step=t
        )
    d_m_f_dt = m_a * fatigue_rate - m_f * recovery_rate

    return np.array([d_m_a_dt, d_m_f_dt])


def euler_method(y0: np.array, t: np.array, m0: float, m_ad: interp1d, alpha: float, beta: float, fatigue_rate: float,
                 recovery_rate: float, n_points_multiplier=10):
    time = np.linspace(0, t[-1], n_points_multiplier * int(t[-1] + 1))
    dt = time[1] - time[0]
    y = np.full([time.size, 2], np.nan)
    y[0, :] = y0

    for i in range(time.size - 1):
        y[i + 1, :] = y[i, :] + dt * dy_dt(
            y[i, :],
            time[i],
            m0,
            m_ad,
            alpha=alpha,
            beta=beta,
            fatigue_rate=fatigue_rate,
            recovery_rate=recovery_rate
        )
    return y[::n_points_multiplier, :]
