import os
from functools import partial
from typing import Callable, List

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def sns_dy_dt(y: np.array, t: float, m0: float, m_ad: interp1d, fatigue_rate: float, recovery_rate: float):
    m_a, m_f = y

    m_uc = m0 - m_a - m_f
    if m_ad(t) < m_a + m_uc:
        d_m_a = (m_ad(t + 1e-5) - m_ad(t - 1e-5)) / 2e-5
    else:
        # max changing rate per second
        d_m_a = recovery_rate * m_f - fatigue_rate * m_a + np.max([m_uc, 0])
    d_m_f = fatigue_rate * m_a - recovery_rate * m_f
    return np.array([d_m_a, d_m_f])


def model_function(t, m_ad: interp1d, params: List[float], dy_dt: Callable):
    y0 = np.array([0, 0])
    y = odeint(dy_dt, y0, t, args=(params[0], m_ad, *params[1:]), h0=1)
    m_a, m_f = y[:, 0], y[:, 1]
    return m_a


# Function to compute residuals
def residuals(params, t, m_ad, dy_dt: Callable):
    model_predictions = model_function(t, m_ad=m_ad, params=params, dy_dt=dy_dt)
    return model_predictions


def calculate_least_square_fit(t: np.array, *argv, m_ad: interp1d, dy_dt: Callable):
    return residuals(argv, t, m_ad, dy_dt=dy_dt)


def get_least_square_results(df: pd.DataFrame, initial_params: list, m_ad: interp1d, dy_dt: Callable, lb: list,
                             ub: list, y_param: str) -> np.array:
    fit_function = partial(calculate_least_square_fit, m_ad=m_ad, dy_dt=dy_dt)
    optimized_params, _ = curve_fit(
        f=fit_function,
        xdata=df.t,
        ydata=df[y_param],
        p0=initial_params,
        bounds=(lb, ub)
    )
    return optimized_params
