import datetime
from typing import TypedDict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

AlphaBetaArg = Tuple[float, float]

PlayerMatchMeasurement = TypedDict(
    'PlayerMatchMeasurement', {
        'seconds': np.array,
        'm_ad': interp1d,
        'real_values': np.array,
        'calculated_values': np.array,
        'calculated_fatigue_values': np.array,
        'num_minutes': int
    })

PlayerTestData = TypedDict(
    'PlayerTestData',
    {
        'test_values': PlayerMatchMeasurement,
        'match_values': List[PlayerMatchMeasurement],
        'total_match_minutes': int,
        'max_speed_ms': float,
        'weight': int,
        'total_w': float,
    }
)

RunDict = TypedDict(
    'RunDict', {
        'run': int,
        'X': List[float],
        'f': List[float]
    }
)

MReqDict = TypedDict(
    'MReqDict', {
        'x': List[int],
        'y': List[int],
        'label': str
    }
)


class MuscleFatigueRecovery:
    def __init__(self, M0: float, alpha: float, beta: float, F: float, R: float):
        self.M0 = M0
        self.F = F
        self.gamma = None
        self.R = R
        self.alpha = alpha
        self.beta = beta

    def get_series(self) -> pd.Series:
        df = pd.DataFrame({
            'M0': self.M0, 'alpha': self.alpha, 'beta': self.beta, 'F': self.F, 'gamma': self.gamma, 'R': self.R
        }, index=[0])
        df = df.round(5)
        return df.squeeze()

    def get_input_names(self) -> np.array:
        return np.array(['M0', 'alpha', 'beta', 'F', 'R'])

    def get_input_vector(self) -> np.array:
        return np.array([self.M0, self.alpha, self.beta, self.F, self.R])
