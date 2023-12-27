import pdb
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List, Tuple

import numpy as np
from src.features.optimisation.math_utils import euler_method
from src.features.typing import MuscleFatigueRecovery, PlayerTestData, PlayerMatchMeasurement
from src.features.utils import ComputationError


class BaseDataProcessor(ABC):
    @abstractmethod
    def calculate_euler_values(self, mfr_descriptor: MuscleFatigueRecovery):
        pass

    @abstractmethod
    def get_units_in_use(self) -> np.array:
        pass

    @abstractmethod
    def get_error(self) -> float:
        pass


class PlayerDataProcessor(BaseDataProcessor):

    def __init__(self, player_name: str, data: PlayerTestData):
        self.player_name = player_name
        self.data = data

    def calculate_euler_values(self, mfr_descriptor: MuscleFatigueRecovery) -> None:
        y0 = np.array([1, 0]) * self.data['test_values']['real_values'][0]
        y_euler = euler_method(
            y0,
            self.data['test_values']['seconds'],
            mfr_descriptor.M0,
            self.data['test_values']['m_ad'],
            mfr_descriptor.alpha,
            mfr_descriptor.beta,
            mfr_descriptor.F,
            mfr_descriptor.R,
        )
        self.data['test_values']['calculated_values'] = y_euler[:, 0]
        self.data['test_values']['calculated_fatigue_values'] = y_euler[:, 1]

    def get_units_in_use(self):
        '''
        Sum calculated activated units M_A and fatigued units (M_F)
        :return:
        '''
        return self.data['test_values']['calculated_values'] + self.data['test_values']['calculated_fatigue_values']

    def get_error(self) -> float:
        test_error = (self.data['test_values']['calculated_values'] - self.data['test_values']['real_values']) ** 2
        return test_error.sum() / self.data['test_values']['num_minutes']

    def get_player_matches(self) -> List[PlayerMatchMeasurement]:
        player_matches = self.data['match_values']
        matches_sorted_by_play_time = sorted(player_matches, key=itemgetter('num_minutes'), reverse=True)
        return matches_sorted_by_play_time


def scale_value_to_range(value: float, lower_bound: float, higher_bound: float) -> float:
    return (value - lower_bound) / (higher_bound - lower_bound)


def cost_function_sec(input_vector: np.array, player_data_processor: BaseDataProcessor, m0: float) -> Tuple[int, float]:
    mfr_descriptor = MuscleFatigueRecovery(
        alpha=input_vector[0],
        beta=input_vector[1],
        F=input_vector[2],
        R=input_vector[3],
        M0=m0,
    )
    try:
        player_data_processor.calculate_euler_values(mfr_descriptor)
    except ComputationError as err:
        return 1e30 / err.step

    calc_mp = mfr_descriptor.M0 - player_data_processor.get_units_in_use()
    min_mp = np.min(calc_mp)
    # Constraint m_a + m_f - m_p <= 0
    min_mp = -min_mp
    # round error so early stopping criteria is activated
    data_error = int(player_data_processor.get_error())
    return data_error, min_mp
