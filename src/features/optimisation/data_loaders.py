import os
import pickle

import pandas as pd
from scipy.interpolate import interp1d

from settings import PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from src.features.file_helpers import create_dir
from src.features.typing import PlayerTestData, PlayerMatchMeasurement
from src.features.utils import log


class DataHolder:
    player_measurements: dict = {}

    def __init__(self):
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, '10x80m_test_anonymized.csv'))
        df_match = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'matches_anonymized.csv'))
        all_players = df.player_name.unique()
        for player in all_players:
            p_df = df[df.player_name == player]
            p_match_df = df_match[df_match.player_name == player]
            create_dir(INTERIM_DATA_DIR)
            pickle_save_path = os.path.join(INTERIM_DATA_DIR, f'{player}_measurements_10x80m.pck')
            try:
                with open(pickle_save_path, 'rb') as file_handle:
                    player_measurement = pickle.load(file_handle)
                self.player_measurements[player] = player_measurement
                continue
            except FileNotFoundError:
                pass
            p_df.loc[:, 'm_ad_ms'] = p_df.m_ad / 3.6
            p_df.loc[:, 'speed_ms'] = p_df.speed_kmh / 3.6
            p_df.loc[:, 'm_ad_w'] = 0.5 * p_df.weight * (df.m_ad_ms ** 2)
            p_df.loc[:, 'm_ad_w'] = p_df.m_ad_w.astype(int)
            p_df.loc[:, 'w'] = 0.5 * p_df.weight * (df.speed_ms ** 2)
            p_df.loc[:, 'w'] = p_df.w.astype(int)

            player_weight = p_df.weight.iloc[0]
            self.player_measurements[player]: PlayerTestData = {
                'test_values': None,
                'max_speed_ms': p_df.speed_ms.max(),
                'weight': player_weight,
                'total_w': int(0.5 * player_weight * p_match_df.speed_ms.max() ** 2),
            }
            m_ad_interp = interp1d(p_df.second.values, p_df.m_ad_w.values, kind='next', fill_value='extrapolate')
            p_df = p_df.drop_duplicates(subset='second')
            player_dict: PlayerMatchMeasurement = {
                'seconds': p_df.second.values,
                'm_ad': m_ad_interp,
                'real_values': p_df.w.values,
                'calculated_values': [],
                'calculated_fatigue_values': [],
                'num_minutes': int(p_df.second.max() / 60)
            }
            self.player_measurements[player]['test_values'] = player_dict
            self.player_measurements[player]['match_values'] = []
            self.player_measurements[player]['total_match_minutes'] = 0

            # Add matches
            for idx, match_df in p_match_df.groupby('date'):
                match_df.loc[:, 'previous_t'] = match_df.second.shift(1).fillna(-1)
                match_df.loc[:, 'dt'] = match_df.second - match_df.previous_t

                # Trim halftime gap
                ht_gap = match_df[match_df.dt > 100].squeeze()
                if ht_gap.shape[0] > 0:
                    match_df.loc[match_df.second >= ht_gap.second, 'second'] -= ht_gap['dt']
                num_minutes = int((match_df.second.max() - match_df.second.min()) / 60)

                m_ad_match_interp = interp1d(match_df.second.values, match_df.m_ad_w.values, kind='linear', fill_value='extrapolate')
                match_df = match_df.drop_duplicates(subset='second')
                match_dict: PlayerMatchMeasurement = {
                    'seconds': match_df.second.values,
                    'm_ad': m_ad_match_interp,
                    'real_values': match_df.w.values,
                    'calculated_values': [],
                    'calculated_fatigue_values': [],
                    'num_minutes': num_minutes
                }
                self.player_measurements[player]['match_values'].append(match_dict)
                self.player_measurements[player]['total_match_minutes'] += num_minutes
            log(f'Saving {player} data to disk.')
            with open(pickle_save_path, 'wb') as file_handle:
                pickle.dump(self.player_measurements[player], file_handle)

    def get_players(self) -> list:
        return [*self.player_measurements.keys()]

    def get_player_data(self, player_name: str) -> PlayerTestData:
        return self.player_measurements[player_name]
