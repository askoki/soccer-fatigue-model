import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def add_group_count_id(match_df: pd.DataFrame) -> pd.DataFrame:
    i = 1
    group_cnt = []
    for current, last in zip(match_df.is_speed_raising.values, match_df.is_speed_raising.shift(1).fillna(True)):
        if last != current:
            i += 1
        group_cnt.append(i)
    match_df.loc[:, 'group_cnt'] = group_cnt
    return match_df


def create_custom_m_ad_interpolator(seconds: np.array, match_df: pd.DataFrame, player_weight: int,
                                    use_speed: bool) -> interp1d:
    pd.options.mode.chained_assignment = None
    match_df.loc[:, 'last_speed_ms'] = match_df.speed_ms.shift(1).fillna(0)
    match_df.loc[:, 'is_speed_raising'] = match_df.speed_ms > match_df.last_speed_ms
    match_df.loc[:, 'is_speed_raising_shift1'] = match_df.is_speed_raising.shift(1).fillna(True)
    match_df.loc[:, 'seconds_from_entering'] = seconds
    match_df = add_group_count_id(match_df)
    match_df.loc[:, 'should_insert'] = match_df.apply(
        lambda r: True if r.is_speed_raising and not r.is_speed_raising_shift1 else False,
        axis=1
    )
    insert_df = match_df[match_df.should_insert == True]
    insert_df.loc[:, 'seconds_from_entering'] = insert_df.seconds_from_entering - 1
    match_df = pd.concat([match_df, insert_df])
    match_df = match_df.sort_values(['match_date', 'seconds_from_entering', 'group_cnt'])
    # vertical jump to the desired speed
    match_df.loc[:, 'desired_speed_ms'] = match_df.apply(
        lambda r: r.speed_ms if not r.is_speed_raising else match_df[match_df.group_cnt == r.group_cnt].speed_ms.max(),
        axis=1
    )
    # start seconds from 0
    if use_speed:
        return interp1d(match_df.seconds_from_entering, match_df.desired_speed_ms.values, kind='linear',
                        fill_value='extrapolate')
    match_df.loc[:, 'W_requested'] = 0.5 * player_weight * match_df.desired_speed_ms ** 2
    return interp1d(match_df.seconds_from_entering, match_df.W_requested.values, kind='linear',
                    fill_value='extrapolate')


def adjust_real_values(use_speed: bool, speed_arr: np.array, weight: float) -> np.array:
    if use_speed:
        return speed_arr
    return 0.5 * weight * speed_arr ** 2
