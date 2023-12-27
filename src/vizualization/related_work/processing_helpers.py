import pandas as pd


def add_group_count_id(df: pd.DataFrame) -> pd.DataFrame:
    i = 1
    group_cnt = []
    for current, last in zip(df.is_raising.values, df.is_raising.shift(1).fillna(True)):
        if last != current:
            i += 1
        group_cnt.append(i)
    df.loc[:, 'group_cnt'] = group_cnt
    return df


def create_custom_m_ad(df: pd.DataFrame, y_param: str, use_min_grouping=True) -> pd.DataFrame:
    pd.options.mode.chained_assignment = None
    df.loc[:, f'last_{y_param}'] = df[y_param].shift(1).fillna(0)
    df.loc[:, 'is_raising'] = df[y_param] > df[f'last_{y_param}']
    df.loc[:, 'is_raising_shift1'] = df.is_raising.shift(1).fillna(True)
    df = add_group_count_id(df)

    if use_min_grouping:
        df.loc[:, 'm_ad'] = df.apply(
            lambda r: df[df.group_cnt == r.group_cnt][y_param].min() if not r.is_raising else
            df[df.group_cnt == r.group_cnt][y_param].max(),
            axis=1
        )
    else:
        df.loc[:, 'm_ad'] = df.apply(
            lambda r: r[y_param] if not r.is_raising else
            df[df.group_cnt == r.group_cnt][y_param].max(),
            axis=1
        )
    return df
