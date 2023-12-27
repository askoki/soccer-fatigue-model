import os
import pandas as pd
from settings import PSO_DIR, NELDER_MEAD_DIR, REPORTS_DIR
from src.features.utils import set_nm_pso_collect_path_arg_parser

is_pso, folder_path = set_nm_pso_collect_path_arg_parser()
source_path = PSO_DIR if is_pso else NELDER_MEAD_DIR
save_name = 'pso' if is_pso else 'nm'

all_runs_df = pd.DataFrame()
for player_dir in os.listdir(os.path.join(source_path, folder_path)):
    if player_dir.endswith('.csv'):
        # Skip resulting files which are created with this script
        continue
    player_all_runs_df = pd.read_csv(os.path.join(source_path, folder_path, player_dir, f'all_runs_stage2_{folder_path}_df.csv'))
    # Return it to an original format, so it is the same as load_data
    player_all_runs_df.loc[:, 'player'] = player_dir.replace('_', ' ')
    all_runs_df = pd.concat([all_runs_df, player_all_runs_df])
all_runs_df.reset_index(inplace=True, drop=True)
best_runs_df = all_runs_df.sort_values('final_cost_function').groupby('player').agg('first')
best_runs_df = best_runs_df.reset_index()

all_runs_df.to_csv(os.path.join(REPORTS_DIR, source_path, folder_path, f'{save_name}_all_runs_df.csv'), index=False)
best_runs_df.to_csv(os.path.join(REPORTS_DIR, source_path, folder_path, f'{save_name}_best_run_df.csv'), index=False)
