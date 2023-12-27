import os
import pandas as pd
from settings import PSO_DIR, REPORTS_DIR
from src.features.utils import set_save_path_arg_parser

folder_path = set_save_path_arg_parser()
all_runs_df = pd.DataFrame()
for player_dir in os.listdir(os.path.join(PSO_DIR, folder_path)):
    if player_dir.endswith('.csv'):
        # Skip resulting files which are created with this script
        continue
    player_all_runs_df = pd.read_csv(
        os.path.join(PSO_DIR, folder_path, player_dir, f'all_runs_{folder_path}_df.csv')
    )

    # player name is the same like player dir
    player_all_runs_df.loc[:, 'player'] = player_dir
    all_runs_df = pd.concat([all_runs_df, player_all_runs_df])
all_runs_df.reset_index(inplace=True, drop=True)
best_runs_df = all_runs_df.sort_values('final_cost_function').groupby('player').agg('first')
best_runs_df = best_runs_df.reset_index()

all_runs_df.to_csv(os.path.join(REPORTS_DIR, PSO_DIR, folder_path, f'pso_all_runs_df.csv'), index=False)
best_runs_df.to_csv(os.path.join(REPORTS_DIR, PSO_DIR, folder_path, f'pso_best_run_df.csv'), index=False)
