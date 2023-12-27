import os
import time
import pandas as pd
from indago import CandidateState

from settings import PSO_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_loaders import DataHolder
from src.features.optimisation.prerun_helpers import create_run_dict, setup_pso_optimizer
from src.features.optimisation.processing import PlayerDataProcessor
from src.features.optimisation.results_helpers import create_opt_var_df, IndagoResults, create_run_results_df
from src.features.typing import MuscleFatigueRecovery
from src.features.utils import log, get_duration_hour_min_sec, set_save_path_arg_parser
from src.models.constants import NUM_RUNS, BOUNDS_WORK_ALPHA, BOUNDS_WORK_BETA, BOUNDS_WORK_F, BOUNDS_WORK_R, BOUNDS_M0

if __name__ == '__main__':
    save_folder_name = set_save_path_arg_parser()
    sprint_data_holder = DataHolder()
    players_list = sprint_data_holder.get_players()
    players_count = len(players_list)

    for i, player_name in enumerate(players_list):
        log(f'Processing player: {player_name} {i + 1}/{players_count}')

        player_save_name = player_name.replace(' ', '_')
        save_path = os.path.join(PSO_DIR, save_folder_name, player_save_name)
        create_dir(save_path)

        # OPTIMISATION PROCESS
        player_dh = PlayerDataProcessor(player_name, sprint_data_holder.get_player_data(player_name))

        start = time.time()
        all_runs_df = pd.DataFrame()
        all_runs_stage2_df = pd.DataFrame()

        avg_energy_per_min = player_dh.data['test_values']['real_values'].sum() / player_dh.data['test_values']['num_minutes']
        # 1% of avg energy per minute
        target_fitness = (0.01 * avg_energy_per_min)**2
        m0 = player_dh.data['total_w']
        for j in range(NUM_RUNS):
            log(f'Run {j + 1} {save_folder_name}')
            run_dict = create_run_dict(run_num=j)
            lb_vector = [BOUNDS_WORK_ALPHA[0], BOUNDS_WORK_BETA[0], BOUNDS_WORK_F[0], BOUNDS_WORK_R[0]]
            ub_vector = [BOUNDS_WORK_ALPHA[1], BOUNDS_WORK_BETA[1], BOUNDS_WORK_F[1], BOUNDS_WORK_R[1]]

            optimizer = setup_pso_optimizer(
                run_dict=run_dict, lb_vector=lb_vector, ub_vector=ub_vector,
                player_data_processor=player_dh, target_fitness=target_fitness, m0=m0
            )
            result: CandidateState = optimizer.optimize()
            # First step, determine optimal alpha and beta for F,gamma = 0
            # init score here
            init_list = run_dict['X'][0]
            mfr_init = MuscleFatigueRecovery(
                alpha=init_list[0], beta=init_list[1], F=init_list[2], R=init_list[3], M0=m0
            )

            # create results
            it_results = IndagoResults(
                x=result.X,
                fun=result.f,
                allvecs=run_dict['X']
            )

            mfr_results = MuscleFatigueRecovery(
                alpha=result.X[0], beta=result.X[1], F=result.X[2], R=result.X[3], M0=m0
            )
            run_df = create_run_results_df(
                run_count=j,
                init_mfr=mfr_init,
                num_steps=optimizer.it,
                num_eval=optimizer.eval,
                mfr_results=mfr_results,
                fin_cost_fun=result.f
            )
            opt_var_vect_df = create_opt_var_df(run_dict, columns=['alpha', 'beta', 'F', 'R'])
            opt_var_vect_df.to_csv(
                os.path.join(save_path, f'run_{j}_opt_var_steps_{save_folder_name}.csv'), index=False
            )

            all_runs_df = pd.concat([all_runs_df, run_df])
            all_runs_df.to_csv(
                os.path.join(save_path, f'all_runs_{save_folder_name}_df.csv'), index=False
            )
        end = time.time()
        hours, minutes, seconds = get_duration_hour_min_sec(start=start, end=end)
        log(f'Time for optimisation: {hours}h {minutes}min {seconds}s')
