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
from src.models.constants import NUM_RUNS, BOUNDS_WORK_ALPHA, BOUNDS_WORK_BETA, BOUNDS_WORK_F, BOUNDS_GAMMA

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
        all_runs_stage1_df = pd.DataFrame()
        all_runs_stage2_df = pd.DataFrame()
        # work or speed
        m0 = player_dh.data['total_w']
        for j in range(NUM_RUNS):
            log(f'Run {j + 1} phase 1 {save_folder_name}')
            run_dict = create_run_dict(run_num=j)
            lb_vector = [BOUNDS_WORK_ALPHA[0], BOUNDS_WORK_BETA[0], BOUNDS_WORK_F[0], BOUNDS_GAMMA[0]]
            ub_vector = [BOUNDS_WORK_ALPHA[1], BOUNDS_WORK_BETA[1], BOUNDS_WORK_F[1], BOUNDS_GAMMA[1]]

            optimizer = setup_pso_optimizer(
                run_dict=run_dict, lb_vector=lb_vector, ub_vector=ub_vector,
                player_data_processor=player_dh, m0=m0
            )
            result: CandidateState = optimizer.optimize()
            # First step, determine optimal alpha and beta for F,gamma = 0
            # init score here
            init_list = run_dict['X'][0]
            mfr_init = MuscleFatigueRecovery(
                M0=m0, alpha=init_list[0], beta=init_list[1], F=init_list[2], gamma=init_list[3]
            )

            # create results
            it_results = IndagoResults(
                x=result.X,
                fun=result.f,
                allvecs=run_dict['X']
            )

            mfr_phase1_results = MuscleFatigueRecovery(
                M0=m0, alpha=result.X[0], beta=result.X[1], F=result.X[2], gamma=result.X[3]
            )
            run_phase1_df = create_run_results_df(
                run_count=j,
                init_mfr=mfr_init,
                num_steps=optimizer.it,
                num_eval=optimizer.eval,
                mfr_results=mfr_phase1_results,
                fin_cost_fun=result.f
            )
            opt_var_vect_phase1_df = create_opt_var_df(run_dict, columns=['alpha', 'beta', 'F', 'gamma'])
            opt_var_vect_phase1_df.to_csv(
                os.path.join(save_path, f'run_phase1_{j}_opt_var_steps_{save_folder_name}.csv'), index=False
            )

            all_runs_stage1_df = pd.concat([all_runs_stage1_df, run_phase1_df])
            all_runs_stage1_df.to_csv(
                os.path.join(save_path, f'all_runs_stage1_{save_folder_name}_df.csv'), index=False
            )
            log(f'Run {j + 1} phase 2 {save_folder_name}')
            # ------------------- Now tweak F and gamma parameters according to the found alpha and beta
            run_dict_stage2 = create_run_dict(run_num=j)
            lb_vector_stage2 = [BOUNDS_WORK_F[0], BOUNDS_GAMMA[0]]
            ub_vector_stage2 = [BOUNDS_WORK_F[1], BOUNDS_GAMMA[1]]
            optimizer_2nd = setup_pso_optimizer(
                run_dict=run_dict_stage2, lb_vector=lb_vector_stage2, ub_vector=ub_vector_stage2,
                player_data_processor=player_dh, m0=m0,
                fix_alpha_beta=(mfr_phase1_results.alpha, mfr_phase1_results.beta)
            )
            result_2nd: CandidateState = optimizer_2nd.optimize()

            init_list_stage2 = run_dict_stage2['X'][0]
            mfr_2nd_init = MuscleFatigueRecovery(
                M0=m0, alpha=mfr_phase1_results.alpha, beta=mfr_phase1_results.beta,
                F=init_list_stage2[0], gamma=init_list_stage2[1]
            )

            # create results
            it_results_stage2 = IndagoResults(
                x=result_2nd.X,
                fun=result_2nd.f,
                allvecs=run_dict_stage2['X']
            )

            mfr_phase2_results = MuscleFatigueRecovery(
                M0=m0, alpha=mfr_phase1_results.alpha, beta=mfr_phase1_results.beta,
                F=result_2nd.X[0], gamma=result_2nd.X[1]
            )
            run_phase2_df = create_run_results_df(
                run_count=j,
                init_mfr=mfr_2nd_init,
                num_steps=optimizer_2nd.it,
                num_eval=optimizer_2nd.eval,
                mfr_results=mfr_phase2_results,
                fin_cost_fun=result_2nd.f
            )
            opt_var_vect_phase2_df = create_opt_var_df(run_dict_stage2, columns=['F', 'gamma'])
            opt_var_vect_phase2_df.to_csv(
                os.path.join(save_path, f'run_phase2_{j}_opt_var_steps_{save_folder_name}.csv'), index=False
            )
            all_runs_stage2_df = pd.concat([all_runs_stage2_df, run_phase2_df])

            if j == 0:
                end2 = time.time()
                hours, minutes, seconds = get_duration_hour_min_sec(start=start, end=end2)
                log(f'Time for one iteration all: {hours}h {minutes}min {seconds}s')
            all_runs_stage2_df.to_csv(
                os.path.join(save_path, f'all_runs_stage2_{save_folder_name}_df.csv'), index=False
            )
        end = time.time()
        hours, minutes, seconds = get_duration_hour_min_sec(start=start, end=end)
        log(f'Time for optimisation: {hours}h {minutes}min {seconds}s')
