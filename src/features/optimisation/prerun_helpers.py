import functools
from indago import PSO, Optimizer
from helpers import post_iteration_processing
from src.features.optimisation.processing import cost_function_sec, BaseDataProcessor
from src.features.typing import RunDict, AlphaBetaArg


def create_run_dict(run_num: int) -> RunDict:
    return {
        'run': run_num,
        'X': [],
        'f': []
    }


def setup_pso_optimizer(run_dict: RunDict, lb_vector: list, ub_vector: list, player_data_processor: BaseDataProcessor,
                        m0: float, fix_alpha_beta=(None, None) or AlphaBetaArg) -> Optimizer:
    optimizer = PSO()
    post_iteration_function = functools.partial(post_iteration_processing, iteration_dict=run_dict)
    optimizer.post_iteration_processing = post_iteration_function
    optimizer.number_of_processes = 'max'
    optimizer.monitoring = 'none'
    # optimizer.max_iterations = 1
    # optimizer.target_fitness = 5
    optimizer.params['swarm_size'] = 10
    optimizer.max_elapsed_time = 600
    optimizer.max_stalled_iterations = 20

    optimizer.dimensions = len(lb_vector)
    optimizer.lb = lb_vector
    optimizer.ub = ub_vector

    evaluation_function = functools.partial(
        cost_function_sec,
        player_data_processor=player_data_processor,
        m0=m0,
        alpha=fix_alpha_beta[0],
        beta=fix_alpha_beta[1]
    )
    optimizer.evaluation_function = evaluation_function
    return optimizer
