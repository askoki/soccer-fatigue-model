import functools
from indago import PSO, Optimizer
from src.features.optimisation.processing import cost_function_sec, BaseDataProcessor
from src.features.optimisation.results_helpers import post_iteration_processing
from src.features.typing import RunDict


def create_run_dict(run_num: int) -> RunDict:
    return {
        'run': run_num,
        'X': [],
        'f': []
    }


def setup_pso_optimizer(run_dict: RunDict, lb_vector: list, ub_vector: list, player_data_processor: BaseDataProcessor,
                        target_fitness: float, m0: float) -> Optimizer:
    optimizer = PSO()
    post_iteration_function = functools.partial(post_iteration_processing, iteration_dict=run_dict)
    optimizer.post_iteration_processing = post_iteration_function
    optimizer.number_of_processes = 'max'
    optimizer.monitoring = 'none'
    # optimizer.max_iterations = 1
    # optimizer.target_fitness = 5
    optimizer.params['swarm_size'] = 10
    optimizer.target_fitness = target_fitness
    # optimizer.max_elapsed_time = 600
    optimizer.max_stalled_iterations = 15

    optimizer.dimensions = len(lb_vector)
    optimizer.lb = lb_vector
    optimizer.ub = ub_vector
    optimizer.constraints = 1
    optimizer.constraint_labels = ['$M_P$']

    evaluation_function = functools.partial(
        cost_function_sec,
        player_data_processor=player_data_processor,
        m0=m0
    )
    optimizer.evaluation_function = evaluation_function
    return optimizer
