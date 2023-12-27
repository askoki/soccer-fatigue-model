from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple



class ComputationError(Exception):
    def __init__(self, message, euler_step=1):
        self.message = message
        self.step = euler_step

    def __str__(self):
        return self.message


def log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S %d/%m/%y')
    print(f'{timestamp}: {message}')


def get_duration_hour_min_sec(start: float, end: float) -> Tuple[float, float, float]:
    hours = (end - start) // 3600
    minutes = ((end - start) - hours * 3600) // 60
    seconds = int((end - start) - hours * 3600 - minutes * 60)
    return hours, minutes, seconds


def set_nm_pso_arg_parser() -> bool:
    parser = ArgumentParser(description='Please provide data source info')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pso', dest='is_pso', action='store_true')
    group.add_argument('--nm', dest='is_pso', action='store_false')

    return parser.parse_args().is_pso


def set_save_path_arg_parser() -> str:
    parser = ArgumentParser(description='Please provide data source info')
    parser.add_argument('--path', dest='path', type=str, required=True)
    return parser.parse_args().path


def set_nm_pso_collect_path_arg_parser() -> Tuple[bool, str]:
    parser = ArgumentParser(description='Please provide data source info')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pso', dest='is_pso', action='store_true')
    group.add_argument('--nm', dest='is_pso', action='store_false')
    parser.add_argument('--path', dest='path', type=str, required=True)

    is_pso = parser.parse_args().is_pso
    folder_path = parser.parse_args().path
    return is_pso, folder_path
