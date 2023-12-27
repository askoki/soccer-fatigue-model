import os


def create_dir(dir_path: str):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
