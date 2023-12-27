import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
NELDER_MEAD_DIR = os.path.join(REPORTS_DIR, 'nelder-mead')
PSO_DIR = os.path.join(REPORTS_DIR, 'pso')
