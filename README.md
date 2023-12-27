soccer-fatigue-model
==============================
Repository for replicating results of the paper "Enhancing Biophysical Muscle Fatigue Model in the Dynamic Context of Soccer" by Skoki et al.


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- Anonymized repeated-sprinting test and soccer match data used in the study.
    │   └── raw            <- Soccer-specific test data extracted from the literature.
    │
    ├── reports                  <- Results of optimisation and visualization are saved in this folder.            
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Helper scripts for optimisation and processing
    │   │
    │   ├── models         <- Scripts to start optimisation
    │   │   ├── pso
    │   │   │   └── optimize.py
    │   │   ├── collect_results.py <- generate .csv files in reports folder
    │   │   └── constants.py
    │   │  
    │   ├── reports  <- Scripts to create manuscript tables
    │   │   └── generate_latex_table_opt_results.py
    │   │  
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── related work <- Scripts for plotting on the data extracted from the literature.
    │       ├── man_m_ad_example.py <- Script for plotting an example M_AD situation
    │       ├── plot_one_match_best_params_graph.py
    │       ├── plot_related_limitations.py
    │       └── plot_test_n_matches_best_params.py
    │
    └── settings.py            <- Project related configuration and settings.

--------
