# Spectral-method-for-SE_d (ASE)
1.The code is the implementation of spectral method (ASE) for special Euclidean group, i.e., the numerical experiment for ASE comparison with the Two-Stage Approach under gaussian noise assumption.

2.`Appendix.pdf` provides the proof details of lemmas in our article.

**Structure of files**
---
    .
    │
    ├── algorithm
    │   ├── Spectral.py
    │   ├── gpm_sync.py
    ├── compared_experiment_results_with_trans
    ├── compared_experiment_results_without_trans
    ├── experiment_figures  
    ├── plot_functions
        ├── experiment_figure_plot_1.py
    ├── experiment.py
    ├── experiment1.py
---
# Run
Please run `experiment1.py`. You will get the data gernerated in `compared_experiment_results_without_trans` file or `compared_experiment_results_with_trans` file.

For figures, we provide some data we generated for this experiment in the two files. Please run the `experiment_figure_plot_1.py` directly, you will reproduce the figure in our article. Or you could modified the data files in the `experiment_figure_plot_1.py` and get new figure of your own data.
