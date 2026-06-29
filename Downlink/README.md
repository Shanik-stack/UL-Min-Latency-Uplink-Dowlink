Downlink

This downlink subset keeps only:

- `Methods/Convergence per sweep`: the greedy safe-sweep baseline
- `Methods/Monte Carlo`: the offline blocklength-aware precoder-net path

The extra weighted baselines and their wrappers were intentionally removed from this cleaned copy.

Shared experiment conventions in this cleaned folder:

- entry points use the shared result-tag helper from `experiment_utils.py`
- results are split into `training/` and `testing/` subfolders inside each experiment folder
- testing summaries live under `testing/data/`
- training artifacts and training summaries live under `training/data/`
- the convergence baseline objective is controlled by `simulation.convergence_block_objective_mode`
