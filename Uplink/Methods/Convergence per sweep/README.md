Convergence-per-epoch baseline

This now follows the same overall convergence style as the downlink baseline:

- one shared precoder net per uplink user
- one online convergence run per seed
- block-by-block optimization inside that run
- optimize the precoder at `n = T`
- if the requested bits are feasible, try smaller `n_kl` values with a fresh warm-started
  re-optimization at each candidate
- no separate training/testing split for this method

Entry point:

`python "Uplink\\Methods\\Convergence per epoch\\main.py" --cfg_name uplink_payload_completion.yaml --seed 3`

This baseline now uses one shared seed for one convergence run. The legacy
`--train_seed` and `--test_seed` arguments are accepted only when they match the
same shared seed value.

The main user-facing stop budget is `simulation.max_epochs`. The solve accepts
as soon as one epoch satisfies the KKT tolerances; otherwise it stops at the
best feasible or best-primal state found within that budget.

Results are written under:

`Results\\Uplink\\Convergence per epoch\\<experiment_name>`

Standard files in `data/`:

- `result.json`
- `summary.txt`
- `convergence_results.json`
- `convergence_results.txt`
