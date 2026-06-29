Converge-in-each-sweep baseline

This now follows the same overall convergence style as the downlink baseline:

- one shared precoder net per uplink user
- one online convergence run per seed
- block-by-block optimization inside that run
- optimize the precoder at `n = T`, then keep that beam fixed while checking smaller
  `n_kl` values for the tail block, matching the downlink convergence flow
- no separate training/testing split for this method

Entry point:

`python "Uplink\\Methods\\Convergence per sweep\\main.py" --cfg_name uplink_payload_completion.yaml --seed 3`

This baseline now uses one shared seed for one convergence run. The legacy
`--train_seed` and `--test_seed` arguments are accepted only when they match the
same shared seed value.

By default, the convergence wrapper caps the effective inner sweeps per
`(user, block, n_kl)` state so runtime stays in line with the downlink
baseline. The main user-facing cap for that is
`simulation.main_solve_max_sweeps`.

Results are written under:

`Results\\Uplink\\Convergence per sweep\\<experiment_name>`

Standard files in `data/`:

- `result.json`
- `summary.txt`
- `convergence_results.json`
- `convergence_results.txt`
