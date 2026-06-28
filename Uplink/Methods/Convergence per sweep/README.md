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

By default, the convergence method also caps the effective inner sweeps per
`(user, block, n_kl)` step to keep runtime in line with the downlink
convergence baseline. You can override that with
`simulation.convergence_max_precoder_sweeps` in the config if needed.

Results are written under:

`Results\\Uplink\\Convergence per sweep\\<experiment_name>`

Standard files in `data/`:

- `result.json`
- `summary.txt`
- `convergence_results.json`
- `convergence_results.txt`
