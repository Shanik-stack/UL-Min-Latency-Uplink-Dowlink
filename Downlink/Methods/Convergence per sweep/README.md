Converge-in-each-sweep baseline

This is the existing downlink online baseline:

- optimize the active users' block precoders with synchronized safe sweeps
- allocate bits with the greedy outer `n_kl` search
- recompute SINR using the committed block beams
- choose the safe-sweep objective from `simulation.safe_sweep_objective_mode`
  - supported modes: `user_rate`, `weighted_sum_rate`, `blended_network_rate`

Entry point:

`python "Downlink\\Methods\\Convergence per sweep\\main.py" --cfg_name config_downlink_example.yaml --seed 3`

Results are written under:

`Results\\Downlink\\Convergence per sweep\\<experiment_name>`

The objective mode is appended to the experiment tag so runs with different
safe-sweep objectives do not overwrite each other.

Standard files in `testing/data/`:

- `result.json`
- `summary.txt`
