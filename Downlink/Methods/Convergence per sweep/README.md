Convergence-per-epoch baseline

This is the existing downlink online baseline:

- optimize the active users' block precoders with synchronized epochs
- the downlink BS power constraint is enforced on the full block precoder `F_b = [F_{1,b}, F_{2,b}, ..., F_{K,b}]`, not on each user beam separately
- allocate bits with the greedy outer `n_kl` search
- when a user tries a smaller `n_kl`, first test it against the current beams
- only trigger a fresh re-optimization if that smaller `n_kl` breaks committed-user feasibility
- control which users are updated in that repair step with `simulation.n_kl_reduction_update_scope`
  - `all_active_users`: update every active user in the block
  - `infeasible_users_only`: update only the users that became infeasible
  - `candidate_and_infeasible_users`: update the reduced-`n_kl` user and any infeasible users
- recompute SINR using the committed block beams
- choose the convergence block objective from `simulation.convergence_block_objective_mode`
  - supported canonical modes: `unweighted_sum_rate`, `remaining_bits_weighted_sum_rate`, `blended_network_rate`
  - `remaining_bits_weighted_sum_rate` weights each active user's rate by its current remaining-bit backlog
  - legacy aliases still load: `user_rate -> unweighted_sum_rate`, `weighted_sum_rate -> remaining_bits_weighted_sum_rate`

Entry point:

`python "Downlink\\Methods\\Convergence per epoch\\main.py" --cfg_name config_downlink_example.yaml --seed 3`

Results are written under:

`Results\\Downlink\\Convergence per epoch\\<experiment_name>`

The objective mode is appended to the experiment tag so runs with different
block objectives do not overwrite each other.

Standard files in `testing/data/`:

- `result.json`
- `summary.txt`
