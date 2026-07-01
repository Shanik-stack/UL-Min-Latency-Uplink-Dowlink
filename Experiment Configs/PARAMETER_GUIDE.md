# Config Parameter Guide

This guide explains the canonical config parameters used by the cleaned
`UL_UPLINK_DOWNLINK_MONTE_CARLO` experiment folder.

The current YAML files are meant to read like experiment definitions, not like
internal training code. Older names such as `precoder_net_train_*`,
`policy_train_*`, and the old "curriculum" fields are treated as legacy aliases
by the loaders, but they are not the recommended names anymore.

## How To Read The Configs

Each config has two top-level sections:

- `test`
  - Defines the physical system and payload setup.
- `simulation`
  - Defines how optimization, Monte Carlo training, and the outer experiment run.

## `test` Parameters

These parameters define the channel dimensions, payload, and timing.

### Shared meaning

- `K`
  - Number of users.
  - Larger `K` usually makes the problem harder because more users must be served.

- `T`
  - Maximum blocklength available to each user in one block.
  - Larger `T` makes rate feasibility easier.

- `B`
  - User bit budget.
  - In `payload_completion`, this is the full payload for each user.
  - In `fixed_block_targets`, this is the per-block target for each user.

- `P`
  - Per-user transmit power budget.
  - Larger values usually improve rate feasibility.

- `snr_db`
  - Per-user reference SNR used to set the noise scale.
  - Larger values mean easier links.

- `fs`
  - Symbol rate used to convert transmitted symbols into latency.
  - Larger `fs` reduces latency for the same total symbol count.

- `epsilon`
  - Block error probability target.
  - Smaller `epsilon` is stricter and reduces the finite-blocklength rate.

### Uplink-specific `test` fields

- `Nr`
  - Number of receive antennas at the base station.

- `Nt`
  - Number of transmit antennas at each uplink user.

### Downlink-specific `test` fields

- `Nb`
  - Number of base-station transmit antennas.

- `Nr`
  - Number of receive antennas at each user.

- `initial_bits_per_symbol`
  - Used only for the initial random baseline and initial latency estimate.
  - It does not directly constrain the optimized or learned precoder.

## Uplink `simulation` Parameters

### Constrained uplink solve

- `convergence_precoder_update_mode`
  - Chooses what the convergence uplink baseline updates inside each constrained solve.
  - `precoder_net`: optimize the uplink precoder-net weights online, then extract the user precoder from the net.
  - `direct_precoder`: directly optimize the complex user precoder for that user and block, without updating a neural network.

- `initial_lambda_rate_constraint`
  - Initial rate dual variable.
  - Larger values push the solver harder toward rate feasibility from the start.

- `initial_lambda_power_constraint`
  - Initial power dual variable.
  - Larger values penalize power overshoot more strongly at the start.

- `max_epochs`
  - Shared optimization budget for one constrained uplink solve.
  - The same cap is used for the first solve at `n = T` and for any smaller-`n_kl` re-solves.
  - The solver stops earlier if one epoch satisfies the KKT tolerances.

- `lr_net`
  - Learning rate for the uplink precoder net during inner constrained solves.
  - Too large can make the inner solve oscillate.
  - Too small can make it stall.

- `lr_rate_constraint`
  - Dual update step size for the rate constraint.
  - Larger values enforce the rate condition more aggressively.

- `lr_power_constraint`
  - Dual update step size for the power constraint.
  - Larger values enforce the power condition more aggressively.

- `constraint_loss_form`
  - Choice of constrained loss shaping.
  - `plain_lagrangian`: basic Lagrangian form.
  - `augmented_lagrangian`: adds quadratic penalty on positive constraint violation.

- `augmented_lagrangian_rho_rate`
  - Quadratic rate-violation penalty strength when `constraint_loss_form: augmented_lagrangian`.

- `augmented_lagrangian_rho_power`
  - Quadratic power-violation penalty strength when `constraint_loss_form: augmented_lagrangian`.

- `print_every_epoch`
  - Logging frequency for inner solves.
  - Affects console output only.

- `kkt_primal_tol`
  - Tolerance on the primal residual.
  - This controls how close the rate and power constraints must be to satisfied.

- `kkt_complementarity_tol`
  - Tolerance on the complementarity residual.
  - This controls how close the dual-weighted constraint residuals must be to zero.

- `kkt_stationarity_tol`
  - Tolerance on the stationarity residual.
  - In the current code this is based on relative beam change between epochs.
  - Smaller values demand a more settled beam before stopping.
  - If the beam has already settled under this tolerance but primal infeasibility remains, the solver rejects that candidate early instead of using the full `max_epochs` budget.

- `reduced_n_kl_log_interval`
  - Logging interval when the code scans smaller `n_kl` values after a feasible main solve.
  - Affects only console verbosity.

### Uplink Monte Carlo training data

- `monte_carlo_training_blocks_per_seed`
  - Number of channel blocks sampled per training seed.
  - Larger values increase dataset diversity and runtime.

- `monte_carlo_training_fallback_target_bits`
  - Bit target used when building Monte Carlo training cases for payload-completion scenarios that do not already provide explicit per-block targets.
  - In your current setup this is usually `1`.

- `monte_carlo_training_n_kl_coarse_step`
  - Coarse step used when probing the feasible `n_kl` frontier during rollout query generation.
  - Larger values speed up data generation but sample the frontier more sparsely.

### Uplink experiment structure

- `max_total_blocks`
  - Maximum number of blocks the outer experiment may create for one user.
  - Prevents very long runs when payloads are hard to drain.

- `uplink_rate_model`
  - Selects how the finite-blocklength rate is computed.
  - `snr`: uses only user noise variance.
  - `sinr`: uses interference-plus-noise covariance.

- `n_kl_range.min`
  - Minimum allowed blocklength candidate.

- `n_kl_range.step`
  - Downward search step for `n_kl`.
  - Smaller values are more precise but slower.

## Downlink `simulation` Parameters

### Per-user beam updates inside one epoch

- `convergence_precoder_update_mode`
  - Chooses what the convergence downlink baseline updates inside each constrained block solve.
  - `precoder_net`: update the downlink precoder net online, then read the active-user precoders from that net.
  - `direct_precoder`: directly update the active block precoders themselves, without updating a neural network.

- `max_epochs`
  - Shared epoch ceiling used by the downlink constrained block solver.
  - The same cap is used for the first solve at the full blocklengths and for any reduced-`n_kl` repair solve.
  - The solver stops earlier if one epoch satisfies the KKT tolerances.

- `downlink_precoder_net_scope`
  - Selects whether the downlink net parameterization is per user or shared at the BS.
  - This matters only when `convergence_precoder_update_mode: precoder_net`.
  - In `direct_precoder` mode, the solver directly updates the active block precoders, so there is no neural-net scope to choose.

- `print_every_epoch`
  - Logging interval for block epochs.

- `user_update_steps`
  - Number of local parameter updates performed for one user model when that user is visited in an epoch.
  - Larger values make each epoch stronger but slower.

- `user_update_lr`
  - Learning rate for those local user-model updates.

### Constrained downlink block solve

- `initial_lambda_rate_constraint`
  - Initial rate dual variable for each active user.

- `initial_lambda_power_constraint`
  - Initial power dual variable for each active user.

- `lr_rate_constraint`
  - Dual update step size for rate feasibility.

- `lr_power_constraint`
  - Dual update step size for power feasibility.

- `constraint_loss_form`
  - `plain_lagrangian` or `augmented_lagrangian`.

- `augmented_lagrangian_rho_rate`
  - Quadratic rate-violation penalty strength used only in augmented-Lagrangian mode.

- `augmented_lagrangian_rho_power`
  - Quadratic power-violation penalty strength used only in augmented-Lagrangian mode.

- `kkt_primal_tol`
  - Primal residual tolerance.

- `kkt_complementarity_tol`
  - Complementarity residual tolerance.

- `kkt_stationarity_tol`
  - Beam-change tolerance for KKT stopping.
  - In downlink, this is based on the largest relative beam update in the active set.
  - If the active-set beams have already settled under this tolerance but the primal residual is still above tolerance, the solver rejects that candidate early instead of spending the whole `max_epochs` budget.

### Downlink reduced-`n_kl` repair behavior

- `n_kl_reduction_update_scope`
  - Controls which user models are updated when a smaller `n_kl` candidate breaks committed-user feasibility.
  - `all_active_users`: repair all currently active users.
  - `infeasible_users_only`: repair only the users whose committed bits became infeasible.
  - `candidate_and_infeasible_users`: repair the user that tried the smaller `n_kl` plus any infeasible users.

### Downlink Monte Carlo training data

- `monte_carlo_training_blocks_per_seed`
  - Number of channel blocks sampled per training seed.

- `monte_carlo_training_fallback_target_bits`
  - Bit target used in payload-completion training episodes when there is no explicit block target.
  - In the current setup this is typically `1`.

- `monte_carlo_training_n_kl_coarse_step`
  - Coarse blocklength step used by rollout frontier probing.
  - Larger values reduce training-data generation cost.

### Downlink objective shaping

- `convergence_block_objective_mode`
  - Objective used by the downlink convergence baseline.
  - `user_rate`: local user-rate style objective.
  - `weighted_sum_rate`: sum rate with remaining-bits weighting.
  - `blended_network_rate`: blends pure sum rate with weighted network rate.

- `remaining_bits_weight_power`
  - Exponent used when turning remaining payload into a user priority weight.
  - Larger values emphasize users with larger remaining payload.

- `minimum_user_weight`
  - Lower bound on each user's weight so low-payload users are not ignored.

- `network_rate_weight`
  - Weight used by the blended network-rate objective.
  - Larger values put more emphasis on network-level weighted service.

- `latency_penalty_weight`
  - Penalty used by the weighted utility allocator when that allocator is selected.
  - Larger values push the allocator toward smaller blocklength choices.

### Downlink experiment structure

- `max_total_blocks`
  - Maximum block horizon allowed by the experiment.

- `n_kl_range.min`
  - Minimum allowed downlink blocklength candidate.

- `n_kl_range.step`
  - Downward step for blocklength search.

## `experiment_scenario` Parameters

These parameters define how the outer experiment interprets `test.B`.

- `mode`
  - `payload_completion` or `fixed_block_targets`.

- `skip_infeasible_blocks`
  - Allows the experiment to leave a block partially or fully unserved when the requested bits are not feasible.

- `skip_block_adds_full_T_latency`
  - If `true`, time still advances by a full block when a block is skipped or yields zero service.

- `track_skipped_blocks`
  - Enables skipped-block statistics in saved summaries.

### `payload_completion`

- `payload_bits_source`
  - `system_B` means the total payload comes directly from `test.B`.
  - `explicit` allows an explicit payload vector inside the scenario section.

### `fixed_block_targets`

- `fixed_block_targets.num_blocks`
  - Number of blocks in the fixed-horizon experiment.

- `fixed_block_targets.generation_mode`
  - `constant`, `explicit`, or `uniform_integer`.
  - `constant` means each block target is `test.B[k]` for user `k`.

- `fixed_block_targets.values`
  - Explicit user-by-block target matrix when `generation_mode: explicit`.

- `fixed_block_targets.min_bits`
  - Minimum candidate per-block target in `uniform_integer` mode.

- `fixed_block_targets.max_bits`
  - Maximum candidate per-block target in `uniform_integer` mode.

- `fixed_block_targets.step_bits`
  - Spacing between candidate targets in `uniform_integer` mode.

## Practical Tuning Notes

- If uplink convergence runs too long, first lower `max_epochs`.
- If uplink KKT stop fires too early, lower `kkt_stationarity_tol`.
- If uplink Monte Carlo samples the `n_kl` frontier too sparsely, lower `monte_carlo_training_n_kl_coarse_step`.
- If downlink repair solves are too expensive, lower `max_epochs` or use `n_kl_reduction_update_scope: infeasible_users_only`.
- If downlink weighted scheduling is too aggressive toward large backlogs, lower `remaining_bits_weight_power`.
- If fixed-block-target runs frequently leave bits unserved, lower `test.B`, increase `T`, or increase `P`.

## Legacy Alias Notes

The loaders still accept older names so older experiment files do not immediately
break. The main legacy aliases are:

- `epochs_per_n_kl` -> `max_epochs`
- `solve_epochs_per_n_kl` -> `max_epochs`
- `max_precoder_epochs` -> `max_epochs`
- `main_solve_guard_sweeps` -> `max_epochs`
- `main_solve_max_epochs` -> `max_epochs`
- `repair_solve_guard_sweeps` -> `max_epochs`
- `reduced_n_kl_repair_max_epochs` -> `max_epochs`
- `print_every_reduced_n_kl` -> `reduced_n_kl_log_interval`
- `precoder_net_train_blocks_per_seed` -> `monte_carlo_training_blocks_per_seed`
- `precoder_net_train_min_bits_required` -> `monte_carlo_training_fallback_target_bits`
- `precoder_net_train_n_kl_coarse_step` -> `monte_carlo_training_n_kl_coarse_step`
- `safe_sweep_objective_mode` -> `convergence_block_objective_mode`
- `queue_weight_power` -> `remaining_bits_weight_power`
- `queue_weight_min` -> `minimum_user_weight`
- `network_weight_beta` -> `network_rate_weight`
- `utility_latency_penalty` -> `latency_penalty_weight`
- `reduced_n_kl_reoptimization_scope` -> `n_kl_reduction_update_scope`

The removed downlink "curriculum" settings are not part of the canonical config
surface anymore because the current cleaned Monte Carlo implementation does not
use them.
