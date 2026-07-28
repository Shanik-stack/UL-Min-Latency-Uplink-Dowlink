# Uplink Monte Carlo Config Guide

This guide applies to the uplink training + testing method under:

- `Uplink\Methods\Monte Carlo`

## Decision Variables

- `simulation.experiment_scenario.mode`
  - `payload_completion`
    - Train and test on full-payload episodes.
  - `fixed_block_targets`
    - Train and test on independent fixed-target blocks.

- `simulation.constraint_loss_form`
  - `plain_lagrangian`
    - Uses the standard Lagrangian training loss.
  - `augmented_lagrangian`
    - Adds quadratic penalties on active constraint violations.

- `simulation.uplink_rate_model`
  - `snr`
    - Uses only the user's own noise term in rate evaluation.
  - `sinr`
    - Uses interference-plus-noise covariance in rate evaluation.

There is no downlink-style `bs_shared_net` versus `per_user_nets` switch here.
The uplink Monte Carlo method already uses one precoder net per user.

## Numeric Variables By Theme

### Monte Carlo Seed Coverage

- `simulation.monte_carlo_test_seed`
  - Default test seed if `--test_seed` is not passed on the command line.

- `simulation.monte_carlo_num_train_seeds`
  - Default upper bound for automatic training-seed generation.
  - If this is `N`, the default training seeds become `1, 2, ..., N` with the
    test seed removed if it falls inside that range.
  - One training seed produces one channel episode per user.

- `simulation.monte_carlo_train_seeds`
  - Optional explicit training-seed list.
  - If present, it overrides `monte_carlo_num_train_seeds`.

### Monte Carlo Dataset Structure

- `simulation.monte_carlo_training_max_epochs`
  - Maximum Monte Carlo training epochs before the trainer falls back to the
    best feasible or best-primal checkpoint for each user net.
  - The trainer also stops early with the same KKT-style rule used by the
    uplink convergence baseline, using:
    - `simulation.kkt_primal_tol`
    - `simulation.kkt_complementarity_tol`
    - `simulation.kkt_stationarity_tol`

- `simulation.monte_carlo_training_fallback_target_bits`
  - Fallback target bits used when building payload-completion training cases
    that do not already come with explicit block targets.
  - In the current cleaned setup this is usually `1`.

- `simulation.monte_carlo_training_full_block_weight`
  - Total loss weight assigned to full-block rate-maximization states inside
    one training episode.
  - This is the phase where the user beam is trained at `n_kl = T_k`.

- `simulation.monte_carlo_training_tail_feasible_weight`
  - Total loss weight assigned to feasible tail states where the same user can
    finish its remaining bits with smaller `n_kl`.

- `simulation.monte_carlo_training_tail_frontier_weight`
  - Total loss weight assigned to the first rejected tail state beyond the
    feasible `n_kl` frontier.
  - Increasing this sharpens the learned feasible boundary.

- `simulation.monte_carlo_rollout_query_weighting_mode`
  - Controls whether rollout queries are reweighted before the training loss is
    formed.
  - `phase_balanced`: current behavior. Queries are balanced by rollout phase,
    using the three phase weights above.
  - `uniform_per_query`: disables rollout `n_kl` weighting. Every visited query
    contributes with weight `1.0`.

### Training Loss And Dual Variables

- `simulation.initial_lambda_rate_constraint`
  - Initial rate dual variable used inside the Monte Carlo Lagrangian loss.

- `simulation.initial_lambda_power_constraint`
  - Initial power dual variable used inside the Monte Carlo Lagrangian loss.

- `simulation.lr_rate_constraint`
  - Step size for the rate-dual updates during training.

- `simulation.lr_power_constraint`
  - Step size for the power-dual updates during training.

- `simulation.constraint_loss_form`
  - Selects plain versus augmented Lagrangian training.

- `simulation.augmented_lagrangian_rho_rate`
  - Rate-violation quadratic penalty strength for augmented-Lagrangian mode.

- `simulation.augmented_lagrangian_rho_power`
  - Power-violation quadratic penalty strength for augmented-Lagrangian mode.

### Blocklength Search And Outer Run Horizon

- `simulation.n_kl_range.min`
  - Minimum allowed candidate blocklength.

- `simulation.n_kl_range.step`
  - Base blocklength increment used by Monte Carlo `n_kl` search.

- `simulation.n_search_direction`
  - Monte Carlo respects the first search tier for uplink payload rollouts and testing.
  - `descending`: visit smaller `n_kl` values from the full-block side.
  - `ascending`: visit from the minimum side upward until feasibility appears.

- `simulation.n_search_strategy`
  - Monte Carlo currently supports `fixed_step` only.
  - This keeps the rollout dataset aligned with the actual visited `n_kl` states.

- `simulation.n_search_coarse_step`
  - Present for consistency with convergence configs, but not used by the current Monte Carlo rollout.

- `simulation.n_search_exponential_factor`
  - Present for consistency with convergence configs, but not used by the current Monte Carlo rollout.

- `simulation.max_total_blocks`
  - Safety cap on how many blocks a test episode may create.

## Shared YAML Fields Ignored By This Method

These fields belong to the uplink convergence baseline and do not affect the
Monte Carlo method:

- `simulation.convergence_precoder_update_mode`
- `simulation.lr_net`
- `simulation.reduced_n_kl_log_interval`

## Method Launch Variables

These are method launch arguments. They override the YAML when passed:

- `--precoder_net_epochs`
  - Overrides `simulation.monte_carlo_training_max_epochs`.

- `--precoder_net_batch_size`
  - Batch size for training.

- `--precoder_net_lr`
  - Learning rate for the optimizer that trains the uplink user nets.

## Scenario Variables

- `simulation.experiment_scenario.payload_bits_source`
  - Used only in `payload_completion`.
  - `system_B` means the total payload comes from `test.B`.

- `simulation.experiment_scenario.skip_infeasible_blocks`
  - Allows partial or zero service on infeasible blocks.

- `simulation.experiment_scenario.skip_block_adds_full_T_latency`
  - If true, a skipped block still adds a full block time to latency.

- `simulation.experiment_scenario.track_skipped_blocks`
  - Enables skipped-block statistics in saved outputs.

- `simulation.experiment_scenario.fixed_block_targets.num_blocks`
  - Used only in `fixed_block_targets`.
  - Number of blocks in the fixed-target experiment.

- `simulation.experiment_scenario.fixed_block_targets.generation_mode`
  - Used only in `fixed_block_targets`.
  - `constant` means each block uses `test.B[k]`.

## Physical System Variables

- `test.K`
  - Number of uplink users.

- `test.T`
  - Maximum blocklength per user.

- `test.B`
  - Full payload per user in `payload_completion`, or fixed per-block target in
    `fixed_block_targets`.

- `test.P`
  - Per-user uplink transmit power budget.

- `test.snr_db`
  - Reference SNR used to set the noise level.

- `test.fs`
  - Symbol rate used in latency conversion.

- `test.epsilon`
  - Target block error probability.

- `test.Nt`
  - Number of transmit antennas at each uplink user.

- `test.Nr`
  - Number of receive antennas at the base station.
