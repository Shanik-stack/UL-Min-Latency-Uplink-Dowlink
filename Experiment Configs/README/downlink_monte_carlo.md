# Downlink Monte Carlo Config Guide

This guide applies to the downlink training + testing method under:

- `Downlink\Methods\Monte Carlo`

## Decision Variables

- `simulation.experiment_scenario.mode`
  - `payload_completion`
    - Train and test on payload-draining episodes.
  - `fixed_block_targets`
    - Train and test on independent fixed-target blocks.

- `simulation.downlink_precoder_net_scope`
  - `per_user_nets`
    - One separate model per user.
    - Each model outputs only that user's beam.
    - Coupling is enforced later through joint power projection and joint
      evaluation.
  - `bs_shared_net`
    - One shared model outputs the full BS precoder for the whole block in one
      forward pass.
    - The user beams are then read as parts of that full BS precoder.
    - This is the cleaner architecture when you want the learned model itself
      to represent the joint BS decision.

- `simulation.bs_shared_net_fixed_target_n_target_mode`
  - Used only when both are true:
    - `simulation.downlink_precoder_net_scope: bs_shared_net`
    - `simulation.experiment_scenario.mode: fixed_block_targets`
  - `shared_n_targets`
    - The shared BS net is evaluated with one joint `n_{k,l}` vector for the
      whole block.
    - Downward `n_{k,l}` search is performed jointly at the block level.
    - This is the cleaner and more physically consistent shared-BS evaluation.
  - `per_user_n_targets`
    - The shared BS net keeps the older user-by-user reduction idea.
    - Only one user's `n_{k,l}` is reduced at a time.
    - The cleaned version now performs that reduction in round-robin one-step
      passes so one user cannot consume many reductions before others are
      checked.
    - This is cheaper and closer to the older behavior, but less jointly
      consistent than `shared_n_targets`.

- `simulation.constraint_loss_form`
  - `plain_lagrangian`
    - Standard Lagrangian training objective.
  - `augmented_lagrangian`
    - Adds quadratic penalties on active violations.

The Monte Carlo method does not use
`simulation.convergence_precoder_update_mode`. That switch belongs to the
convergence baseline, not to offline Monte Carlo training.

## Numeric Variables By Theme

### Monte Carlo Seed Coverage

- `simulation.monte_carlo_test_seed`
  - Default test seed if `--test_seed` is not passed.

- `simulation.monte_carlo_num_train_seeds`
  - Default upper bound for automatic training-seed generation.
  - If this is `N`, the default training seeds are `1, 2, ..., N` excluding
    the test seed if necessary.
  - One training seed produces one joint channel episode.

- `simulation.monte_carlo_train_seeds`
  - Optional explicit training-seed list.
  - Overrides `monte_carlo_num_train_seeds` when present.

### Monte Carlo Dataset Structure

- `simulation.monte_carlo_training_max_epochs`
  - Maximum Monte Carlo training epochs before the trainer falls back to the
    best feasible or best-primal checkpoint.
  - The trainer also stops early with the same KKT-style rule used by the
    convergence baseline, using:
    - `simulation.kkt_primal_tol`
    - `simulation.kkt_complementarity_tol`
    - `simulation.kkt_stationarity_tol`

- `simulation.monte_carlo_training_fallback_target_bits`
  - Fallback target bits used when payload-completion training cases do not
    already provide explicit block targets.
  - In the current setup this is usually `1`.

- `simulation.monte_carlo_training_full_block_weight`
  - Total loss weight assigned to full-block sum-rate states inside one joint
    training episode.

- `simulation.monte_carlo_training_tail_feasible_weight`
  - Total loss weight assigned to feasible tail states where the jointly served
    users can keep their committed bits while reducing `n_kl`.

- `simulation.monte_carlo_training_tail_frontier_weight`
  - Total loss weight assigned to the first rejected joint tail state beyond
    the feasible `n_kl` frontier.

- `simulation.monte_carlo_rollout_query_weighting_mode`
  - Controls whether joint rollout queries are reweighted before Monte Carlo
    training.
  - `phase_balanced`: current behavior. The three rollout phases are balanced
    using the configured phase weights.
  - `uniform_per_query`: disables rollout `n_kl` weighting. Each visited query
    keeps weight `1.0`.

### Lagrangian Training Weights

- `simulation.initial_lambda_rate_constraint`
  - Initial rate dual variable used inside the training objective.

- `simulation.initial_lambda_power_constraint`
  - Initial block-power dual variable used inside the training objective.

- `simulation.lr_rate_constraint`
  - Step size for the rate-dual updates during training.

- `simulation.lr_power_constraint`
  - Step size for the block-power dual updates during training.

- `simulation.augmented_lagrangian_rho_rate`
  - Rate-violation quadratic penalty strength for augmented-Lagrangian mode.

- `simulation.augmented_lagrangian_rho_power`
  - Block-power quadratic penalty strength for augmented-Lagrangian mode.

### Blocklength Search And Run Horizon

- `simulation.n_kl_range.min`
  - Minimum allowed candidate blocklength.

- `simulation.n_kl_range.step`
  - Base blocklength increment used by Monte Carlo `n_kl` search.

- `simulation.n_search_direction`
  - Downlink Monte Carlo keeps a full-block joint anchor and currently expects
    `descending` search in the joint rollout collector.
  - The local payload evaluators still use the same direction field.

- `simulation.n_search_strategy`
  - Monte Carlo currently supports `fixed_step` only.
  - This keeps the rollout dataset tied to the visited joint `n_kl` states.

- `simulation.n_search_coarse_step`
  - Present for consistency with convergence configs, but not used by the current Monte Carlo rollout.

- `simulation.n_search_exponential_factor`
  - Present for consistency with convergence configs, but not used by the current Monte Carlo rollout.

- `simulation.max_total_blocks`
  - Maximum number of blocks the outer test episode may create.

## Shared YAML Fields Ignored By This Method

These fields belong to the downlink convergence baseline and do not affect the
Monte Carlo method:

- `simulation.convergence_precoder_update_mode`
- `simulation.user_update_steps`
- `simulation.user_update_lr`
- `simulation.n_kl_reduction_update_scope`
- `simulation.convergence_block_objective_mode`
- `simulation.remaining_bits_weight_power`
- `simulation.minimum_user_weight`
- `simulation.network_rate_weight`
- `simulation.latency_penalty_weight`

## Method Launch Variables

These are method launch arguments. They override the YAML when passed:

- `--precoder_net_epochs`
  - Overrides `simulation.monte_carlo_training_max_epochs`.

- `--precoder_net_batch_size`
  - Training batch size.

- `--precoder_net_lr`
  - Learning rate used to train the downlink precoder net or shared BS net.

## Scenario Variables

- `simulation.experiment_scenario.payload_bits_source`
  - Used only in `payload_completion`.
  - `system_B` means the total payload comes from `test.B`.

- `simulation.experiment_scenario.skip_infeasible_blocks`
  - Allows partial or zero service on infeasible blocks.

- `simulation.experiment_scenario.skip_block_adds_full_T_latency`
  - If true, a skipped or zero-service block still adds a full block time to
    latency.

- `simulation.experiment_scenario.track_skipped_blocks`
  - Enables skipped-block statistics in saved outputs.

- `simulation.experiment_scenario.fixed_block_targets.num_blocks`
  - Used only in `fixed_block_targets`.
  - Number of independent target blocks.

- `simulation.experiment_scenario.fixed_block_targets.generation_mode`
  - Used only in `fixed_block_targets`.
  - `constant` means each block uses `test.B[k]`.

## Physical System Variables

- `test.K`
  - Number of downlink users.

- `test.T`
  - Maximum blocklength per user.

- `test.B`
  - Total payload in `payload_completion`, or fixed per-block target in
    `fixed_block_targets`.

- `test.P`
  - Full BS block power budget.
  - The loader expects one scalar-equivalent BS power budget, so the YAML
    usually repeats the same value across users.

- `test.snr_db`
  - Reference SNR used to set the user noise scale.

- `test.fs`
  - Symbol rate used in latency conversion.

- `test.epsilon`
  - Target block error probability.

- `test.Nb`
  - Number of base-station transmit antennas.

- `test.Nr`
  - Number of receive antennas at each user.

- `test.initial_bits_per_symbol`
  - Used only for the initial random baseline and initial latency estimate.
