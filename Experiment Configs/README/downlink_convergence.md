# Downlink Convergence Config Guide

This guide applies to the downlink convergence baseline under:

- `Downlink\Methods\Convergence per epoch`
- `Downlink\Methods\Convergence per sweep`

The two entry points share the same cleaned config surface.

## Decision Variables

- `simulation.experiment_scenario.mode`
  - `payload_completion`
    - `test.B[k]` is the total payload for user `k`.
  - `fixed_block_targets`
    - `test.B[k]` is the target bits for user `k` in every block.
    - Blocks are independent and unserved bits do not carry over.

- `simulation.convergence_precoder_update_mode`
  - `precoder_net`
    - The convergence solver updates neural precoder parameters online.
    - If you choose this, `downlink_precoder_net_scope` also matters.
  - `direct_precoder`
    - The solver updates the active block precoders directly.
    - If you choose this, `downlink_precoder_net_scope` is ignored.

- `simulation.downlink_precoder_net_scope`
  - `per_user_nets`
    - One separate model per user.
    - Each model outputs only that user's beam.
    - Coupling appears later through the shared power budget, interference, and
      outer joint evaluation.
  - `bs_shared_net`
    - One shared model outputs the full BS precoder for the whole block.
    - The full BS precoder is then split into user parts.
    - This is the cleaner architectural choice when you want the network itself
      to represent one joint BS action.
  - This switch matters only when
    `convergence_precoder_update_mode: precoder_net`.

- `simulation.convergence_block_objective_mode`
  - `unweighted_sum_rate`
    - All active users contribute equally to the block objective.
  - `remaining_bits_weighted_sum_rate`
    - Users with larger remaining backlog or future target load get larger
      weight.
  - `blended_network_rate`
    - Mixes plain sum rate with a weighted network-rate term.

- `simulation.constraint_loss_form`
  - `plain_lagrangian`
    - Standard Lagrangian form.
  - `augmented_lagrangian`
    - Adds quadratic penalties on active violations.

- `simulation.n_kl_reduction_update_scope`
  - `all_active_users`
    - Re-optimize all active users if a reduced `n_kl` candidate breaks
      committed-user feasibility.
  - `infeasible_users_only`
    - Re-optimize only the users that became infeasible.
  - `candidate_and_infeasible_users`
    - Re-optimize the user that tried the smaller `n_kl` plus any infeasible
      users.

## Numeric Variables By Theme

### Optimization Budget And Local Update Scale

- `simulation.max_epochs`
  - Maximum number of block-optimization epochs for one constrained solve.

- `simulation.print_every_epoch`
  - Console logging interval only.

- `simulation.user_update_steps`
  - Number of local neural updates per epoch when using
    `convergence_precoder_update_mode: precoder_net`.

- `simulation.user_update_lr`
  - Learning rate for those neural updates.
  - Ignored in `direct_precoder` mode.

### KKT Stopping

- `simulation.kkt_primal_tol`
  - Tolerance on primal feasibility.

- `simulation.kkt_complementarity_tol`
  - Tolerance on complementarity.

- `simulation.kkt_stationarity_tol`
  - Tolerance on relative beam change.
  - Smaller values require the block precoder to settle more before stopping.

### Dual Initialization And Update Strength

- `simulation.initial_lambda_rate_constraint`
  - Initial rate dual variable.

- `simulation.initial_lambda_power_constraint`
  - Initial block-power dual variable.

- `simulation.lr_rate_constraint`
  - Step size for rate-dual updates.

- `simulation.lr_power_constraint`
  - Step size for block-power dual updates.

### Augmented-Lagrangian Penalties

- `simulation.augmented_lagrangian_rho_rate`
  - Quadratic penalty strength on rate violations.

- `simulation.augmented_lagrangian_rho_power`
  - Quadratic penalty strength on block-power violations.

### Objective Weighting

- `simulation.remaining_bits_weight_power`
  - Exponent used when converting backlog into user weights for weighted modes.

- `simulation.minimum_user_weight`
  - Lower bound on each user weight in weighted modes.

- `simulation.network_rate_weight`
  - Strength of the network-level weighted term in
    `blended_network_rate`.

- `simulation.latency_penalty_weight`
  - Penalty weight used by the weighted utility allocation logic.

### Blocklength Search And Run Horizon

- `simulation.n_kl_range.min`
  - Minimum allowed downlink blocklength.

- `simulation.n_kl_range.step`
  - Downward search step for `n_kl`.

- `simulation.max_total_blocks`
  - Maximum block horizon of the outer experiment.

## Scenario Variables

- `simulation.experiment_scenario.payload_bits_source`
  - Used only in `payload_completion`.
  - `system_B` means `test.B` is the full payload vector.

- `simulation.experiment_scenario.skip_infeasible_blocks`
  - Allows partial or zero service when a requested block target is infeasible.

- `simulation.experiment_scenario.skip_block_adds_full_T_latency`
  - If true, a skipped or zero-service block still adds a full block time to
    latency.

- `simulation.experiment_scenario.track_skipped_blocks`
  - Enables skipped-block statistics in saved results.

- `simulation.experiment_scenario.fixed_block_targets.num_blocks`
  - Used only in `fixed_block_targets`.
  - Number of fixed-target blocks.

- `simulation.experiment_scenario.fixed_block_targets.generation_mode`
  - Used only in `fixed_block_targets`.
  - `constant` means each block uses `test.B[k]`.

## Physical System Variables

- `test.K`
  - Number of downlink users.

- `test.T`
  - Maximum blocklength per user.

- `test.B`
  - Payload bits per user in `payload_completion`, or fixed per-block target in
    `fixed_block_targets`.

- `test.P`
  - Full BS block power budget.
  - In the downlink loader this must be scalar-equivalent across users, so the
    YAML usually repeats the same value for each user.

- `test.snr_db`
  - Reference SNR used to set the user noise scale.

- `test.fs`
  - Symbol rate used to convert blocklength into latency.

- `test.epsilon`
  - Target block error probability.

- `test.Nb`
  - Number of base-station transmit antennas.

- `test.Nr`
  - Number of receive antennas at each user.

- `test.initial_bits_per_symbol`
  - Used only for the initial random baseline and initial latency estimate.
  - It does not constrain the optimized block precoder.
