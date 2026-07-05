# Uplink Convergence Config Guide

This guide applies to the uplink convergence baseline under:

- `Uplink\Methods\Convergence per epoch`
- `Uplink\Methods\Convergence per sweep`

The current cleaned setup uses one shared config surface for both entry points.

## Decision Variables

- `simulation.experiment_scenario.mode`
  - `payload_completion`
    - `test.B[k]` is the full payload for user `k`.
    - The run keeps creating blocks until each user's payload is drained or the
      block cap is reached.
  - `fixed_block_targets`
    - `test.B[k]` is the fixed target bits for user `k` in every block.
    - Blocks are independent and unserved bits do not carry forward.

- `simulation.convergence_precoder_update_mode`
  - `precoder_net`
    - The online constrained solve updates the uplink user's precoder net
      weights, then reads the user precoder from the net.
    - Use this if you want the convergence baseline to optimize the same kind
      of neural parameterization used by the learned method.
  - `direct_precoder`
    - The online constrained solve updates the complex uplink precoder itself.
    - Use this if you want the convergence baseline to be a direct numerical
      optimizer rather than a neural-weight optimizer.

- `simulation.constraint_loss_form`
  - `plain_lagrangian`
    - Uses the standard Lagrangian penalty structure.
    - Usually simpler to interpret.
  - `augmented_lagrangian`
    - Adds quadratic penalties on positive constraint violations.
    - Usually stronger when you want the solver to punish persistent
      infeasibility more aggressively.

- `simulation.uplink_rate_model`
  - `snr`
    - Uses only the user's own noise variance in the rate evaluation.
    - This is the cleaner mode if you want the uplink user model to ignore
      multi-user interference in its input and in the main rate equation.
  - `sinr`
    - Uses interference-plus-noise covariance.
    - This is the stricter coupled mode.

## Numeric Variables By Theme

### Optimization Budget And Logging

- `simulation.max_epochs`
  - Maximum number of epochs allowed inside one constrained uplink solve.
  - This same cap is used for the main solve at `n = T` and any smaller-`n_kl`
    re-solves.

- `simulation.print_every_epoch`
  - Console logging interval only.
  - It changes verbosity, not the optimization result.

- `simulation.reduced_n_kl_log_interval`
  - Logging interval while smaller `n_kl` values are being scanned after a
    feasible solution exists.

### KKT Stopping

- `simulation.kkt_primal_tol`
  - Tolerance on primal feasibility.
  - Smaller values demand tighter rate and power feasibility before stopping.

- `simulation.kkt_complementarity_tol`
  - Tolerance on complementarity.
  - Smaller values demand tighter alignment between dual variables and active
    violations.

- `simulation.kkt_stationarity_tol`
  - Tolerance on the relative beam-change stopping check.
  - Smaller values require the uplink precoder to settle more before stopping.

### Dual Initialization And Update Strength

- `simulation.initial_lambda_rate_constraint`
  - Starting value of the rate dual variable.

- `simulation.initial_lambda_power_constraint`
  - Starting value of the power dual variable.

- `simulation.lr_rate_constraint`
  - Step size for rate-dual updates.

- `simulation.lr_power_constraint`
  - Step size for power-dual updates.

### Online Neural Update Scale

- `simulation.lr_net`
  - Learning rate for the uplink precoder net inside convergence mode when
    `convergence_precoder_update_mode: precoder_net`.
  - This is ignored in `direct_precoder` mode.

### Augmented-Lagrangian Penalties

- `simulation.augmented_lagrangian_rho_rate`
  - Rate-violation quadratic penalty strength.
  - Used only when `constraint_loss_form: augmented_lagrangian`.

- `simulation.augmented_lagrangian_rho_power`
  - Power-violation quadratic penalty strength.
  - Used only when `constraint_loss_form: augmented_lagrangian`.

### Blocklength Search And Run Horizon

- `simulation.n_kl_range.min`
  - Minimum allowed blocklength candidate.

- `simulation.n_kl_range.step`
  - Downward search step for `n_kl`.
  - Smaller values give finer blocklength search but cost more runtime.

- `simulation.max_total_blocks`
  - Maximum number of outer blocks the run may create.
  - This is mainly a safety cap for long payload-draining runs.

## Scenario Variables

- `simulation.experiment_scenario.payload_bits_source`
  - Used only in `payload_completion`.
  - `system_B` means the total payload comes directly from `test.B`.

- `simulation.experiment_scenario.skip_infeasible_blocks`
  - If true, the experiment may leave a block partially or fully unserved when
    the requested service is infeasible.

- `simulation.experiment_scenario.skip_block_adds_full_T_latency`
  - If true, an infeasible or skipped block still adds a full block's time to
    latency.

- `simulation.experiment_scenario.track_skipped_blocks`
  - Enables skipped-block statistics in saved summaries.

- `simulation.experiment_scenario.fixed_block_targets.num_blocks`
  - Used only in `fixed_block_targets`.
  - Number of independent blocks in the fixed-target run.

- `simulation.experiment_scenario.fixed_block_targets.generation_mode`
  - Used only in `fixed_block_targets`.
  - `constant` means every block uses `test.B[k]` as the user target.

## Physical System Variables

- `test.K`
  - Number of uplink users.

- `test.T`
  - Maximum blocklength per user.

- `test.B`
  - User payload or per-block target, depending on scenario mode.

- `test.P`
  - Per-user uplink transmit power budget.

- `test.snr_db`
  - Reference link SNR used to set the noise scale.

- `test.fs`
  - Symbol rate used to convert symbols into latency.

- `test.epsilon`
  - Target block error probability.

- `test.Nt`
  - Number of transmit antennas at each uplink user.

- `test.Nr`
  - Number of receive antennas at the base station.
