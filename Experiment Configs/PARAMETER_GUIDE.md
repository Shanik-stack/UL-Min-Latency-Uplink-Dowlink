# Config Parameter Guide

This guide explains the parameters used by the unified `UL_UPLINK_DOWNLINK_MONTE_CARLO` experiment configs and how changing them affects the simulation.

## Uplink Convergence Stop Rule

The uplink convergence baseline stops the inner precoder update loop for one `(user, block, n_kl)` state only when all of the following are true:

1. The state is feasible within tolerance.
2. The current precoder changed only a little from the previous sweep.
3. The minimum required number of sweeps has already been completed.

The exact checks are:

- Feasibility check:
  - `rate_violation_pos <= convergence_feasibility_tol`
  - `power_violation_pos <= convergence_feasibility_tol`
- Beam-change check:
  - `||F^(t) - F^(t-1)||_F <= convergence_precoder_tol`
- Minimum sweep check:
  - `t >= convergence_min_precoder_sweeps_before_stop`

These checks are applied in:

- `Uplink/Optimizer_per_block.py`
- `Uplink/Methods/Convergence per sweep/optimizer.py`

### Convergence Sweep Cap

The convergence wrapper does not always use the raw `epochs_per_n_kl` value directly. It applies an effective cap:

`effective_sweeps = min(epochs_per_n_kl, max_precoder_sweeps, convergence_max_precoder_sweeps)`

This means:

- `epochs_per_n_kl` is the requested inner-loop budget.
- `max_precoder_sweeps` is the general hard ceiling.
- `convergence_max_precoder_sweeps` is a smaller convergence-only ceiling so the baseline does not inherit very large Monte Carlo training budgets.

## Uplink Net Inputs

The current uplink code now uses only local-user information as neural-network input:

- Convergence uplink precoder net:
  - `H_k,l`, `sigma_k^2`, `epsilon_k`
- Monte Carlo uplink precoder net:
  - `H_k,l`, `n_k,l`, `sigma_k^2`, `epsilon_k`
- Monte Carlo shared-beam uplink net:
  - `H_k,l`, `sigma_k^2`, `epsilon_k`

The uplink rate evaluation itself is now configurable:

- `uplink_rate_model: snr`
  - Uses only `sigma_k^2 I` in the finite-blocklength rate equation.
  - Other users do not affect the rate calculation.

- `uplink_rate_model: sinr`
  - Uses the full interference-plus-noise covariance.
  - Other users affect the rate calculation through the covariance, but their channels are still not fed into the network input.

## Common `test` Parameters

These fields define the physical system and payload setup.

### Uplink `test`

- `K`
  - Number of uplink users.
  - Increasing it usually raises interference and makes feasibility harder.

- `Nr`
  - Number of receive antennas at the base station for each user link.
  - Larger `Nr` usually improves receive diversity and achievable rate.

- `Nt`
  - Number of transmit antennas per uplink user.
  - Larger `Nt` increases precoder dimension and model size.

- `T`
  - Maximum blocklength available to each user in one block.
  - Larger `T` makes feasibility easier because rate dispersion penalty shrinks.

- `B`
  - User payload size in bits.
  - Larger `B` increases total latency in payload-completion experiments.

- `P`
  - Power budget per uplink user.
  - Larger `P` usually helps rate but can increase inter-user interference.

- `snr_db`
  - Target per-user SNR used when calibrating the fixed noise variance `sigma2`.
  - Larger `snr_db` means smaller noise variance and easier links.

- `fs`
  - Symbol rate per user.
  - Larger `fs` reduces latency for the same total transmitted symbols.

- `epsilon`
  - Block error probability target.
  - Smaller `epsilon` is stricter and reduces finite-blocklength rate.

- `initial_bits_per_symbol`
  - Used only when building the initial random baseline and initial latency estimate.
  - It does not directly constrain the trained precoder net.

- `f_carrier`, `v`
  - Used only by the dynamic-channel uplink initialization path.
  - Higher carrier frequency or user speed generally produces faster channel variation.

### Downlink `test`

- `K`
  - Number of downlink users.

- `Nb`
  - Number of base-station transmit antennas.
  - Larger `Nb` usually makes beamforming more flexible.

- `Nr`
  - Number of receive antennas per user.
  - Larger `Nr` can improve receive combining gain.

- `T`
  - Maximum blocklength per user.

- `B`
  - Total payload bits per user.

- `P`
  - Power budget associated with each user stream.

- `snr_db`
  - Used to set the noise scale seen by each user.

- `fs`
  - Symbol rate for latency conversion.

- `epsilon`
  - Finite-blocklength reliability target.

- `initial_bits_per_symbol`
  - Used for the initial random baseline and initial latency estimate.

## Uplink `simulation` Parameters

These are read by the uplink config loader and then used by the convergence and Monte Carlo methods.

- `initial_lambda_rate_constraint`
  - Initial Lagrange multiplier for rate feasibility.
  - Larger values penalize rate violations more strongly at the start.

- `initial_lambda_power_constraint`
  - Initial Lagrange multiplier for power feasibility.
  - Larger values penalize power overshoot more strongly at the start.

- `epochs_per_n_kl`
  - Requested number of optimizer sweeps for one fixed `(user, block, n_kl)` state.
  - Larger values give more time to settle but increase runtime.

- `lr_net`
  - Adam learning rate for uplink precoder-net parameters.
  - Too large can cause oscillation; too small can stall updates.

- `lr_rate_constraint`
  - Step size for updating the rate Lagrange multiplier.
  - Larger values enforce rate feasibility more aggressively.

- `lr_power_constraint`
  - Step size for updating the power Lagrange multiplier.
  - Larger values enforce the power limit more aggressively.

- `max_precoder_sweeps`
  - General ceiling on inner optimization sweeps.
  - Prevents very long runs even if `epochs_per_n_kl` is large.

- `print_every_sweep`
  - Logging frequency during iterative optimization.
  - Affects console verbosity, not algorithm behavior.

- `precoder_net_train_blocks_per_seed`
  - Number of channel blocks sampled per training seed when building uplink Monte Carlo training data.
  - Larger values increase dataset diversity and runtime.

- `precoder_net_train_min_bits_required`
  - Minimum positive bit target used in the rollout dataset.
  - In your current preferred setup this is typically `1`.

- `precoder_net_train_n_kl_coarse_step`
  - Coarse decrement used when exploring candidate `n_kl` values in rollout-style training data generation.
  - Larger values reduce runtime but sample the `n_kl` frontier more sparsely.

- `step_lr`
  - Generic step-size field used by some older iterative routines.
  - It is mainly a fallback when a method-specific learning rate is not given.

- `user_update_steps`
  - Number of local user-update substeps in iterative baselines that support it.
  - Larger values spend more work per outer sweep.

- `user_update_lr`
  - Learning rate for those local user-update substeps.
  - Larger values react faster but can become unstable.

- `max_total_blocks`
  - Hard cap on how many blocks may be created in one user simulation.
  - Larger values allow long payloads to continue instead of stopping early.

- `convergence_max_precoder_sweeps`
  - Extra sweep cap used only by the uplink convergence baseline.
  - Lower values speed up the baseline at the cost of less inner convergence.

- `convergence_min_precoder_sweeps_before_stop`
  - Minimum sweeps that must happen before the convergence stop rule is allowed to fire.
  - Increasing it makes the baseline less eager to stop.

- `convergence_precoder_tol`
  - Threshold for the beam-change stop test.
  - Smaller values require tighter beam stabilization before stopping.

- `convergence_feasibility_tol`
  - Threshold for accepting small residual rate or power violations as effectively feasible.
  - Smaller values are stricter and may require more sweeps.

- `uplink_rate_model`
  - Selects which covariance is used inside the uplink finite-blocklength rate equation.
  - `snr` uses only the user noise variance `sigma_k^2 I`.
  - `sinr` uses interference-plus-noise covariance from the other active users.
  - This changes feasibility checks, payload reduction, `n_kl` search, and Monte Carlo training targets.

- `n_kl_range.min`
  - Smallest allowed blocklength candidate.
  - Lower values allow more aggressive latency reduction but make feasibility harder.

- `n_kl_range.step`
  - Decrement step when scanning downward in `n_kl`.
  - Larger values speed up search but may skip the true minimum feasible `n_kl`.

## Downlink `simulation` Parameters

These are used by the downlink convergence and Monte Carlo methods.

- `max_precoder_sweeps`
  - Maximum number of safe-sweep refinement sweeps per block.

- `print_every_sweep`
  - How often sweep progress is printed.

- `step_lr`
  - Step size used by the direct downlink safe-sweep updates.

- `user_update_steps`
  - Number of inner updates per active user inside one sweep.
  - Higher values make each sweep stronger but slower.

- `user_update_lr`
  - Learning rate for those user-local updates.

- `precoder_tol`
  - Stopping threshold for downlink beam change in safe-sweep refinement.
  - Smaller values require tighter convergence.

- `max_total_blocks`
  - Maximum block horizon allowed by the experiment.

- `precoder_net_train_blocks_per_seed`
  - Number of channel blocks sampled per training seed for downlink Monte Carlo data generation.

- `precoder_net_train_n_kl_coarse_step`
  - Coarse blocklength decrement used when building the training frontier.

- `precoder_net_train_min_bits_required`
  - Minimum positive target bits used during training data generation.

- `precoder_net_train_max_reduction_rounds_per_epoch`
  - Maximum number of curriculum reductions of `n_kl` per epoch.
  - Lower values slow the curriculum and keep training closer to easier states.

- `precoder_net_train_curriculum_warmup_epochs`
  - Number of epochs before curriculum reduction begins.
  - Larger values keep training on easier states longer.

- `precoder_net_train_curriculum_interval_epochs`
  - Number of epochs between curriculum reductions.
  - Larger values make the `n_kl` frontier move downward more slowly.

- `precoder_net_train_enumerate_all_masks_up_to_k`
  - Enumerates all active-user masks up to this number of active users when forming training scenarios.
  - Larger values improve mask diversity but increase dataset size quickly.

- `safe_sweep_objective_mode`
  - Downlink block objective used in safe-sweep optimization.
  - Typical choices favor user rate or weighted sum rate.

- `queue_weight_power`
  - Exponent used when turning remaining payload or queue state into user weights.
  - Larger values emphasize users with larger queues.

- `queue_weight_min`
  - Minimum user weight floor.
  - Prevents small-queue users from receiving near-zero priority.

- `network_weight_beta`
  - Controls how strongly network-level weighting affects the downlink objective.
  - Larger values make the objective more queue-aware and less purely rate-driven.

- `utility_latency_penalty`
  - Penalty weight for latency-oriented utility shaping.
  - Larger values bias optimization toward serving delayed users earlier.

- `n_kl_range.min`
  - Minimum downlink blocklength allowed during search or curriculum.

- `n_kl_range.step`
  - Downward step size for blocklength exploration.

## `experiment_scenario` Parameters

These parameters define the outer experiment interpretation and are shared across uplink and downlink.

- `mode`
  - `payload_completion` or `fixed_block_targets`.
  - `payload_completion` keeps sending until each user's payload is drained.
  - `fixed_block_targets` assigns a fixed positive bit target to every block over a fixed horizon.
  - In `fixed_block_targets`, blocks are independent: unmet bits do not carry into the next block.

- `skip_infeasible_blocks`
  - If `true`, the solver is allowed to leave a block unsent when the target is infeasible.
  - If `false`, the method must keep trying to serve that block.
  - In the current fixed-block-target implementation, the solver instead serves as many bits as possible in that block; any unmet bits are recorded and do not carry forward.

- `skip_block_adds_full_T_latency`
  - If `true`, a skipped block still contributes full-block latency.
  - This matches the interpretation that time passed even though no useful payload was delivered.

- `track_skipped_blocks`
  - Enables skipped-block statistics in summaries.

### `payload_completion` Extras

- `payload_bits_source`
  - `system_B` or `explicit`.
  - `system_B` uses the per-user `test.B` payload directly.
  - `explicit` allows a separate payload vector in the scenario config.

- `payload_bits.values`
  - Explicit per-user payload sizes when `payload_bits_source: explicit`.

### `fixed_block_targets` Extras

- `fixed_block_targets.num_blocks`
  - Number of blocks in the fixed-horizon experiment.
  - Larger values increase total simulated time and total target bits.

- `fixed_block_targets.generation_mode`
  - `constant`, `explicit`, or `uniform_integer`.
  - `constant` means the per-block target comes directly from `test.B[k]` for each user.
  - Larger `test.B[k]` values make fixed-target blocks harder to satisfy.
  - If the full target is infeasible, the method serves the maximum feasible bits in that same block and records the unserved remainder.

- `fixed_block_targets.values`
  - Explicit user-by-block target matrix when `generation_mode: explicit`.

- `fixed_block_targets.min_bits`
  - Minimum candidate bits when `generation_mode: uniform_integer`.

- `fixed_block_targets.max_bits`
  - Maximum candidate bits when `generation_mode: uniform_integer`.

- `fixed_block_targets.step_bits`
  - Spacing between candidate bit values when `generation_mode: uniform_integer`.

## Practical Tuning Notes

- If runtime is too long in uplink convergence, first lower `convergence_max_precoder_sweeps`.
- If uplink convergence stops too early, increase `convergence_min_precoder_sweeps_before_stop` or reduce `convergence_precoder_tol`.
- If uplink Monte Carlo over-focuses on very small `n_kl`, increase `precoder_net_train_n_kl_coarse_step` or slow the curriculum logic in the method.
- If downlink Monte Carlo lacks active-mask diversity, increase `precoder_net_train_blocks_per_seed` or `precoder_net_train_enumerate_all_masks_up_to_k`.
- If skipped blocks dominate fixed-block-target experiments, lower `test.B` for those users or raise `T` or `P`.
