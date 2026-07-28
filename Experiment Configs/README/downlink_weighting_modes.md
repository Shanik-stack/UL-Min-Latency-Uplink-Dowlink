# Downlink Weighting Modes

This note explains the public downlink convergence objective names and how they
change the optimization behavior.

## Core idea

For one active downlink block with active-user set `A`, the solver always uses
the users' finite-blocklength rates `R_k`.

The difference between weighting modes is only how those per-user rates are
combined into the block objective.

## 1. `unweighted_sum_rate`

Objective:

`J = sum_{k in A} R_k`

Meaning:

- Every active user contributes equally.
- This pushes hardest toward total throughput.
- It does not explicitly favor weak users or users with larger backlog.

Typical effect:

- Often gives strong latency reduction.
- Can leave larger latency imbalance between users if one user is much weaker.

## 2. Weighted sum-rate modes

General objective:

`J = sum_{k in A} w_k R_k`

For most weighted modes, the active-user weights `w_k` are built from a raw
priority score `s_k`, then shaped as:

`normalized_k = (s_k / max_j s_j) ^ p`

`w_k = w_min + (1 - w_min) normalized_k`

where:

- `p = simulation.remaining_bits_weight_power`
- `w_min = simulation.minimum_user_weight`

So, for these modes, the choice of mode decides the raw score `s_k`.

### `asynchronality_weighted_sum_rate`

Raw score:

`s_k = projected_completion_latency_k / min_j projected_completion_latency_j`

where the projected completion latency is estimated from:

- committed latency before the current block
- current pending bits for that user in the visited state
- current achieved bits per block at the visited `n_{k,l}`

Meaning:

- Users that are projected to finish later get larger weight.
- This targets the latency-gap objective directly instead of using weak-channel status as a proxy.
- It is the recommended mode when the goal is to reduce asynchronality itself.

Typical effect:

- More directly pushes completion times together.
- Can still trade off some pure sum rate when one user is far behind.

### `inverse_cnr_weighted_sum_rate`

Weight definition:

`w_k = (1 / CNR_k) / ((1 / |A|) sum_{j in A} (1 / CNR_j))`

with direct channel-based CNR

`CNR_k = channel_gain_k / sigma_k^2`

and

`channel_gain_k = ||H_k||_F^2 / N_{r,k}`

Meaning:

- Weaker users get larger weight directly through `1 / CNR_k`.
- The weights are mean-normalized across the active users, so the average active-user weight is `1`.
- There is no exponent shaping and no minimum-weight floor in this mode.

Typical effect:

- Usually improves latency balance across users.
- Often raises final asynchronality reduction even when total sum rate changes only a little.

### `remaining_bits_weighted_sum_rate`

Raw score:

`s_k = remaining_bits_k / max_j remaining_bits_j`

Meaning:

- Users with more remaining payload get larger weight.
- This is backlog driven, not channel driven.
- It is useful when the main goal is draining outstanding payload faster.

Typical effect:

- Helps the solver focus on users that still have a lot left to send.
- May or may not reduce asynchronality depending on whether the large-backlog users are also the weak users.

### `inverse_channel_gain_weighted_sum_rate`

Raw score:

`s_k = 1 / channel_gain_k`

where `channel_gain_k` is based on the Frobenius norm of the user's raw channel.

Meaning:

- Weaker raw channels get larger weight.
- Unlike inverse-CNR weighting, this does not depend on the current interference state.

Typical effect:

- A more static fairness push than inverse-CNR weighting.
- Useful if you want weakness measured from the raw channel itself.

### `uniform_weighted_sum_rate`

Raw score is effectively constant, so all active users get the same weight.

Meaning:

- This behaves like `unweighted_sum_rate`, but through the weighted code path.
- It is mainly useful for controlled ablations.

## 3. Blended weighted modes

General objective:

`J = sum_{k in A} R_k + beta sum_{k in A} w_k R_k`

where `beta = simulation.network_rate_weight`.

Public names follow the same pattern as the weighted modes, for example:

- `blended_asynchronality_weighted_sum_rate`
- `blended_inverse_cnr_weighted_sum_rate`
- `blended_remaining_bits_weighted_sum_rate`
- `blended_inverse_channel_gain_weighted_sum_rate`
- `blended_uniform_weighted_sum_rate`

Meaning:

- The first term keeps pressure on total network throughput.
- The second term adds a fairness or priority push.
- Larger `network_rate_weight` makes the fairness term matter more.

Typical effect:

- Often sits between pure unweighted throughput and aggressively weighted fairness.

## Which mode to use

- Use `unweighted_sum_rate` if total throughput and shortest total latency are the main goal.
- Use `asynchronality_weighted_sum_rate` if reducing asynchronality is the main goal.
- Use `inverse_cnr_weighted_sum_rate` if you still want a weaker-user proxy rather than a direct latency-gap objective.
- Use `remaining_bits_weighted_sum_rate` if backlog reduction is the main goal.
- Use a blended mode if you want a middle ground between pure throughput and fairness weighting.

## Important note on naming

Older names such as `equal_priority_sum_rate` and `priority_weighted_sum_rate`
are still accepted as legacy aliases, but public summaries and new configs
should use the explicit names above because they say what is actually being
weighted.
