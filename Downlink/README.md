Downlink

This downlink subset keeps only:

- `Methods/Convergence per epoch`: the online convergence baseline
- `Methods/Monte Carlo`: the offline blocklength-aware precoder-net path

The extra weighted baselines and their wrappers were intentionally removed from this cleaned copy.

Shared experiment conventions in this cleaned folder:

- entry points use the shared result-tag helper from `experiment_utils.py`
- each run is saved directly under `Results/Downlink/Method-...` and mirrored into `Results/Downlink/Scenario-...`
- results are split into `training/` and `testing/` subfolders inside each experiment folder
- testing summaries live under `testing/data/`
- training artifacts and training summaries live under `training/data/`
- the convergence baseline objective is controlled by `simulation.convergence_block_objective_mode`
- the downlink precoder architecture scope is controlled by `simulation.downlink_precoder_net_scope`
- valid downlink precoder scopes are `per_user_nets` and `bs_shared_net`
- in Monte Carlo fixed-block-target testing, `simulation.bs_shared_net_fixed_target_n_target_mode` selects how `bs_shared_net` reduces `n_{k,l}`
- `shared_n_targets` means one joint block-level `n_{k,l}` vector is searched for the shared BS precoder
- `per_user_n_targets` means the shared BS model keeps an older user-by-user reduction style, but now with round-robin one-step-per-user passes to reduce order bias
- canonical convergence objectives are `unweighted_sum_rate`, direct latency-gap weighting such as `asynchronality_weighted_sum_rate`, concrete proxy-weighted modes such as `inverse_cnr_weighted_sum_rate`, and blended weighted modes such as `blended_asynchronality_weighted_sum_rate`
- `unweighted_sum_rate` means every active user contributes equally to the block objective
- weighted modes mean the sum rate is weighted by a concrete user-priority rule selected either directly in the objective name or through `simulation.convergence_priority_weight_strategy`
- `asynchronality_gap` gives higher weight to users that are projected to finish later, which is the recommended mode when the goal is to reduce asynchronality directly
- `inverse_cnr` gives higher weight to weaker users by using the inverse of their current effective CNR, which is a proxy fairness mode
- `remaining_bits` gives higher weight to users with larger remaining payload or target backlog
- legacy aliases are still accepted in configs: `user_rate -> unweighted_sum_rate`, `equal_priority_sum_rate -> unweighted_sum_rate`, `priority_weighted_sum_rate -> concrete weighted mode selected by convergence_priority_weight_strategy`
- `test.P` is interpreted as one shared BS block-power budget for the full downlink precoder `F_b`, so it should be a scalar or repeated identical values
