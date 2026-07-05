# Decision Comparison: Downlink precoder scope

- Decision path: `simulation.downlink_precoder_net_scope`
- Total runs touching this decision: `2`
- Completed runs: `2`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| bs_shared_net | best_final_total_latency | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-bs__loss-plain | 0.700000 | 30.0000 | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian |
| bs_shared_net | best_latency_reduction_percent | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-bs__loss-plain | 0.700000 | 30.0000 | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian |
| per_user_nets | best_final_total_latency | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-user__loss-plain | 0.800000 | 20.0000 | downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian |
| per_user_nets | best_latency_reduction_percent | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-user__loss-plain | 0.800000 | 20.0000 | downlink_precoder_net_scope=per_user_nets | constraint_loss_form=plain_lagrangian |

## Value folders

- `bs_shared_net` -> `values/bs_shared_net`
- `per_user_nets` -> `values/per_user_nets`