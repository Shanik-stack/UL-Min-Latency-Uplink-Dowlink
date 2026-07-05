# Decision Comparison: Constraint loss form

- Decision path: `simulation.constraint_loss_form`
- Total runs touching this decision: `2`
- Completed runs: `2`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| plain_lagrangian | best_final_total_latency | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-bs__loss-plain | 0.700000 | 30.0000 | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian |
| plain_lagrangian | best_latency_reduction_percent | downlink | monte_carlo | downlink_fixed_block_targets.yaml | scope-bs__loss-plain | 0.700000 | 30.0000 | downlink_precoder_net_scope=bs_shared_net | constraint_loss_form=plain_lagrangian |

## Value folders

- `plain_lagrangian` -> `values/plain_lagrangian`