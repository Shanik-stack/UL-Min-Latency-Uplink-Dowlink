# Decision Comparison: BS-shared fixed-target n handling

- Decision path: `simulation.bs_shared_net_fixed_target_n_target_mode`
- Total runs touching this decision: `16`
- Completed runs: `16`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| per_user_n_targets | best_final_total_latency | downlink | convergence | downlink_fixed_block_targets.yaml | upd-net__loss-plain__obj-blend__scope-bs__ntgt-usern | 0.035667 | 43.6842 | bs_shared_net_fixed_target_n_target_mode=per_user_n_targets; constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| per_user_n_targets | best_latency_reduction_percent | downlink | convergence | downlink_fixed_block_targets.yaml | upd-net__loss-plain__obj-blend__scope-bs__ntgt-usern | 0.035667 | 43.6842 | bs_shared_net_fixed_target_n_target_mode=per_user_n_targets; constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| shared_n_targets | best_final_total_latency | downlink | convergence | downlink_fixed_block_targets.yaml | upd-net__loss-plain__obj-blend__scope-bs__ntgt-jointn | 0.035667 | 43.6842 | bs_shared_net_fixed_target_n_target_mode=shared_n_targets; constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| shared_n_targets | best_latency_reduction_percent | downlink | convergence | downlink_fixed_block_targets.yaml | upd-net__loss-plain__obj-blend__scope-bs__ntgt-jointn | 0.035667 | 43.6842 | bs_shared_net_fixed_target_n_target_mode=shared_n_targets; constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |

## Value folders

- `per_user_n_targets` -> `values/per_user_n_targets`
- `shared_n_targets` -> `values/shared_n_targets`