# Decision Comparison: Downlink precoder scope

- Decision path: `simulation.downlink_precoder_net_scope`
- Total runs touching this decision: `40`
- Completed runs: `40`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| bs_shared_net | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| bs_shared_net | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| per_user_nets | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-aug__obj-bitsw__scope-user | 0.035000 | 88.5246 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=per_user_nets |
| per_user_nets | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-aug__obj-bitsw__scope-user | 0.035000 | 88.5246 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=per_user_nets |

## Value folders

- `bs_shared_net` -> `values/bs_shared_net`
- `per_user_nets` -> `values/per_user_nets`