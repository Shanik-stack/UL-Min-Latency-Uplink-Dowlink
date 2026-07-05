# Decision Comparison: Downlink convergence objective

- Decision path: `simulation.convergence_block_objective_mode`
- Total runs touching this decision: `42`
- Completed runs: `42`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| blended_network_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| blended_network_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| remaining_bits_weighted_sum_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-bitsw__scope-bs | 0.034467 | 88.6995 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| remaining_bits_weighted_sum_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-bitsw__scope-bs | 0.034467 | 88.6995 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| unweighted_sum_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-sum__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| unweighted_sum_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-sum__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |

## Value folders

- `blended_network_rate` -> `values/blended_network_rate`
- `remaining_bits_weighted_sum_rate` -> `values/remaining_bits_weighted_sum_rate`
- `unweighted_sum_rate` -> `values/unweighted_sum_rate`