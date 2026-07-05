# Decision Comparison: Constraint loss form

- Decision path: `simulation.constraint_loss_form`
- Total runs touching this decision: `76`
- Completed runs: `76`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| augmented_lagrangian | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| augmented_lagrangian | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-aug__obj-blend__scope-bs | 0.033733 | 88.9399 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |
| plain_lagrangian | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-plain__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| plain_lagrangian | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |

## Value folders

- `augmented_lagrangian` -> `values/augmented_lagrangian`
- `plain_lagrangian` -> `values/plain_lagrangian`