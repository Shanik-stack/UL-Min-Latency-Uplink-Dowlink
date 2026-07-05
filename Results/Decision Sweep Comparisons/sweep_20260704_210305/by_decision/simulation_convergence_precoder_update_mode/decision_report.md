# Decision Comparison: Convergence precoder update mode

- Decision path: `simulation.convergence_precoder_update_mode`
- Total runs touching this decision: `58`
- Completed runs: `58`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| direct_precoder | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| direct_precoder | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-plain__obj-blend | 0.036400 | 88.0656 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder |
| precoder_net | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-net__loss-aug__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=snr |
| precoder_net | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | upd-net__loss-plain__obj-blend__scope-bs | 0.033667 | 88.9617 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=precoder_net; downlink_precoder_net_scope=bs_shared_net |

## Value folders

- `direct_precoder` -> `values/direct_precoder`
- `precoder_net` -> `values/precoder_net`