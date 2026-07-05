# Decision Comparison: Uplink rate model

- Decision path: `simulation.uplink_rate_model`
- Total runs touching this decision: `24`
- Completed runs: `24`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| sinr | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-sinr | 0.031667 | 15.4804 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| sinr | best_latency_reduction_percent | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-sinr | 0.031667 | 15.4804 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| snr | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| snr | best_latency_reduction_percent | uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |

## Value folders

- `sinr` -> `values/sinr`
- `snr` -> `values/snr`