# Decision Value Comparison: Uplink rate model = snr

- Decision path: `simulation.uplink_rate_model`
- Completed runs: `12` / `12`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | -3.3732 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-plain__rate-snr | 0.024667 | 19.7397 | -3.3731 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_payload_completion.yaml | upd-net__loss-aug__rate-snr | 0.024667 | 19.7397 | -3.2986 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=snr |
| uplink | convergence | uplink_payload_completion.yaml | upd-net__loss-plain__rate-snr | 0.024667 | 19.7397 | -3.2584 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-aug__rate-snr | 0.027333 | 17.5050 | -3.3367 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-plain__rate-snr | 0.027333 | 17.5050 | -3.3366 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-net__loss-aug__rate-snr | 0.027333 | 17.5050 | -3.3302 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-net__loss-plain__rate-snr | 0.027333 | 17.5050 | -3.3295 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=snr |
| uplink | monte_carlo | uplink_payload_completion.yaml | loss-aug__rate-snr | 0.030867 | -0.4338 | -3.4149 | 6000 | constraint_loss_form=augmented_lagrangian; uplink_rate_model=snr |
| uplink | monte_carlo | uplink_payload_completion.yaml | loss-plain__rate-snr | 0.030867 | -0.4338 | -3.4140 | 6000 | constraint_loss_form=plain_lagrangian; uplink_rate_model=snr |
| uplink | monte_carlo | uplink_fixed_block_targets.yaml | loss-aug__rate-snr | 0.033733 | -1.8109 | -3.3728 | 6000 | constraint_loss_form=augmented_lagrangian; uplink_rate_model=snr |
| uplink | monte_carlo | uplink_fixed_block_targets.yaml | loss-plain__rate-snr | 0.033733 | -1.8109 | -3.3726 | 6000 | constraint_loss_form=plain_lagrangian; uplink_rate_model=snr |

- Best final total latency: `upd-dir__loss-aug__rate-snr` -> `0.024667`
- Best latency configuration: `constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr`
- Best latency reduction: `upd-dir__loss-aug__rate-snr` -> `19.7397%`
- Best reduction configuration: `constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr`