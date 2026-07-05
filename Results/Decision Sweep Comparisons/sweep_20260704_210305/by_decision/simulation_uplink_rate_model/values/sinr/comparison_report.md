# Decision Value Comparison: Uplink rate model = sinr

- Decision path: `simulation.uplink_rate_model`
- Completed runs: `12` / `12`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-sinr | 0.031667 | 15.4804 | -3.3104 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-plain__rate-sinr | 0.031733 | 15.3025 | -3.3350 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_payload_completion.yaml | upd-net__loss-aug__rate-sinr | 0.031733 | 15.3025 | -3.3410 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=sinr |
| uplink | convergence | uplink_payload_completion.yaml | upd-net__loss-plain__rate-sinr | 0.031733 | 15.3025 | -3.3637 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-aug__rate-sinr | 0.035400 | 14.6302 | -3.3204 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-plain__rate-sinr | 0.035400 | 14.6302 | -3.3204 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-net__loss-aug__rate-sinr | 0.035400 | 14.6302 | -3.3366 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-net__loss-plain__rate-sinr | 0.035400 | 14.6302 | -3.3230 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=precoder_net; uplink_rate_model=sinr |
| uplink | monte_carlo | uplink_payload_completion.yaml | loss-aug__rate-sinr | 0.039533 | -5.5160 | -3.3831 | 6000 | constraint_loss_form=augmented_lagrangian; uplink_rate_model=sinr |
| uplink | monte_carlo | uplink_payload_completion.yaml | loss-plain__rate-sinr | 0.039533 | -5.5160 | -3.3828 | 6000 | constraint_loss_form=plain_lagrangian; uplink_rate_model=sinr |
| uplink | monte_carlo | uplink_fixed_block_targets.yaml | loss-aug__rate-sinr | 0.042467 | -2.4116 | -3.3850 | 6000 | constraint_loss_form=augmented_lagrangian; uplink_rate_model=sinr |
| uplink | monte_carlo | uplink_fixed_block_targets.yaml | loss-plain__rate-sinr | 0.042533 | -2.5723 | -3.3847 | 6000 | constraint_loss_form=plain_lagrangian; uplink_rate_model=sinr |

- Best final total latency: `upd-dir__loss-aug__rate-sinr` -> `0.031667`
- Best latency configuration: `constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr`
- Best latency reduction: `upd-dir__loss-aug__rate-sinr` -> `15.4804%`
- Best reduction configuration: `constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr`