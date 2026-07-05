# Decision Value Comparison: Convergence precoder update mode = direct_precoder

- Decision path: `simulation.convergence_precoder_update_mode`
- Completed runs: `20` / `20`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-snr | 0.024667 | 19.7397 | -3.3732 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-plain__rate-snr | 0.024667 | 19.7397 | -3.3731 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-aug__rate-snr | 0.027333 | 17.5050 | -3.3367 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-plain__rate-snr | 0.027333 | 17.5050 | -3.3366 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-aug__rate-sinr | 0.031667 | 15.4804 | -3.3104 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_payload_completion.yaml | upd-dir__loss-plain__rate-sinr | 0.031733 | 15.3025 | -3.3350 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-aug__rate-sinr | 0.035400 | 14.6302 | -3.3204 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| uplink | convergence | uplink_fixed_block_targets.yaml | upd-dir__loss-plain__rate-sinr | 0.035400 | 14.6302 | -3.3204 | 6000 | constraint_loss_form=plain_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=sinr |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-plain__obj-blend | 0.036400 | 88.0656 | 5.4602 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-plain__obj-sum | 0.036400 | 88.0656 | 5.4007 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-aug__obj-bitsw | 0.037067 | 87.8470 | 6.5863 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-aug__obj-blend | 0.037800 | 87.6066 | 6.2500 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-plain__obj-blend | 0.038267 | 39.5789 | 6.2974 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-plain__obj-bitsw | 0.038467 | 39.2632 | 6.0070 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-plain__obj-sum | 0.038467 | 39.2632 | 6.0070 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-aug__obj-sum | 0.038533 | 87.3661 | 5.4128 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_payload_completion.yaml | upd-dir__loss-plain__obj-bitsw | 0.038667 | 87.3224 | 4.8818 | 6000 | constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-aug__obj-blend | 0.039067 | 38.3158 | 6.3454 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-aug__obj-bitsw | 0.039667 | 37.3684 | 6.0928 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=remaining_bits_weighted_sum_rate; convergence_precoder_update_mode=direct_precoder |
| downlink | convergence | downlink_fixed_block_targets.yaml | upd-dir__loss-aug__obj-sum | 0.039667 | 37.3684 | 6.0928 | 6000 | constraint_loss_form=augmented_lagrangian; convergence_block_objective_mode=unweighted_sum_rate; convergence_precoder_update_mode=direct_precoder |

- Best final total latency: `upd-dir__loss-aug__rate-snr` -> `0.024667`
- Best latency configuration: `constraint_loss_form=augmented_lagrangian; convergence_precoder_update_mode=direct_precoder; uplink_rate_model=snr`
- Best latency reduction: `upd-dir__loss-plain__obj-blend` -> `88.0656%`
- Best reduction configuration: `constraint_loss_form=plain_lagrangian; convergence_block_objective_mode=blended_network_rate; convergence_precoder_update_mode=direct_precoder`