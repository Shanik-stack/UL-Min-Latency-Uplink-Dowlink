# Decision Value Comparison: n_kl search direction = descending

- Decision path: `simulation.n_search_direction`
- Completed runs: `7` / `8`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-desc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | -3.3022 | 6000 | n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate |
| uplink | convergence | uplink_payload_completion.yaml | obj-invcnr__ndir-desc | 0.024733 | 19.5228 | 0.015733 | 5.6000 | -3.3800 | 6000 | n_search_direction=descending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-invcnr__ndir-desc | 0.024933 | 18.8720 | 0.015333 | 8.0000 | -3.3250 | 6000 | n_search_direction=descending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-unwt__ndir-desc | 0.024933 | 18.8720 | 0.015333 | 8.0000 | -3.3269 | 6000 | n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate |
| downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | 8.2492 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.073600 | 75.8689 | 0.021733 | 83.4853 | 13.0376 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-unwt | 0.075667 | 75.1913 | 0.048267 | 63.3232 | 11.9746 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-invcnr |  |  |  |  |  |  | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |

- Best final total latency: `obj-unwt__ndir-desc` -> `0.024667`
- Best latency configuration: `n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate`
- Best latency reduction: `ndir-desc__obj-unwt` -> `83.8251%`
- Best reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending`
- Best final asynchronality: `ndir-desc__obj-unwt` -> `0.003733`
- Best asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending`
- Best asynchronality reduction: `ndir-desc__obj-unwt` -> `97.1631%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending`