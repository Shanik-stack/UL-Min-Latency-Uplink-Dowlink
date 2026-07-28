# Decision Value Comparison: n_kl search direction = ascending

- Decision path: `simulation.n_search_direction`
- Completed runs: `7` / `8`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | -3.3125 | 6000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |
| uplink | convergence | uplink_payload_completion.yaml | obj-invcnr__ndir-asc | 0.024733 | 19.5228 | 0.015733 | 5.6000 | -3.3098 | 6000 | n_search_direction=ascending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-invcnr__ndir-asc | 0.027467 | 10.6291 | 0.016267 | 2.4000 | -3.4332 | 6000 | n_search_direction=ascending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.027467 | 10.6291 | 0.016267 | 2.4000 | -3.4335 | 6000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | 8.2492 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.090733 | 70.2514 | 0.037867 | 71.2259 | 6.8631 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.180467 | 40.8306 | 0.066800 | 49.2401 | 2.3859 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr |  |  |  |  |  |  | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |

- Best final total latency: `obj-unwt__ndir-asc` -> `0.024667`
- Best latency configuration: `n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate`
- Best latency reduction: `ndir-asc__obj-unwt` -> `83.8251%`
- Best reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best final asynchronality: `ndir-asc__obj-unwt` -> `0.003733`
- Best asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc__obj-unwt` -> `97.1631%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`