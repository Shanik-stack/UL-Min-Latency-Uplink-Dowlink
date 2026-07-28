# Decision Value Comparison: n_kl search direction = ascending

- Decision path: `simulation.n_search_direction`
- Completed runs: `4` / `4`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | -3.2712 | 6000 | n_search_direction=ascending |
| uplink | monte_carlo | uplink_payload_completion.yaml | ndir-asc | 0.027467 | 10.6291 | 0.016267 | 2.4000 | -3.4335 | 6000 | n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | 18.6234 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | 8.2492 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |

- Best final total latency: `ndir-asc` -> `0.024667`
- Best latency configuration: `n_search_direction=ascending`
- Best latency reduction: `ndir-asc__obj-invcnr` -> `86.7978%`
- Best reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best final asynchronality: `ndir-asc__obj-unwt` -> `0.003733`
- Best asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc__obj-unwt` -> `97.1631%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`