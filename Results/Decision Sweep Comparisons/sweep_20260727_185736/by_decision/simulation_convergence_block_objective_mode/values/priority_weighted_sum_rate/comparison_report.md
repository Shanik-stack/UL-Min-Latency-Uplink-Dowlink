# Decision Value Comparison: Downlink convergence objective = inverse_cnr_weighted_sum_rate

- Decision path: `simulation.convergence_block_objective_mode`
- Completed runs: `2` / `2`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | 18.6234 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | 18.6234 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |

- Best final total latency: `ndir-asc__obj-invcnr` -> `0.040267`
- Best latency configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best latency reduction: `ndir-asc__obj-invcnr` -> `86.7978%`
- Best reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best final asynchronality: `ndir-asc__obj-invcnr` -> `0.013467`
- Best asynchronality configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc__obj-invcnr` -> `89.7670%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending`