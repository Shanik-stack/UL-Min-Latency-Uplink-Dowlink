# Decision Value Comparison: Downlink convergence objective = inverse_cnr_weighted_sum_rate

- Decision path: `simulation.convergence_block_objective_mode`
- Completed runs: `2` / `4`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.073600 | 75.8689 | 0.021733 | 83.4853 | 13.0376 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.090733 | 70.2514 | 0.037867 | 71.2259 | 6.8631 | 6000 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr |  |  |  |  |  |  | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-invcnr |  |  |  |  |  |  | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |

- Best final total latency: `ndir-desc__obj-invcnr` -> `0.073600`
- Best latency configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending`
- Best latency reduction: `ndir-desc__obj-invcnr` -> `75.8689%`
- Best reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending`
- Best final asynchronality: `ndir-desc__obj-invcnr` -> `0.021733`
- Best asynchronality configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending`
- Best asynchronality reduction: `ndir-desc__obj-invcnr` -> `83.4853%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending`