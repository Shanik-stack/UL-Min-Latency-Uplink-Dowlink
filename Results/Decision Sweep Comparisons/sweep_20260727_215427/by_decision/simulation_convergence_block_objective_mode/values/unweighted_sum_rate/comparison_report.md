# Decision Value Comparison: Downlink convergence objective = unweighted_sum_rate

- Decision path: `simulation.convergence_block_objective_mode`
- Completed runs: `4` / `4`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | 8.2492 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | 8.2492 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-unwt | 0.075667 | 75.1913 | 0.048267 | 63.3232 | 11.9746 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending |
| downlink | monte_carlo | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.180467 | 40.8306 | 0.066800 | 49.2401 | 2.3859 | 6000 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |

- Best final total latency: `ndir-asc__obj-unwt` -> `0.049333`
- Best latency configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best latency reduction: `ndir-asc__obj-unwt` -> `83.8251%`
- Best reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best final asynchronality: `ndir-asc__obj-unwt` -> `0.003733`
- Best asynchronality configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`
- Best asynchronality reduction: `ndir-asc__obj-unwt` -> `97.1631%`
- Best asynchronality-reduction configuration: `convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending`