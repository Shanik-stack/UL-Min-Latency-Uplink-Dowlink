# Decision Value Comparison: Uplink objective = unweighted_sum_rate

- Decision path: `simulation.uplink_objective_mode`
- Completed runs: `4` / `4`

| Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SINR (dB) | Served bits | Highlighted configuration |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | -3.3125 | 6000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |
| uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-desc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | -3.3022 | 6000 | n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-unwt__ndir-desc | 0.024933 | 18.8720 | 0.015333 | 8.0000 | -3.3269 | 6000 | n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate |
| uplink | monte_carlo | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.027467 | 10.6291 | 0.016267 | 2.4000 | -3.4335 | 6000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |

- Best final total latency: `obj-unwt__ndir-asc` -> `0.024667`
- Best latency configuration: `n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate`
- Best latency reduction: `obj-unwt__ndir-asc` -> `19.7397%`
- Best reduction configuration: `n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate`
- Best final asynchronality: `obj-unwt__ndir-desc` -> `0.015333`
- Best asynchronality configuration: `n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate`
- Best asynchronality reduction: `obj-unwt__ndir-desc` -> `8.0000%`
- Best asynchronality-reduction configuration: `n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate`