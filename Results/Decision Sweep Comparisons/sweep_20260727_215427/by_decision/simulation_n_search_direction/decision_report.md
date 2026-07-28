# Decision Comparison: n_kl search direction

- Decision path: `simulation.n_search_direction`
- Total runs touching this decision: `16`
- Completed runs: `14`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ascending | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |
| ascending | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| descending | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-desc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=descending; uplink_objective_mode=unweighted_sum_rate |
| descending | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=descending |

## Value folders

- `ascending` -> `values/ascending`
- `descending` -> `values/descending`