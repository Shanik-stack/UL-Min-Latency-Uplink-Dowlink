# Decision Comparison: n_kl search direction

- Decision path: `simulation.n_search_direction`
- Total runs touching this decision: `9`
- Completed runs: `9`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ascending | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=ascending |
| ascending | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| descending | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | ndir-desc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=descending |
| descending | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |

## Value folders

- `ascending` -> `values/ascending`
- `descending` -> `values/descending`