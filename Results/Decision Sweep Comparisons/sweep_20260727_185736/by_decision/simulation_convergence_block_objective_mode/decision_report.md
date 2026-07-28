# Decision Comparison: Downlink convergence objective

- Decision path: `simulation.convergence_block_objective_mode`
- Total runs touching this decision: `4`
- Completed runs: `4`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| unweighted_sum_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| unweighted_sum_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| inverse_cnr_weighted_sum_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |
| inverse_cnr_weighted_sum_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-invcnr | 0.040267 | 86.7978 | 0.013467 | 89.7670 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=ascending |

## Value folders

- `unweighted_sum_rate` -> `values/equal_priority_sum_rate`
- `inverse_cnr_weighted_sum_rate` -> `values/priority_weighted_sum_rate`