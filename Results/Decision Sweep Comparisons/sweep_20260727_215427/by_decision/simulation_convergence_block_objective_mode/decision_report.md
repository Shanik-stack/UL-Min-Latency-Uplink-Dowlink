# Decision Comparison: Downlink convergence objective

- Decision path: `simulation.convergence_block_objective_mode`
- Total runs touching this decision: `8`
- Completed runs: `6`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| inverse_cnr_weighted_sum_rate | best_final_total_latency | downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.073600 | 75.8689 | 0.021733 | 83.4853 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |
| inverse_cnr_weighted_sum_rate | best_latency_reduction_percent | downlink | monte_carlo | downlink_payload_completion.yaml | ndir-desc__obj-invcnr | 0.073600 | 75.8689 | 0.021733 | 83.4853 | convergence_block_objective_mode=inverse_cnr_weighted_sum_rate; n_search_direction=descending |
| unweighted_sum_rate | best_final_total_latency | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |
| unweighted_sum_rate | best_latency_reduction_percent | downlink | convergence | downlink_payload_completion.yaml | ndir-asc__obj-unwt | 0.049333 | 83.8251 | 0.003733 | 97.1631 | convergence_block_objective_mode=unweighted_sum_rate; n_search_direction=ascending |

## Value folders

- `inverse_cnr_weighted_sum_rate` -> `values/inverse_cnr_weighted_sum_rate`
- `unweighted_sum_rate` -> `values/unweighted_sum_rate`