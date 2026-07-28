# Decision Comparison: Uplink objective

- Decision path: `simulation.uplink_objective_mode`
- Total runs touching this decision: `8`
- Completed runs: `8`

| Decision value | Best criterion | Link | Method | Base config | Variant | Final total latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Highlighted configuration |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| inverse_cnr_weighted_sum_rate | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | obj-invcnr__ndir-asc | 0.024733 | 19.5228 | 0.015733 | 5.6000 | n_search_direction=ascending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| inverse_cnr_weighted_sum_rate | best_latency_reduction_percent | uplink | convergence | uplink_payload_completion.yaml | obj-invcnr__ndir-asc | 0.024733 | 19.5228 | 0.015733 | 5.6000 | n_search_direction=ascending; uplink_objective_mode=inverse_cnr_weighted_sum_rate |
| unweighted_sum_rate | best_final_total_latency | uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |
| unweighted_sum_rate | best_latency_reduction_percent | uplink | convergence | uplink_payload_completion.yaml | obj-unwt__ndir-asc | 0.024667 | 19.7397 | 0.015333 | 8.0000 | n_search_direction=ascending; uplink_objective_mode=unweighted_sum_rate |

## Value folders

- `inverse_cnr_weighted_sum_rate` -> `values/inverse_cnr_weighted_sum_rate`
- `unweighted_sum_rate` -> `values/unweighted_sum_rate`