# Sweep Comparison: by_group

Total completed runs: 4

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 0.000000 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |
| uplink | convergence | unweighted_sum_rate | descending | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 0.000000 | 34.315254 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_desc__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | ascending | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 0.000000 | 122.008450 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_asc__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | descending | 19.522777 | 5.600000 | 0.024733 | -3.380047 | 8.358654 | 0.000000 | 35.132495 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_desc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | convergence | inverse_cnr_weighted_sum_rate | 2 | 19.522777 | 5.600000 | 0.024733 | -3.344904 | 8.329684 | 78.570472 |
| uplink | convergence | unweighted_sum_rate | 2 | 19.739696 | 8.000000 | 0.024667 | -3.307363 | 8.352759 | 76.598654 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | convergence | ascending | 2 | 19.631236 | 6.800000 | 0.024700 | -3.311143 | 8.293306 | 120.445252 |
| uplink | convergence | descending | 2 | 19.631236 | 6.800000 | 0.024700 | -3.341125 | 8.389137 | 34.723875 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
