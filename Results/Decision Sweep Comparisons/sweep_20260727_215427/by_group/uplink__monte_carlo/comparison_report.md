# Sweep Comparison: by_group

Total completed runs: 4

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 0.000000 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 0.000000 | 189.580567 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_desc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 0.000000 | 363.054767 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_asc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433488 | 8.185217 | 0.000000 | 304.775712 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_asc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | 2 | 14.750542 | 5.200000 | 0.026200 | -3.379126 | 8.301870 | 273.235365 |
| uplink | monte_carlo | unweighted_sum_rate | 2 | 14.750542 | 5.200000 | 0.026200 | -3.380179 | 8.285882 | 247.178139 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uplink | monte_carlo | ascending | 2 | 10.629067 | 2.400000 | 0.027467 | -3.433354 | 8.190046 | 333.915239 |
| uplink | monte_carlo | descending | 2 | 18.872017 | 8.000000 | 0.024933 | -3.325951 | 8.397706 | 186.498265 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
