# Sweep Comparison: by_group

Total completed runs: 4

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 0.000000 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |
| downlink | monte_carlo | unweighted_sum_rate | descending | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 0.000000 | 119.482989 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_unwt__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 0.000000 | 1223.862897 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_invcnr__s3 |
| downlink | monte_carlo | unweighted_sum_rate | ascending | 40.830601 | 49.240122 | 0.180467 | 2.385907 | 7.504830 | 0.000000 | 1197.362474 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_unwt__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | 2 | 73.060109 | 77.355623 | 0.082167 | 9.950324 | 9.900655 | 661.697166 |
| downlink | monte_carlo | unweighted_sum_rate | 2 | 58.010929 | 56.281662 | 0.128067 | 7.180235 | 8.432092 | 658.422731 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | monte_carlo | ascending | 2 | 55.540984 | 60.233029 | 0.135600 | 4.624482 | 8.725263 | 1210.612686 |
| downlink | monte_carlo | descending | 2 | 75.530055 | 73.404255 | 0.074633 | 12.506077 | 9.607485 | 109.507212 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
