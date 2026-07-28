# Sweep Comparison: objective_mode

Total completed runs: 8

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | unweighted_sum_rate | ascending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 375.812445 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_unwt__s3 |
| downlink | monte_carlo | unweighted_sum_rate | descending | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 119.482989 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_unwt__s3 |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 189.580567 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_desc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | unweighted_sum_rate | ascending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 375.812445 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_unwt__s3 |
| downlink | convergence | unweighted_sum_rate | descending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 148.755247 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_unwt__s3 |
| downlink | monte_carlo | unweighted_sum_rate | descending | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 0.000000 | 119.482989 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_unwt__s3 |
| downlink | monte_carlo | unweighted_sum_rate | ascending | 40.830601 | 49.240122 | 0.180467 | 2.385907 | 7.504830 | 0.000000 | 1197.362474 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_unwt__s3 |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 0.000000 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |
| uplink | convergence | unweighted_sum_rate | descending | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 0.000000 | 34.315254 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_desc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 0.000000 | 189.580567 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_desc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433488 | 8.185217 | 0.000000 | 304.775712 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_asc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | unweighted_sum_rate | 2 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 262.283846 |
| downlink | monte_carlo | unweighted_sum_rate | 2 | 58.010929 | 56.281662 | 0.128067 | 7.180235 | 8.432092 | 658.422731 |
| uplink | convergence | unweighted_sum_rate | 2 | 19.739696 | 8.000000 | 0.024667 | -3.307363 | 8.352759 | 76.598654 |
| uplink | monte_carlo | unweighted_sum_rate | 2 | 14.750542 | 5.200000 | 0.026200 | -3.380179 | 8.285882 | 247.178139 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | ascending | 1 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 375.812445 |
| downlink | convergence | descending | 1 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 148.755247 |
| downlink | monte_carlo | ascending | 1 | 40.830601 | 49.240122 | 0.180467 | 2.385907 | 7.504830 | 1197.362474 |
| downlink | monte_carlo | descending | 1 | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 119.482989 |
| uplink | convergence | ascending | 1 | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 118.882054 |
| uplink | convergence | descending | 1 | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 34.315254 |
| uplink | monte_carlo | ascending | 1 | 10.629067 | 2.400000 | 0.027467 | -3.433488 | 8.185217 | 304.775712 |
| uplink | monte_carlo | descending | 1 | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 189.580567 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
