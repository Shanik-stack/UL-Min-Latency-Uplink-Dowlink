# Sweep Comparison: n_search_direction

Total completed runs: 8

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 1223.862897 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_invcnr__s3 |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 363.054767 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_asc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |
| downlink | convergence | unweighted_sum_rate | ascending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 375.812445 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_unwt__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 0.000000 | 1223.862897 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_invcnr__s3 |
| downlink | monte_carlo | unweighted_sum_rate | ascending | 40.830601 | 49.240122 | 0.180467 | 2.385907 | 7.504830 | 0.000000 | 1197.362474 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_unwt__s3 |
| uplink | convergence | unweighted_sum_rate | ascending | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 0.000000 | 118.882054 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_asc__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | ascending | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 0.000000 | 122.008450 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_asc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 0.000000 | 363.054767 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_asc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433488 | 8.185217 | 0.000000 | 304.775712 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_asc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | 1 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 382.124747 |
| downlink | convergence | unweighted_sum_rate | 1 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 375.812445 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | 1 | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 1223.862897 |
| downlink | monte_carlo | unweighted_sum_rate | 1 | 40.830601 | 49.240122 | 0.180467 | 2.385907 | 7.504830 | 1197.362474 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | 1 | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 122.008450 |
| uplink | convergence | unweighted_sum_rate | 1 | 19.739696 | 8.000000 | 0.024667 | -3.312524 | 8.285898 | 118.882054 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | 1 | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 363.054767 |
| uplink | monte_carlo | unweighted_sum_rate | 1 | 10.629067 | 2.400000 | 0.027467 | -3.433488 | 8.185217 | 304.775712 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | ascending | 2 | 85.311475 | 93.465046 | 0.044800 | 13.436311 | 14.749723 | 378.968596 |
| downlink | monte_carlo | ascending | 2 | 55.540984 | 60.233029 | 0.135600 | 4.624482 | 8.725263 | 1210.612686 |
| uplink | convergence | ascending | 2 | 19.631236 | 6.800000 | 0.024700 | -3.311143 | 8.293306 | 120.445252 |
| uplink | monte_carlo | ascending | 2 | 10.629067 | 2.400000 | 0.027467 | -3.433354 | 8.190046 | 333.915239 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
