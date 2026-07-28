# Sweep Comparison: objective_mode

Total completed runs: 8

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | ascending | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 122.008450 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_asc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |
| downlink | convergence | inverse_cnr_weighted_sum_rate | descending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 95.508768 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_invcnr__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 0.000000 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 0.000000 | 1223.862897 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_asc_obj_invcnr__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | ascending | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 0.000000 | 122.008450 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_asc__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | descending | 19.522777 | 5.600000 | 0.024733 | -3.380047 | 8.358654 | 0.000000 | 35.132495 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_desc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 0.000000 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | ascending | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 0.000000 | 363.054767 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_asc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | 2 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 238.816757 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | 2 | 73.060109 | 77.355623 | 0.082167 | 9.950324 | 9.900655 | 661.697166 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | 2 | 19.522777 | 5.600000 | 0.024733 | -3.344904 | 8.329684 | 78.570472 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | 2 | 14.750542 | 5.200000 | 0.026200 | -3.379126 | 8.301870 | 273.235365 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | ascending | 1 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 382.124747 |
| downlink | convergence | descending | 1 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 95.508768 |
| downlink | monte_carlo | ascending | 1 | 70.251366 | 71.225937 | 0.090733 | 6.863058 | 9.945695 | 1223.862897 |
| downlink | monte_carlo | descending | 1 | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 99.531435 |
| uplink | convergence | ascending | 1 | 19.522777 | 5.600000 | 0.024733 | -3.309762 | 8.300714 | 122.008450 |
| uplink | convergence | descending | 1 | 19.522777 | 5.600000 | 0.024733 | -3.380047 | 8.358654 | 35.132495 |
| uplink | monte_carlo | ascending | 1 | 10.629067 | 2.400000 | 0.027467 | -3.433220 | 8.194875 | 363.054767 |
| uplink | monte_carlo | descending | 1 | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 183.415963 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
