# Sweep Comparison: n_search_direction

Total completed runs: 8

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | descending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 95.508768 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_invcnr__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |
| uplink | convergence | unweighted_sum_rate | descending | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 34.315254 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_desc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | descending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 95.508768 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_invcnr__s3 |
| downlink | convergence | unweighted_sum_rate | descending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 148.755247 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_unwt__s3 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 0.000000 | 99.531435 | mc_invcnr_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_invcnr__s3 |
| downlink | monte_carlo | unweighted_sum_rate | descending | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 0.000000 | 119.482989 | mc_unwt_user__dl_payload_completion_sweep_20260727_215427_mc_ndir_desc_obj_unwt__s3 |
| uplink | convergence | unweighted_sum_rate | descending | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 0.000000 | 34.315254 | conv_unwt_net__ul_payload_completion_sweep_20260727_215427_conv_obj_unwt_ndir_desc__s3 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | descending | 19.522777 | 5.600000 | 0.024733 | -3.380047 | 8.358654 | 0.000000 | 35.132495 | conv_invcnr_net__ul_payload_completion_sweep_20260727_215427_conv_obj_invcnr_ndir_desc__s3 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 0.000000 | 183.415963 | mc_invcnr__ul_payload_completion_sweep_20260727_215427_mc_obj_invcnr_ndir_desc__s3 |
| uplink | monte_carlo | unweighted_sum_rate | descending | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 0.000000 | 189.580567 | mc_unwt__ul_payload_completion_sweep_20260727_215427_mc_obj_unwt_ndir_desc__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | 1 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 95.508768 |
| downlink | convergence | unweighted_sum_rate | 1 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 148.755247 |
| downlink | monte_carlo | inverse_cnr_weighted_sum_rate | 1 | 75.868852 | 83.485309 | 0.073600 | 13.037591 | 9.855616 | 99.531435 |
| downlink | monte_carlo | unweighted_sum_rate | 1 | 75.191257 | 63.323202 | 0.075667 | 11.974563 | 9.359354 | 119.482989 |
| uplink | convergence | inverse_cnr_weighted_sum_rate | 1 | 19.522777 | 5.600000 | 0.024733 | -3.380047 | 8.358654 | 35.132495 |
| uplink | convergence | unweighted_sum_rate | 1 | 19.739696 | 8.000000 | 0.024667 | -3.302202 | 8.419620 | 34.315254 |
| uplink | monte_carlo | inverse_cnr_weighted_sum_rate | 1 | 18.872017 | 8.000000 | 0.024933 | -3.325032 | 8.408865 | 183.415963 |
| uplink | monte_carlo | unweighted_sum_rate | 1 | 18.872017 | 8.000000 | 0.024933 | -3.326871 | 8.386546 | 189.580567 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | descending | 2 | 85.311475 | 93.465046 | 0.044800 | 13.436311 | 14.749723 | 122.132007 |
| downlink | monte_carlo | descending | 2 | 75.530055 | 73.404255 | 0.074633 | 12.506077 | 9.607485 | 109.507212 |
| uplink | convergence | descending | 2 | 19.631236 | 6.800000 | 0.024700 | -3.341125 | 8.389137 | 34.723875 |
| uplink | monte_carlo | descending | 2 | 18.872017 | 8.000000 | 0.024933 | -3.325951 | 8.397706 | 186.498265 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
