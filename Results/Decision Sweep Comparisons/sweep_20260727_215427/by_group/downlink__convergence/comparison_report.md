# Sweep Comparison: by_group

Total completed runs: 4

## Best by link and method
| group_link | group_method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |

## All runs sorted by latency reduction
| link | method | objective_mode | n_search_direction | total_latency_reduction_percent | asynchronality_reduction_percent | final_total_latency | final_avg_sinr_db | final_avg_snr_db | unserved_bits_total | core_wall_time_seconds_total | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | ascending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 382.124747 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_invcnr__s3 |
| downlink | convergence | inverse_cnr_weighted_sum_rate | descending | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 0.000000 | 95.508768 | conv_weighted_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_invcnr__s3 |
| downlink | convergence | unweighted_sum_rate | ascending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 375.812445 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_asc_obj_unwt__s3 |
| downlink | convergence | unweighted_sum_rate | descending | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 0.000000 | 148.755247 | conv_unwt_user_net__dl_payload_completion_sweep_20260727_215427_conv_ndir_desc_obj_unwt__s3 |

## Objective averages
| link | method | objective_mode | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | inverse_cnr_weighted_sum_rate | 2 | 86.797814 | 89.766971 | 0.040267 | 18.623403 | 15.320260 | 238.816757 |
| downlink | convergence | unweighted_sum_rate | 2 | 83.825137 | 97.163121 | 0.049333 | 8.249220 | 14.179186 | 262.283846 |

## Direction averages
| link | method | n_search_direction | run_count | avg_total_latency_reduction_percent | avg_asynchronality_reduction_percent | avg_final_total_latency | avg_final_avg_sinr_db | avg_final_avg_snr_db | avg_core_wall_time_seconds_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downlink | convergence | ascending | 2 | 85.311475 | 93.465046 | 0.044800 | 13.436311 | 14.749723 | 378.968596 |
| downlink | convergence | descending | 2 | 85.311475 | 93.465046 | 0.044800 | 13.436311 | 14.749723 | 122.132007 |

## Notes
- `objective_mode` is taken from the saved result JSON, not only the folder name.
- `n_search_direction` is inferred from the run folder tag.
- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.
