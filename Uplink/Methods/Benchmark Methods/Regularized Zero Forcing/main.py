from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


METHOD_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = METHOD_DIR.parent
LINK_ROOT = BENCHMARK_ROOT.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (BENCHMARK_ROOT, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from common import run_uplink_closed_form_benchmark
from plotting import (
    initialize_plot_globals,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_kkt_residual_history,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_per_user_schedule_details,
    plot_user_config,
)
from project_paths import mirror_experiment_root_to_result_aliases
from utils import save_test_results_to_txt
from experiment_utils import save_json, save_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Uplink Regularized Zero Forcing benchmark")
    parser.add_argument("--cfg_name", type=str, default="uplink_payload_completion.yaml")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    experiment = run_uplink_closed_form_benchmark(
        method_key="rzf",
        cfg_name=args.cfg_name,
        seed=int(args.seed),
        verbose=not bool(args.quiet),
    )
    result = experiment["result"]
    report_system = experiment["report_system"]
    benchmark_data = experiment["benchmark_data"]
    initial_baseline = experiment["initial_baseline"]
    result_dirs = experiment["result_dirs"]
    initialize_plot_globals(os.path.basename(result_dirs["experiment_root"]), result_dirs)

    plot_params = dict(report_system.sc)
    plot_params["initial_bits_per_symbol"] = initial_baseline["initial_bits_per_symbol"]
    plot_user_config(
        plot_params,
        extra_params={
            "measured_snr_db_k": initial_baseline["initial_snr_db"],
            "measured_sinr_db_k": initial_baseline["initial_sinr_db"],
        },
    )

    benchmark_raw_dict = {
        "B_kl_star_test": benchmark_data["B_kl_star"],
        "n_star_test": benchmark_data["n_star"],
        "R_star_test": benchmark_data["R_star"],
        "all_user_block_results_test": benchmark_data["all_user_block_results_train"],
        "scenario_mode": benchmark_data.get("scenario_mode", ""),
        "scenario_block_targets": benchmark_data.get("scenario_block_targets", []),
    }
    save_test_results_to_txt(
        test_uplinksystem=report_system,
        test_data_dict=benchmark_raw_dict,
        initial_Rfbl=[np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]],
        initial_n_kl=[list(values) for values in initial_baseline["initial_n_kl"]],
        initial_n=list(initial_baseline["initial_n"]),
        initial_latency=list(initial_baseline["initial_latency"]),
        initial_snr_db=list(result["initial_snr_db"]),
        initial_sinr_db=list(result["initial_sinr_db"]),
        initial_bits_per_symbol=list(initial_baseline["initial_bits_per_symbol"]),
        save_dir=result_dirs["data"],
        filename="convergence_results.txt",
        initial_B_kl=[list(values) for values in initial_baseline["initial_B_kl"]],
        initial_bits_per_symbol_by_block=[
            list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
        ],
    )

    save_json(result, os.path.join(result_dirs["data"], "result.json"))
    plot_kkt_residual_history(
        result,
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Benchmark",
        filename_prefix="benchmark",
    )
    plot_latency_and_asynchronality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["latency_asynchronality"],
        prefix="benchmark",
    )
    plot_link_quality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["link_quality"],
        prefix="benchmark",
    )
    plot_per_user_schedule_details(result, result_dirs["schedule_details"])
    plot_interference_before_after_heatmaps(result, result_dirs["interference"])
    plot_per_user_interference_before_after(result, result_dirs["interference"])
    plot_interference_heatmaps(report_system, result_dirs["interference"])
    plot_per_user_interference_profiles(report_system, result_dirs["interference"])
    save_text(experiment["summary_lines"], os.path.join(result_dirs["data"], "summary.txt"))
    mirror_paths = mirror_experiment_root_to_result_aliases(
        link_name="Uplink",
        scenario_mode="payload_completion",
        method_name="Benchmark RZF",
        source_experiment_root=result_dirs["experiment_root"],
    )
    print(f"Mirrored uplink benchmark scenario results to: {mirror_paths['scenario_root']}")
    print(f"Saved uplink benchmark method results to: {mirror_paths['method_root']}")


if __name__ == "__main__":
    import numpy as np

    main()
