import argparse
import os
import sys
from pathlib import Path

import numpy as np


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from advanced_methods_common import (
    apply_training_solution,
    estimate_initial_random_precoder_schedule_for_scenario,
)
from config_loader import _resolve_config_path, get_config
from experiment_report import build_convergence_result, build_convergence_summary_lines
from experiment_utils import make_method_result_tag, save_json, save_text
from optimizer import dynamic_subblocklength_precoder_training_baseline
from plotting import (
    initialize_plot_globals,
    plot_F_vs_n_for_all_subblocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_user_config,
)
from project_paths import build_uplink_convergence_result_dirs
from UplinkSystem import UplinkSystem
from utils import save_test_results_to_txt


def _resolve_run_seed(args: argparse.Namespace) -> int:
    legacy_seeds = [value for value in (args.train_seed, args.test_seed) if value is not None]
    run_seed = int(args.seed) if args.seed is not None else int(legacy_seeds[0]) if legacy_seeds else 0

    for value in legacy_seeds:
        if int(value) != run_seed:
            raise ValueError(
                "Uplink convergence per sweep now uses one shared seed only. "
                "Provide the same value for --seed, --train_seed, and --test_seed, or use only --seed."
            )
    return run_seed


def run_convergence_experiment(
    cfg_name: str,
    seed: int,
    *,
    do_plots: bool = True,
) -> dict:
    system_params, sim_cfg = get_config(cfg_name)
    sim_cfg = dict(sim_cfg)

    initial_baseline = estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
    )

    report_system = UplinkSystem(system_params, seed=int(seed))
    initial_snr_db = list(initial_baseline["initial_snr_db"])
    initial_sinr_db = list(initial_baseline["initial_sinr_db"])

    if do_plots:
        plot_params = dict(system_params)
        plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
        plot_user_config(
            plot_params,
            extra_params={
                "measured_snr_db_k": np.asarray(initial_snr_db),
                "measured_sinr_db_k": np.asarray(initial_sinr_db),
            },
        )

    convergence_system = UplinkSystem(system_params, seed=int(seed))
    convergence_data = dynamic_subblocklength_precoder_training_baseline(
        uplinksystem=convergence_system,
        sim_cfg=sim_cfg,
        channel_norm=True,
    )

    apply_training_solution(report_system, convergence_data["n_star"], convergence_data["F_star"])

    result = build_convergence_result(
        report_system,
        convergence_data,
        method_name="converge_in_each_sweep_baseline",
        cfg_path=_resolve_config_path(cfg_name),
        seed=int(seed),
        initial_R_fbl=[np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]],
        initial_n_kl=[list(values) for values in initial_baseline["initial_n_kl"]],
        initial_n=list(initial_baseline["initial_n"]),
        initial_latency=list(initial_baseline["initial_latency"]),
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        initial_bits_per_symbol=list(initial_baseline["initial_bits_per_symbol"]),
        initial_B_kl=[list(values) for values in initial_baseline["initial_B_kl"]],
        initial_bits_per_symbol_by_block=[
            list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
        ],
        initial_interference_diag=initial_baseline.get("initial_interference_diag"),
    )
    result["scenario_mode"] = convergence_data.get(
        "scenario_mode",
        sim_cfg.get("experiment_scenario_mode", "payload_completion"),
    )
    return {
        "result": result,
        "report_system": report_system,
        "convergence_data": convergence_data,
        "initial_baseline": initial_baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Uplink online convergence baseline")
    parser.add_argument("--cfg_name", type=str, default="config_raw_T_exp1.yaml", help="Configuration file name or path")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed")
    parser.add_argument("--train_seed", "--train-seed", dest="train_seed", type=int, default=None)
    parser.add_argument("--test_seed", "--test-seed", dest="test_seed", type=int, default=None)
    args = parser.parse_args()

    run_seed = _resolve_run_seed(args)
    result_tag = make_method_result_tag("converge_in_each_sweep_baseline", args.cfg_name, seed=run_seed)
    result_dirs = build_uplink_convergence_result_dirs("Convergence per sweep", result_tag)
    initialize_plot_globals(result_tag, result_dirs)

    experiment = run_convergence_experiment(
        cfg_name=args.cfg_name,
        seed=run_seed,
        do_plots=True,
    )
    result = experiment["result"]
    report_system = experiment["report_system"]
    convergence_data = experiment["convergence_data"]
    initial_baseline = experiment["initial_baseline"]

    convergence_plot_dict = {
        "n_star": convergence_data["n_star"],
        "R_star": convergence_data["R_star"],
    }
    convergence_raw_dict = {
        "B_kl_star_test": convergence_data["B_kl_star"],
        "n_star_test": convergence_data["n_star"],
        "R_star_test": convergence_data["R_star"],
        "all_user_block_results_test": convergence_data["all_user_block_results_train"],
        "scenario_mode": convergence_data.get("scenario_mode", ""),
        "scenario_block_targets": convergence_data.get("scenario_block_targets", []),
    }

    save_test_results_to_txt(
        test_uplinksystem=report_system,
        test_data_dict=convergence_raw_dict,
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

    plot_optimization_result(
        convergence_data["all_user_block_results_train"],
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Convergence",
        filename_prefix="convergence",
    )
    plot_optimization_result_summary_dict(
        convergence_plot_dict,
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Convergence",
        filename_prefix="convergence",
    )
    plot_F_vs_n_for_all_subblocks(
        convergence_data,
        save_dir="F_vs_n",
        base_dir=result_dirs["optimization_history"],
    )
    plot_latency_and_asynchronality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["latency_asynchronality"],
        prefix="convergence",
    )
    plot_link_quality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["link_quality"],
        prefix="convergence",
    )

    save_json(result, os.path.join(result_dirs["data"], "result.json"))
    plot_interference_before_after_heatmaps(result, result_dirs["interference"])
    plot_per_user_interference_before_after(result, result_dirs["interference"])
    plot_interference_heatmaps(report_system, result_dirs["interference"])
    plot_per_user_interference_profiles(report_system, result_dirs["interference"])
    save_text(build_convergence_summary_lines(result), os.path.join(result_dirs["data"], "summary.txt"))
    print(f"Saved uplink convergence results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
