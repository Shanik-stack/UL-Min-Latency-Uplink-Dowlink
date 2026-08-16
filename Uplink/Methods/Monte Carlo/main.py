import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from UplinkSystem import UplinkSystem
from config_loader import _resolve_config_path, get_config, load_config, resolve_uplink_objective_mode
from experiment_cost import (
    build_uplink_monte_carlo_total_cost,
    build_uplink_monte_carlo_training_cost,
)
from experiment_determinism import configure_determinism
from experiment_scenarios import (
    build_experiment_scenario,
    build_experiment_scenario_summary,
    build_experiment_scenario_summary_lines,
    build_experiment_scenarios_for_seeds,
)
from experiment_utils import (
    compact_method_tag,
    compact_objective_tag,
    compact_training_style_tag,
    current_local_timestamp,
    join_compact_tag_parts,
    make_method_result_tag,
    resolve_monte_carlo_train_and_test_seeds,
    save_json,
    save_text,
)
from experiment_report import (
    build_dispersion_diagnostic_lines,
    build_post_training_summary_lines,
    build_precoder_net_result,
    build_summary_lines,
    build_training_dataset_summary_lines,
)
from policy_optimizer import (
    estimate_initial_random_precoder_schedule_for_scenario,
    evaluate_blocklength_precoder_net,
    train_blocklength_aware_precoder_net,
)
from terminal_logging import format_latency_log_line
from plotting import (
    initialize_plot_globals,
    plot_F_vs_n_for_all_subblocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_per_user_schedule_details,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_user_config,
)
from precoder_models import load_user_precoder_models
from project_paths import build_uplink_result_dirs, mirror_experiment_root_to_result_aliases
from utils import save_test_results_to_txt


def _build_test_search_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.test_n_search_strategy:
        overrides["monte_carlo_test_n_search_strategy"] = str(args.test_n_search_strategy)
    if args.test_n_search_direction:
        overrides["monte_carlo_test_n_search_direction"] = str(args.test_n_search_direction)
    if args.test_n_search_coarse_step is not None:
        overrides["monte_carlo_test_n_search_coarse_step"] = int(args.test_n_search_coarse_step)
    if args.test_n_search_exponential_factor is not None:
        overrides["monte_carlo_test_n_search_exponential_factor"] = int(
            args.test_n_search_exponential_factor
        )
    return overrides


def _build_test_search_tag(test_search_overrides: dict[str, object]) -> str | None:
    if not test_search_overrides:
        return None
    strategy = str(test_search_overrides.get("monte_carlo_test_n_search_strategy", "")).strip().lower()
    direction = str(test_search_overrides.get("monte_carlo_test_n_search_direction", "")).strip().lower()
    if strategy == "binary":
        return "ntest_bin"
    if strategy == "fixed_step":
        if direction.startswith("asc"):
            return "ntest_asc"
        if direction.startswith("desc"):
            return "ntest_desc"
    return join_compact_tag_parts("ntest", strategy or None, direction or None)


def _build_seeded_scenario_collection_lines(
    summaries: list[dict],
    *,
    title: str,
) -> list[str]:
    lines = [title]
    for idx, summary in enumerate(summaries):
        if idx > 0:
            lines.append("")
        lines.extend(build_experiment_scenario_summary_lines(summary))
    return lines


def _run_precoder_net_test(
    train_artifact: dict,
    cfg_name: str,
    test_seed: int,
    *,
    do_plots: bool,
    result_dirs: dict[str, str],
    train_seeds: list[int],
    test_search_overrides: dict[str, object] | None = None,
    reused_training_artifact: str | None = None,
):
    configure_determinism(int(test_seed))
    system_params, sim_cfg = get_config(cfg_name)
    sim_cfg = dict(sim_cfg)
    if test_search_overrides:
        sim_cfg.update(test_search_overrides)
    test_scenario = build_experiment_scenario(system_params, sim_cfg, seed=int(test_seed))
    test_scenario_summary = build_experiment_scenario_summary(test_scenario)
    initial_baseline = estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(test_seed),
    )
    naive_full_t_baseline = estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(test_seed),
        allow_n_reduction=False,
    )
    print(
        format_latency_log_line(
            "[UL Initial Baseline]",
            initial_baseline["initial_latency"],
            seed=int(test_seed),
            scenario=str(test_scenario.get("mode", sim_cfg.get("experiment_scenario_mode", "unknown"))),
            method="monte_carlo",
        )
    )
    test_uplinksystem = UplinkSystem(system_params, seed=int(test_seed))
    initial_snr_db = list(initial_baseline["initial_snr_db"])
    initial_sinr_db = list(initial_baseline["initial_sinr_db"])

    plot_params = dict(system_params)
    plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
    plot_user_config(
        plot_params,
        extra_params={
            "measured_snr_db_k": np.asarray(initial_snr_db),
            "measured_sinr_db_k": np.asarray(initial_sinr_db),
        },
    )

    initial_Rfbl = [np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]]
    initial_latency = list(initial_baseline["initial_latency"])
    initial_bits_per_symbol = list(initial_baseline["initial_bits_per_symbol"])
    initial_bits_per_symbol_by_block = [
        list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
    ]
    initial_n = list(initial_baseline["initial_n"])
    initial_n_kl = [list(values) for values in initial_baseline["initial_n_kl"]]
    initial_B_kl = [list(values) for values in initial_baseline["initial_B_kl"]]

    user_models = load_user_precoder_models(
        train_artifact["user_model_specs"],
        train_artifact["user_model_states"],
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    testing_started_at_local = current_local_timestamp()
    core_evaluation_start = perf_counter()
    post_test = evaluate_blocklength_precoder_net(
        uplinksystem=test_uplinksystem,
        user_models=user_models,
        sim_cfg=sim_cfg,
        method_name="monte_carlo_precoder_net_test",
    )
    core_evaluation_wall_time_seconds = perf_counter() - core_evaluation_start
    testing_completed_at_local = current_local_timestamp()

    test_data_dict = {
        "L_out_test": post_test["L_out"],
        "n_star_test": post_test["n_star"],
        "F_star_test": post_test["F_star"],
        "R_star_test": post_test["R_star"],
        "all_user_block_results_test": post_test["all_user_block_results_train"],
        "B_used_star_test": post_test["B_used_star"],
        "B_kl_star_test": post_test["B_kl_star"],
        "skipped_blocks_per_user": post_test.get("skipped_blocks_per_user", []),
        "scenario_mode": post_test.get("scenario_mode", ""),
        "scenario_block_targets": post_test.get("scenario_block_targets", []),
        "precoder_parameterization": train_artifact["precoder_parameterization"],
        "user_model_specs": train_artifact["user_model_specs"],
        "user_model_states": train_artifact["user_model_states"],
    }

    save_test_results_to_txt(
        test_uplinksystem=test_uplinksystem,
        test_data_dict=test_data_dict,
        initial_Rfbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        save_dir=result_dirs["test_data"],
        filename="test_results.txt",
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
    )

    if do_plots:
        plot_optimization_result(test_data_dict["all_user_block_results_test"], train=False)
        plot_optimization_result_summary_dict(
            {
                "n_star": test_data_dict["n_star_test"],
                "R_star": test_data_dict["R_star_test"],
                "all_user_block_results": test_data_dict["all_user_block_results_test"],
            },
            train=False,
        )
        test_plot_data = dict(test_data_dict)
        test_plot_data["precoder_net_training_history"] = train_artifact.get("precoder_net_training_history", {})
        plot_F_vs_n_for_all_subblocks(
            test_plot_data,
            save_dir="F_vs_n",
            base_dir=result_dirs["test_optimization_history"],
        )
        if int(test_uplinksystem.K) > 1:
            plot_latency_and_asynchronality_from_json(
                json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
                save_dir=result_dirs["latency_asynchronality"],
                prefix="test",
            )
        plot_link_quality_from_json(
            json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
            save_dir=result_dirs["link_quality"],
            prefix="test",
        )

    result = build_precoder_net_result(
        test_uplinksystem,
        test_data_dict,
        method_name="monte_carlo_precoder_net_train_test",
        cfg_path=_resolve_config_path(cfg_name),
        test_seed=int(test_seed),
        train_seeds=train_seeds,
        train_artifact=train_artifact,
        initial_R_fbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
        initial_interference_diag=initial_baseline.get("initial_interference_diag"),
        uplink_rate_model=sim_cfg.get("uplink_rate_model", "unknown"),
        naive_full_t_baseline=naive_full_t_baseline,
    )
    result["experiment_scenario_mode"] = sim_cfg.get("experiment_scenario_mode", "payload_completion")
    result["experiment_scenario"] = test_scenario_summary
    result["test_n_search_strategy"] = str(
        sim_cfg.get("monte_carlo_test_n_search_strategy", sim_cfg.get("n_search_strategy", "unknown"))
    )
    result["test_n_search_direction"] = str(
        sim_cfg.get("monte_carlo_test_n_search_direction", sim_cfg.get("n_search_direction", "unknown"))
    )
    if reused_training_artifact:
        result["reused_training_artifact"] = str(reused_training_artifact)
    test_candidate_n_states_per_user = [
        int(sum(len(block_states) for block_states in user_blocks))
        for user_blocks in test_data_dict["all_user_block_results_test"]
    ]
    raw_evaluation_counters = post_test.get("evaluation_cost_counters", {})
    if not isinstance(raw_evaluation_counters, dict):
        raw_evaluation_counters = {}
    result["evaluation_cost_counters"] = {
        "per_user_forward_calls": [
            int(v)
            for v in raw_evaluation_counters.get(
                "per_user_forward_calls",
                [0 for _ in range(int(test_uplinksystem.K))],
            )
        ],
        "total_forward_calls": int(raw_evaluation_counters.get("total_forward_calls", 0)),
        "per_user_candidate_n_states": test_candidate_n_states_per_user,
        "total_candidate_n_states": int(sum(test_candidate_n_states_per_user)),
    }
    result["core_evaluation_wall_time_seconds"] = float(core_evaluation_wall_time_seconds)
    result["testing_started_at_local"] = str(testing_started_at_local)
    result["testing_completed_at_local"] = str(testing_completed_at_local)
    save_json(result, os.path.join(result_dirs["test_data"], "result.json"))
    if do_plots:
        plot_per_user_schedule_details(result, result_dirs["schedule_details"])
        plot_interference_before_after_heatmaps(result, result_dirs["interference"])
        plot_per_user_interference_before_after(result, result_dirs["interference"])
        plot_interference_heatmaps(test_uplinksystem, result_dirs["interference"])
        plot_per_user_interference_profiles(test_uplinksystem, result_dirs["interference"])
    save_text(build_summary_lines(result), os.path.join(result_dirs["test_data"], "summary.txt"))
    dispersion_lines = build_dispersion_diagnostic_lines(result)
    if dispersion_lines:
        save_text(dispersion_lines, os.path.join(result_dirs["test_data"], "dispersion_diagnostics.txt"))
    save_json(test_scenario_summary, os.path.join(result_dirs["test_data"], "experiment_scenario.json"))
    save_text(
        build_experiment_scenario_summary_lines(test_scenario_summary),
        os.path.join(result_dirs["test_data"], "experiment_scenario.txt"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="config_raw_T_exp1.yaml", help="Configuration file name or path")
    parser.add_argument("--train_seeds", type=str, default=None, help="Explicit comma-separated training seeds")
    parser.add_argument(
        "--num_train_seeds",
        type=int,
        default=None,
        help="Build training seeds as 1..N excluding test_seed",
    )
    parser.add_argument("--test_seed", type=int, default=None, help="Monte Carlo test seed")
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", "--policy_epochs", dest="precoder_net_epochs", type=int, default=None)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", "--policy_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", "--policy_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--reuse_train_artifact", type=str, default=None, help="Reuse a saved train_artifact.pt and rerun only the test phase")
    parser.add_argument("--test_n_search_strategy", type=str, default=None, help="Override Monte Carlo test-only n-search strategy")
    parser.add_argument("--test_n_search_direction", type=str, default=None, help="Override Monte Carlo test-only n-search direction")
    parser.add_argument("--test_n_search_coarse_step", type=int, default=None, help="Override Monte Carlo test-only n-search coarse step")
    parser.add_argument("--test_n_search_exponential_factor", type=int, default=None, help="Override Monte Carlo test-only n-search exponential factor")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    system_params, sim_cfg, run_meta = load_config(args.cfg_name)
    run_started_at_local = current_local_timestamp()
    test_search_overrides = _build_test_search_overrides(args)
    train_epochs = int(
        args.precoder_net_epochs
        if args.precoder_net_epochs is not None
        else sim_cfg.get("monte_carlo_training_max_epochs", sim_cfg.get("max_epochs", 100))
    )
    train_seeds, test_seed = resolve_monte_carlo_train_and_test_seeds(
        cli_train_seeds=args.train_seeds,
        cli_num_train_seeds=args.num_train_seeds,
        cli_test_seed=args.test_seed,
        config_train_seeds=sim_cfg.get("monte_carlo_train_seeds"),
        config_num_train_seeds=sim_cfg.get("monte_carlo_num_train_seeds"),
        config_test_seed=sim_cfg.get("monte_carlo_test_seed"),
    )
    train_artifact: dict[str, object] | None = None
    reused_training_artifact: str | None = None
    if args.reuse_train_artifact:
        reused_training_artifact = os.path.abspath(args.reuse_train_artifact)
        train_artifact = torch.load(reused_training_artifact, map_location="cpu", weights_only=False)
        if not isinstance(train_artifact, dict):
            raise TypeError("Expected a dictionary train artifact when reusing Monte Carlo training.")
        artifact_train_seeds = [int(v) for v in train_artifact.get("train_seeds", [])]
        if artifact_train_seeds:
            train_seeds = artifact_train_seeds
    configure_determinism(train_seeds[0] if train_seeds else 0)
    print(f"Resolved Monte Carlo train seeds: {train_seeds}")
    print(f"Resolved Monte Carlo test seed: {int(test_seed)}")
    training_scenario_summaries = [
        build_experiment_scenario_summary(scenario)
        for scenario in build_experiment_scenarios_for_seeds(system_params, sim_cfg, train_seeds)
    ]
    if isinstance(train_artifact, dict) and train_artifact.get("training_experiment_scenarios"):
        training_scenario_summaries = list(train_artifact["training_experiment_scenarios"])
    objective_mode = resolve_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", "unweighted_sum_rate")
    )
    training_style_name = str(
        train_artifact.get("monte_carlo_training_style", sim_cfg.get("monte_carlo_training_style", "rollout_query_lagrangian"))
        if isinstance(train_artifact, dict)
        else sim_cfg.get("monte_carlo_training_style", "rollout_query_lagrangian")
    )
    result_tag = make_method_result_tag(
        join_compact_tag_parts(
            compact_method_tag("monte_carlo_precoder_net_train_test"),
            compact_objective_tag(objective_mode),
            compact_training_style_tag(training_style_name),
            _build_test_search_tag(test_search_overrides),
        ),
        run_meta["cfg_stem"],
        seed=int(test_seed),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_dirs = build_uplink_result_dirs("Monte Carlo", result_tag)
    initialize_plot_globals(result_tag, result_dirs)

    if train_artifact is None:
        training_started_at_local = current_local_timestamp()
        training_start = perf_counter()
        train_artifact = train_blocklength_aware_precoder_net(
            cfg_name=args.cfg_name,
            train_seeds=train_seeds,
            epochs=train_epochs,
            batch_size=args.precoder_net_batch_size,
            lr=args.precoder_net_lr,
        )
        training_wall_time_seconds = perf_counter() - training_start
        training_completed_at_local = current_local_timestamp()
        train_artifact["cfg_path"] = _resolve_config_path(args.cfg_name)
        train_artifact["cfg_hash"] = run_meta.get("cfg_hash")
        train_artifact["method_name"] = "monte_carlo_precoder_net_train_test"
        train_artifact["test_seed"] = int(test_seed)
        train_artifact["experiment_scenario_mode"] = sim_cfg.get("experiment_scenario_mode", "payload_completion")
        train_artifact["training_experiment_scenarios"] = training_scenario_summaries
        training_cost = build_uplink_monte_carlo_training_cost(
            train_artifact,
            batch_size=args.precoder_net_batch_size,
            core_wall_time_seconds_training=training_wall_time_seconds,
        )
        train_artifact["experiment_cost"] = training_cost
        if isinstance(train_artifact.get("post_training_summary"), dict):
            train_artifact["post_training_summary"]["run_started_at_local"] = str(run_started_at_local)
            train_artifact["post_training_summary"]["training_started_at_local"] = str(training_started_at_local)
            train_artifact["post_training_summary"]["training_completed_at_local"] = str(training_completed_at_local)
            train_artifact["post_training_summary"]["experiment_cost"] = training_cost

        plot_optimization_result(train_artifact["all_user_block_results_train"], train=True)
        plot_optimization_result_summary_dict(train_artifact, train=True)
        plot_F_vs_n_for_all_subblocks(train_artifact)

        torch.save(train_artifact, os.path.join(result_dirs["train_data"], "train_artifact.pt"))
        save_json(
            train_artifact.get("training_dataset_summary", {}),
            os.path.join(result_dirs["train_data"], "training_dataset_summary.json"),
        )
        save_text(
            build_training_dataset_summary_lines(train_artifact.get("training_dataset_summary", {})),
            os.path.join(result_dirs["train_data"], "training_dataset_summary.txt"),
        )
        save_json(
            train_artifact.get("post_training_summary", {}),
            os.path.join(result_dirs["train_data"], "post_training_summary.json"),
        )
        save_text(
            build_post_training_summary_lines(train_artifact.get("post_training_summary", {})),
            os.path.join(result_dirs["train_data"], "post_training_summary.txt"),
        )
        save_json(
            {"seed_scenarios": training_scenario_summaries},
            os.path.join(result_dirs["train_data"], "experiment_scenarios.json"),
        )
        save_text(
            _build_seeded_scenario_collection_lines(
                training_scenario_summaries,
                title="Training experiment scenarios by seed",
            ),
            os.path.join(result_dirs["train_data"], "experiment_scenarios.txt"),
        )
    else:
        print(f"Reusing Monte Carlo training artifact: {reused_training_artifact}")
        prior_cost = train_artifact.get("experiment_cost", {})
        if not isinstance(prior_cost, dict):
            prior_cost = {}
        training_wall_time_seconds = float(prior_cost.get("core_wall_time_seconds_training", 0.0))
        prior_training_summary = train_artifact.get("post_training_summary", {})
        if not isinstance(prior_training_summary, dict):
            prior_training_summary = {}
        training_started_at_local = str(
            prior_training_summary.get("training_started_at_local", "reused_training_artifact")
        )
        training_completed_at_local = str(
            prior_training_summary.get("training_completed_at_local", "reused_training_artifact")
        )
        save_json(
            {
                "reused_training_artifact": reused_training_artifact,
                "train_seeds": train_seeds,
                "test_seed": int(test_seed),
                "test_search_overrides": test_search_overrides,
            },
            os.path.join(result_dirs["train_data"], "reused_training_artifact.json"),
        )
        save_text(
            [
                "This Monte Carlo run reused an existing training artifact and reran only the test phase.",
                f"Source artifact: {reused_training_artifact}",
                f"Train seeds: {train_seeds}",
                f"Test seed: {int(test_seed)}",
                f"Test n-search overrides: {test_search_overrides}",
            ],
            os.path.join(result_dirs["train_data"], "reused_training_artifact.txt"),
        )

    if not args.skip_test:
        test_result = _run_precoder_net_test(
            train_artifact=train_artifact,
            cfg_name=args.cfg_name,
            test_seed=int(test_seed),
            do_plots=True,
            result_dirs=result_dirs,
            train_seeds=train_seeds,
            test_search_overrides=test_search_overrides,
            reused_training_artifact=reused_training_artifact,
        )
        testing_wall_time_seconds = float(test_result.get("core_evaluation_wall_time_seconds", 0.0))
        testing_completed_at_local = current_local_timestamp()
        total_cost = build_uplink_monte_carlo_total_cost(
            train_artifact,
            test_result.get("evaluation_cost_counters", {}),
            batch_size=args.precoder_net_batch_size,
            core_wall_time_seconds_training=training_wall_time_seconds,
            core_wall_time_seconds_testing=testing_wall_time_seconds,
        )
        test_result["experiment_cost"] = total_cost
        test_result["cfg_hash"] = run_meta.get("cfg_hash")
        test_result["run_started_at_local"] = str(run_started_at_local)
        test_result["run_completed_at_local"] = str(testing_completed_at_local)
        test_result["training_started_at_local"] = str(training_started_at_local)
        test_result["training_completed_at_local"] = str(training_completed_at_local)
        save_json(test_result, os.path.join(result_dirs["test_data"], "result.json"))
        save_text(build_summary_lines(test_result), os.path.join(result_dirs["test_data"], "summary.txt"))
        dispersion_lines = build_dispersion_diagnostic_lines(test_result)
        if dispersion_lines:
            save_text(dispersion_lines, os.path.join(result_dirs["test_data"], "dispersion_diagnostics.txt"))
    mirror_paths = mirror_experiment_root_to_result_aliases(
        link_name="Uplink",
        scenario_mode=str(sim_cfg.get("experiment_scenario_mode", "payload_completion")),
        method_name="Monte Carlo",
        source_experiment_root=result_dirs["experiment_root"],
    )
    print(f"Mirrored uplink precoder-net scenario results to: {mirror_paths['scenario_root']}")
    print(f"Saved uplink precoder-net method results to: {mirror_paths['method_root']}")


if __name__ == "__main__":
    main()
