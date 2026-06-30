from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import torch


METHOD_DIR = Path(__file__).resolve().parent
LINK_ROOT = METHOD_DIR.parents[1]
PROJECT_ROOT = LINK_ROOT.parent
for path in (METHOD_DIR, LINK_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config_loader import load_config
from determinism import configure_determinism
from downlink_system import DownlinkSystem
from experiment_cost import (
    build_downlink_monte_carlo_total_cost,
    build_downlink_monte_carlo_training_cost,
    format_experiment_cost_lines,
)
from experiment_scenarios import (
    FIXED_BLOCK_TARGETS_MODE,
    build_experiment_scenario,
    build_experiment_scenario_summary,
    build_experiment_scenario_summary_lines,
    build_experiment_scenarios_for_seeds,
)
from experiment_utils import make_method_result_tag, parse_seed_list
from experiment_runner import _compute_summary_metrics
from policy_optimizer import (
    build_precoder_net_artifact,
    build_training_dataset,
    evaluate_downlink_precoder_net,
    train_blocklength_aware_precoder_net,
)
from plotting import (
    plot_asynchronality_comparison,
    plot_blocklength_feasibility_curves,
    plot_blocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_latency,
    plot_link_quality,
    plot_optimization_history,
    plot_per_user_convergence,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_per_user_schedule_details,
    plot_rate_violation_heatmap,
    plot_user_config,
)
from project_paths import build_downlink_result_dirs
from utils import save_json, save_text


def _build_seeded_scenario_collection_lines(
    summaries: list[dict[str, object]],
    *,
    title: str,
) -> list[str]:
    lines = [title]
    for idx, summary in enumerate(summaries):
        if idx > 0:
            lines.append("")
        lines.extend(build_experiment_scenario_summary_lines(summary))
    return lines


def _build_dataset_summary_lines(dataset_summary: dict[str, object]) -> list[str]:
    return [
        "Downlink training dataset summary",
        f"Total training channel episodes: {int(dataset_summary.get('total_channel_episodes', 0))}",
        f"Training scenario modes: {dataset_summary.get('scenario_modes', [])}",
        f"Training channel episodes by seed: {dataset_summary.get('channel_episodes_by_seed', {})}",
        f"Training channel episodes by block: {dataset_summary.get('channel_episodes_by_block', {})}",
        f"Training channel episodes by active user count: {dataset_summary.get('channel_episodes_by_active_user_count', {})}",
        f"Training channel episodes by active mask: {dataset_summary.get('channel_episodes_by_active_mask', {})}",
        f"Active-user channel episodes per user: {dataset_summary.get('channel_episodes_per_user', [])}",
        f"Global active-user channel episodes by target bits: {dataset_summary.get('global_active_user_cases_by_target_bits', {})}",
        "Per-user active-user channel episodes by target bits:",
        f"{dataset_summary.get('per_user_active_user_cases_by_target_bits', [])}",
        "",
        "Terminology",
        "- channel episode: one (seed, block) block realization stored in the base dataset",
        "- active-user channel episode: one active user inside one stored channel episode",
    ]


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    lines = [
        "Downlink post-training summary",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Downlink precoder-net scope: {post_training_summary.get('downlink_precoder_net_scope', 'unknown')}",
        f"Train target-bits mode: {post_training_summary.get('train_target_bits_mode', 'unknown')}",
        f"Train target bits summary: {post_training_summary.get('train_target_bits_summary', {})}",
        f"Final avg sum rate: {float(post_training_summary.get('final_avg_sum_rate', 0.0)):.6f}",
        f"Best avg sum rate: {float(post_training_summary.get('best_avg_sum_rate', 0.0)):.6f}",
        f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
        f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
        f"Final avg lagrangian: {float(post_training_summary.get('final_avg_lagrangian', 0.0)):.6f}",
        f"Best avg lagrangian: {float(post_training_summary.get('best_avg_lagrangian', 0.0)):.6f}",
        f"Final avg rate violation: {float(post_training_summary.get('final_avg_rate_violation', 0.0)):.6f}",
        f"Best avg rate violation: {float(post_training_summary.get('best_avg_rate_violation', 0.0)):.6f}",
        f"Final avg block-power violation: {float(post_training_summary.get('final_avg_block_power_violation', 0.0)):.6f}",
        f"Best avg block-power violation: {float(post_training_summary.get('best_avg_block_power_violation', 0.0)):.6f}",
        (
            "Final feasible rollout-query fraction: "
            f"{float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f}"
        ),
        f"Per-user final lagrangian: {post_training_summary.get('per_user_final_lagrangian', [])}",
        f"Per-user best lagrangian: {post_training_summary.get('per_user_best_lagrangian', [])}",
        f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
        f"Per-user final rate violation: {post_training_summary.get('per_user_final_rate_violation', [])}",
        "Global active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Global active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('global_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Final epoch rollout query summary:",
        f"{post_training_summary.get('final_epoch_rollout_query_summary', {})}",
    ]
    lines.extend(format_experiment_cost_lines(post_training_summary.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- channel episode: one (seed, block) block realization stored in the base dataset",
            "- active-user channel episode: one active user inside one stored channel episode",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
        ]
    )
    return lines


def _build_summary_lines(result: dict[str, object], cfg_path: str, test_seed: int) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})

    lines = [
        "Downlink optimizer summary",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {cfg_path}",
        f"Test seed: {int(test_seed)}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Experiment scenario mode: {result.get('experiment_scenario_mode', 'unknown')}",
        f"Training channel-episode counts per user: {result.get('training_channel_episode_counts_per_user', result.get('training_active_user_case_counts_per_user', result.get('training_dataset_sizes', [])))}",
        f"Training dataset total channel episodes: {int(dataset_summary.get('total_channel_episodes', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Downlink precoder-net scope: {result.get('downlink_precoder_net_scope', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        "",
        "Training summary",
    ]
    training_history = result.get("precoder_net_training_history", {})
    if isinstance(training_history, dict) and training_history.get("sum_rate"):
        sum_rate_hist = training_history.get("sum_rate", [])
        avg_user_rate_hist = training_history.get("avg_user_rate", [])
        lines.extend(
            [
                (
                    f"Train target-bits mode: {post_training_summary.get('train_target_bits_mode', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Train target-bits mode: unknown"
                ),
                (
                    f"Train target bits summary: {post_training_summary.get('train_target_bits_summary', {})}"
                    if isinstance(post_training_summary, dict)
                    else "Train target bits summary: {}"
                ),
                f"Final epoch avg sum rate: {float(sum_rate_hist[-1]):.6f}",
                f"Final epoch avg user rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Final epoch avg user rate: n/a",
                f"Final epoch avg lagrangian: {float(training_history.get('avg_lagrangian', [])[-1]):.6f}" if training_history.get("avg_lagrangian") else "Final epoch avg lagrangian: n/a",
                f"Final epoch avg rate violation: {float(training_history.get('avg_rate_violation_over_users', [])[-1]):.6f}" if training_history.get("avg_rate_violation_over_users") else "Final epoch avg rate violation: n/a",
                f"Final epoch avg block-power violation: {float(training_history.get('avg_block_power_violation', [])[-1]):.6f}" if training_history.get("avg_block_power_violation") else "Final epoch avg block-power violation: n/a",
                f"Final epoch per-user rates: {training_history.get('per_user_rate', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_rate', [])]}",
                f"Final epoch per-user lagrangian: {training_history.get('per_user_lagrangian', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_lagrangian', [])]}",
                f"Final epoch per-user rate violation: {training_history.get('avg_rate_violation', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('avg_rate_violation', [])]}",
                (
                    f"Final feasible rollout-query fraction: "
                    f"{float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f}"
                    if isinstance(post_training_summary, dict)
                    else "Final feasible rollout-query fraction: n/a"
                ),
                f"Global active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}" if isinstance(post_training_summary, dict) else "Global active-user rollout queries by n_kl over all epochs: n/a",
                f"Per-user active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}" if isinstance(post_training_summary, dict) else "Per-user active-user rollout queries by n_kl over all epochs: n/a",
                f"Per-user active-user frontier rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}" if isinstance(post_training_summary, dict) else "Per-user active-user frontier rollout queries by n_kl over all epochs: n/a",
                "",
            ]
        )
    else:
        lines.extend(["Training metrics: n/a", ""])

    lines.extend([
        "Latency summary",
        f"Initial total latency: {metrics['initial_total_latency']:.6f}",
        f"Final total latency: {metrics['final_total_latency']:.6f}",
        f"Total latency reduction (%): {metrics['total_latency_reduction_percent']:.4f}",
        f"Initial avg latency: {metrics['initial_avg_latency']:.6f}",
        f"Final avg latency: {metrics['final_avg_latency']:.6f}",
        f"Initial min/max latency: {metrics['initial_min_latency']:.6f} / {metrics['initial_max_latency']:.6f}",
        f"Final min/max latency: {metrics['final_min_latency']:.6f} / {metrics['final_max_latency']:.6f}",
        "",
        "Asynchronality summary",
        f"Initial asynchronality sum: {metrics['initial_asynchronality_sum']:.6f}",
        f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
        f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
        "",
        "Link quality summary",
        f"Initial avg SNR (dB): {metrics['initial_avg_snr_db']:.4f}",
        f"Final avg SNR (dB): {metrics['final_avg_snr_db']:.4f}",
        f"Initial avg block SINR (dB): {metrics['initial_avg_sinr_db']:.4f}",
        f"Final avg block SINR (dB): {metrics['final_avg_sinr_db']:.4f}",
        "",
        "Per-user details",
    ])

    for row in metrics["per_user_summary"]:
        parts = [
            f"User {row['user']}",
            f"init_lat={row['initial_latency']:.6f}",
            f"final_lat={row['final_latency']:.6f}",
            f"lat_red={row['latency_reduction_percent']:.4f}%",
            f"init_block_sinr={row['initial_sinr_db']:.4f} dB",
            f"final_block_sinr={row['final_sinr_db']:.4f} dB",
            f"blocks={row['blocks']}",
            f"total_n={row['total_n']}",
            f"served_bits={row['served_bits']}",
            f"skipped_blocks={row['skipped_blocks']}",
        ]
        if metrics.get("scenario_mode", "") == FIXED_BLOCK_TARGETS_MODE:
            parts.extend(
                [
                    f"target_bits={row.get('target_bits', 0)}",
                    f"unserved_bits={row.get('unserved_bits', 0)}",
                    f"partial_blocks={row.get('partially_served_blocks', 0)}",
                    f"zero_service_blocks={row.get('zero_service_blocks', 0)}",
                ]
            )
        lines.append(" | ".join(parts))

    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- channel episode: one (seed, block) block realization stored in the base dataset",
            "- active-user channel episode: one active user inside one stored channel episode",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
        ]
    )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline downlink Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="config_downlink_example.yaml", help="Path to a YAML config")
    parser.add_argument("--train_seeds", type=str, default="0,1,2", help="Comma-separated Monte Carlo training seeds")
    parser.add_argument("--test_seed", type=int, default=3, help="Deterministic test seed")
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", "--policy_epochs", dest="precoder_net_epochs", type=int, default=40)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", "--policy_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", "--policy_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    verbose = not args.quiet
    train_seeds = parse_seed_list(args.train_seeds)
    configure_determinism(train_seeds[0] if train_seeds else 0)

    system_params, sim_params, run_meta = load_config(args.cfg_name)
    training_scenario_summaries = [
        build_experiment_scenario_summary(scenario)
        for scenario in build_experiment_scenarios_for_seeds(system_params, sim_params, train_seeds)
    ]
    scope_tag = str(sim_params.get("downlink_precoder_net_scope", "per_user_nets")).strip().lower().replace(" ", "_").replace("-", "_")
    result_tag = make_method_result_tag(
        f"monte_carlo_precoder_net_train_test_scope_{scope_tag}",
        run_meta["cfg_stem"],
        seed=args.test_seed,
    )
    output_dirs = build_downlink_result_dirs("Monte Carlo", result_tag)
    output_root = output_dirs["experiment_root"]

    training_start = perf_counter()
    training_scenarios = build_training_dataset(
        train_seeds,
        system_params,
        sim_params,
        verbose=verbose,
    )
    user_models, precoder_net_training_history, training_dataset_sizes = train_blocklength_aware_precoder_net(
        system_params,
        sim_params,
        training_scenarios,
        epochs=args.precoder_net_epochs,
        batch_size=args.precoder_net_batch_size,
        lr=args.precoder_net_lr,
        verbose=verbose,
    )
    training_wall_time_seconds = perf_counter() - training_start
    dataset_summary = precoder_net_training_history.get("dataset_summary", {})
    post_training_summary = precoder_net_training_history.get("post_training_summary", {})
    artifact = build_precoder_net_artifact(
        system_params,
        sim_params,
        train_seeds,
        user_models,
        precoder_net_training_history,
        training_dataset_sizes,
    )
    training_cost = build_downlink_monte_carlo_training_cost(
        artifact,
        batch_size=args.precoder_net_batch_size,
        core_wall_time_seconds_training=training_wall_time_seconds,
    )
    post_training_summary["experiment_cost"] = training_cost

    configure_determinism(int(args.test_seed))
    testing_start = perf_counter()
    test_system = DownlinkSystem(system_params, seed=int(args.test_seed))
    test_scenario_summary = build_experiment_scenario_summary(
        build_experiment_scenario(system_params, sim_params, seed=int(args.test_seed))
    )
    result = evaluate_downlink_precoder_net(
        test_system,
        sim_params,
        user_models,
        verbose=verbose,
        precoder_net_training_history=precoder_net_training_history,
        train_seeds=train_seeds,
        training_dataset_sizes=training_dataset_sizes,
    )
    testing_wall_time_seconds = perf_counter() - testing_start
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(args.test_seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["training_dataset_summary"] = dataset_summary
    result["post_training_summary"] = post_training_summary
    result["experiment_scenario_mode"] = sim_params.get("experiment_scenario_mode", "payload_completion")
    result["experiment_scenario"] = test_scenario_summary
    result["training_experiment_scenarios"] = training_scenario_summaries
    result["training_objective"] = precoder_net_training_history.get(
        "training_objective",
        "lagrangian_sum_finite_blocklength_rate_with_fixed_target_bits_objective",
    )
    result["experiment_cost"] = build_downlink_monte_carlo_total_cost(
        artifact,
        result.get("evaluation_cost_counters", {}),
        batch_size=args.precoder_net_batch_size,
        core_wall_time_seconds_training=training_wall_time_seconds,
        core_wall_time_seconds_testing=testing_wall_time_seconds,
    )
    result["summary_metrics"] = _compute_summary_metrics(result)

    plot_user_config(system_params, output_dirs["user_config"])
    plot_latency(result, output_dirs["latency_asynchronality"])
    plot_asynchronality_comparison(result, output_dirs["latency_asynchronality"])
    plot_link_quality(result, output_dirs["link_quality"])
    plot_blocks(result, output_dirs["schedule_details"])
    plot_rate_violation_heatmap(result, output_dirs["optimization_history"])
    plot_optimization_history(result, output_dirs["optimization_history"])
    plot_per_user_schedule_details(result, output_dirs["schedule_details"])
    plot_per_user_convergence(result, output_dirs["optimization_history"])
    plot_blocklength_feasibility_curves(test_system, result, output_dirs["optimization_history"])
    plot_interference_before_after_heatmaps(result, output_dirs["interference"])
    plot_per_user_interference_before_after(result, output_dirs["interference"])
    plot_interference_heatmaps(test_system, output_dirs["interference"])
    plot_per_user_interference_profiles(test_system, output_dirs["interference"])

    artifact["training_dataset_summary"] = dataset_summary
    artifact["post_training_summary"] = post_training_summary
    artifact["experiment_scenario_mode"] = sim_params.get("experiment_scenario_mode", "payload_completion")
    artifact["training_experiment_scenarios"] = training_scenario_summaries
    artifact["experiment_cost"] = training_cost
    torch.save(artifact, os.path.join(output_dirs["train_data"], "train_artifact.pt"))
    save_json(dataset_summary, os.path.join(output_dirs["train_data"], "training_dataset_summary.json"))
    save_text(
        _build_dataset_summary_lines(dataset_summary),
        os.path.join(output_dirs["train_data"], "training_dataset_summary.txt"),
    )
    save_json(post_training_summary, os.path.join(output_dirs["train_data"], "post_training_summary.json"))
    save_text(
        _build_post_training_summary_lines(post_training_summary),
        os.path.join(output_dirs["train_data"], "post_training_summary.txt"),
    )
    save_json(
        {"seed_scenarios": training_scenario_summaries},
        os.path.join(output_dirs["train_data"], "experiment_scenarios.json"),
    )
    save_text(
        _build_seeded_scenario_collection_lines(
            training_scenario_summaries,
            title="Training experiment scenarios by seed",
        ),
        os.path.join(output_dirs["train_data"], "experiment_scenarios.txt"),
    )
    save_json(result, os.path.join(output_dirs["test_data"], "result.json"))
    save_text(
        _build_summary_lines(result, run_meta["cfg_path"], int(args.test_seed)),
        os.path.join(output_dirs["test_data"], "summary.txt"),
    )
    save_json(test_scenario_summary, os.path.join(output_dirs["test_data"], "experiment_scenario.json"))
    save_text(
        build_experiment_scenario_summary_lines(test_scenario_summary),
        os.path.join(output_dirs["test_data"], "experiment_scenario.txt"),
    )
    print(f"Saved downlink precoder-net results to: {output_root}")


if __name__ == "__main__":
    main()
