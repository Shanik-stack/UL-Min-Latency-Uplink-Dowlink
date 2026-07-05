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
from experiment_utils import (
    compact_method_tag,
    current_local_timestamp,
    compact_shared_n_target_mode_tag,
    compact_scope_tag,
    join_compact_tag_parts,
    make_method_result_tag,
    resolve_monte_carlo_train_and_test_seeds,
)
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
from project_paths import build_downlink_result_dirs, mirror_experiment_root_to_result_aliases
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
        f"Base dataset kind: {dataset_summary.get('base_dataset_kind', 'unknown')}",
        f"Training scenario modes: {dataset_summary.get('scenario_modes', [])}",
        f"Training channel episodes by seed: {dataset_summary.get('channel_episodes_by_seed', {})}",
        f"Training channel episodes by block: {dataset_summary.get('channel_episodes_by_block', {})}",
        f"Training channel episodes by active user count: {dataset_summary.get('channel_episodes_by_active_user_count', {})}",
        f"Training channel episodes by active mask: {dataset_summary.get('channel_episodes_by_active_mask', {})}",
        f"Active-user channel episodes per user: {dataset_summary.get('channel_episodes_per_user', [])}",
        "",
        "Terminology",
        "- channel episode: one (seed, block=0) block realization stored in the base dataset",
        "- active-user channel episode: one active user inside one stored channel episode",
    ]


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    lines = [
        "Downlink post-training summary",
        f"Run started at: {post_training_summary.get('run_started_at_local', 'unknown')}",
        f"Training started at: {post_training_summary.get('training_started_at_local', 'unknown')}",
        f"Training completed at: {post_training_summary.get('training_completed_at_local', 'unknown')}",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Epochs completed: {int(post_training_summary.get('epochs_completed', 0))}",
        f"Training solve status: {post_training_summary.get('training_solve_status', 'unknown')}",
        f"Restored solution source: {post_training_summary.get('restored_solution_source', 'unknown')}",
        f"Downlink precoder-net scope: {post_training_summary.get('downlink_precoder_net_scope', 'unknown')}",
        (
            "BS-shared fixed-target n-target mode: "
            f"{post_training_summary.get('bs_shared_net_fixed_target_n_target_mode', 'not_applicable')}"
        ),
        f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}",
        f"Training channel episodes: {int(post_training_summary.get('total_training_channel_episodes', 0))}",
        f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}",
        f"Final KKT primal residual: {float(post_training_summary.get('final_kkt_primal_residual', 0.0)):.6e}",
        f"Final KKT complementarity residual: {float(post_training_summary.get('final_kkt_complementarity_residual', 0.0)):.6e}",
        f"Final KKT stationarity residual: {float(post_training_summary.get('final_kkt_stationarity_residual', 0.0)):.6e}",
        f"Last epoch rollout-weighted avg sum rate: {float(post_training_summary.get('last_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('final_avg_sum_rate', 0.0))):.6f}",
        f"Best epoch rollout-weighted avg sum rate: {float(post_training_summary.get('best_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('best_avg_sum_rate', 0.0))):.6f}",
        f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', 0.0))):.6f}",
        f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', 0.0))):.6f}",
        f"Last epoch mean per-user Lagrangian: {float(post_training_summary.get('last_epoch_mean_user_lagrangian', post_training_summary.get('final_avg_lagrangian', 0.0))):.6f}",
        f"Best epoch mean per-user Lagrangian: {float(post_training_summary.get('best_epoch_mean_user_lagrangian', post_training_summary.get('best_avg_lagrangian', 0.0))):.6f}",
        (
            "Last epoch feasible rollout queries: "
            f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
            f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)} "
            f"({float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f})"
        ),
        (
            "Per-user last epoch avg Lagrangian over active rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_lagrangian_over_active_rollout_queries', post_training_summary.get('per_user_final_lagrangian', []))}"
        ),
        f"Per-user best Lagrangian: {post_training_summary.get('per_user_best_lagrangian', [])}",
        (
            "Per-user last epoch avg rate over active rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_rate_over_active_rollout_queries', post_training_summary.get('per_user_final_rate', []))}"
        ),
        "Global active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Global active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('global_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Last epoch active-user rollout queries by n_kl:",
        f"{post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_active_user_rollout_queries_by_n_kl', {})}",
        "Last epoch active-user frontier rollout queries by n_kl:",
        f"{post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_active_user_frontier_rollout_queries_by_n_kl', {})}",
    ]
    lines.extend(format_experiment_cost_lines(post_training_summary.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- channel episode: one (seed, block=0) block realization stored in the base dataset",
            "- active-user channel episode: one active user inside one stored channel episode",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
            "- last epoch rollout-weighted avg sum rate: average sum rate over the visited rollout queries in the last epoch, using the trainer's rollout weights",
            "- last epoch mean per-user rollout rate: first average each user's rate over that user's active rollout queries in the last epoch, then average over users",
        ]
    )
    return lines


def _sum_per_user_summary_field(metrics: dict[str, object], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in metrics.get("per_user_summary", [])))


def _build_final_test_summary_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "Final test results",
        f"Initial total latency: {metrics['initial_total_latency']:.6f}",
        f"Final total latency: {metrics['final_total_latency']:.6f}",
        f"Total latency reduction (%): {metrics['total_latency_reduction_percent']:.4f}",
        f"Initial avg latency: {metrics['initial_avg_latency']:.6f}",
        f"Final avg latency: {metrics['final_avg_latency']:.6f}",
        f"Initial asynchronality sum: {metrics['initial_asynchronality_sum']:.6f}",
        f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
        f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
        f"Initial avg SNR (dB): {metrics['initial_avg_snr_db']:.4f}",
        f"Final avg SNR (dB): {metrics['final_avg_snr_db']:.4f}",
        f"Initial avg block SINR (dB): {metrics['initial_avg_sinr_db']:.4f}",
        f"Final avg block SINR (dB): {metrics['final_avg_sinr_db']:.4f}",
        f"Total served bits: {_sum_per_user_summary_field(metrics, 'served_bits')}",
        f"Total skipped blocks: {_sum_per_user_summary_field(metrics, 'skipped_blocks')}",
    ]
    if metrics.get("scenario_mode", "") == FIXED_BLOCK_TARGETS_MODE:
        lines.extend(
            [
                f"Total target bits: {_sum_per_user_summary_field(metrics, 'target_bits')}",
                f"Total unserved bits: {_sum_per_user_summary_field(metrics, 'unserved_bits')}",
                f"Total partially served blocks: {_sum_per_user_summary_field(metrics, 'partially_served_blocks')}",
                f"Total zero-service blocks: {_sum_per_user_summary_field(metrics, 'zero_service_blocks')}",
            ]
        )
    return lines


def _build_per_user_test_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    lines = ["Per-user final test details"]
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
    return lines


def _build_summary_lines(result: dict[str, object], cfg_path: str, test_seed: int) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})

    lines = [
        "Downlink optimizer summary",
        "",
        "Setup",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {cfg_path}",
        f"Test seed: {int(test_seed)}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('experiment_scenario_mode', 'unknown')}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Downlink precoder-net scope: {result.get('downlink_precoder_net_scope', 'unknown')}",
        (
            "BS-shared fixed-target n-target mode: "
            f"{result.get('bs_shared_net_fixed_target_n_target_mode', 'not_applicable')}"
        ),
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        f"Training dataset total channel episodes: {int(dataset_summary.get('total_channel_episodes', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Training channel-episode counts per user: {result.get('training_channel_episode_counts_per_user', result.get('training_active_user_case_counts_per_user', result.get('training_dataset_sizes', [])))}",
    ]
    lines.extend([""])
    lines.extend(_build_final_test_summary_lines(result))
    training_history = result.get("precoder_net_training_history", {})
    if isinstance(training_history, dict) and training_history.get("sum_rate"):
        sum_rate_hist = training_history.get("sum_rate", [])
        avg_user_rate_hist = training_history.get("avg_user_rate", [])
        lines.extend(
            [
                "",
                "Training results",
                f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}" if isinstance(post_training_summary, dict) else "Epochs requested: 0",
                (
                    f"Epochs completed: {int(post_training_summary.get('epochs_completed', 0))}"
                    if isinstance(post_training_summary, dict)
                    else "Epochs completed: 0"
                ),
                (
                    f"Training solve status: {post_training_summary.get('training_solve_status', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Training solve status: unknown"
                ),
                (
                    f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Base training dataset: unknown"
                ),
                (
                    f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Rollout anchor-bits mode: unknown"
                ),
                (
                    f"Last epoch rollout-weighted avg sum rate: {float(post_training_summary.get('last_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('final_avg_sum_rate', float(sum_rate_hist[-1])))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else f"Last epoch rollout-weighted avg sum rate: {float(sum_rate_hist[-1]):.6f}"
                ),
                (
                    f"Best epoch rollout-weighted avg sum rate: {float(post_training_summary.get('best_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('best_avg_sum_rate', float(sum_rate_hist[-1])))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else f"Best epoch rollout-weighted avg sum rate: {float(sum_rate_hist[-1]):.6f}"
                ),
                (
                    f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', float(avg_user_rate_hist[-1]) if avg_user_rate_hist else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Last epoch mean per-user rollout rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Last epoch mean per-user rollout rate: n/a")
                ),
                (
                    f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', float(avg_user_rate_hist[-1]) if avg_user_rate_hist else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Best epoch mean per-user rollout rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Best epoch mean per-user rollout rate: n/a")
                ),
                (
                    f"Last epoch mean per-user Lagrangian: {float(post_training_summary.get('last_epoch_mean_user_lagrangian', post_training_summary.get('final_avg_lagrangian', float(training_history.get('avg_lagrangian', [])[-1]) if training_history.get('avg_lagrangian') else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Last epoch mean per-user Lagrangian: {float(training_history.get('avg_lagrangian', [])[-1]):.6f}" if training_history.get("avg_lagrangian") else "Last epoch mean per-user Lagrangian: n/a")
                ),
                (
                    f"Best epoch mean per-user Lagrangian: {float(post_training_summary.get('best_epoch_mean_user_lagrangian', post_training_summary.get('best_avg_lagrangian', float(training_history.get('avg_lagrangian', [])[-1]) if training_history.get('avg_lagrangian') else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Best epoch mean per-user Lagrangian: {float(training_history.get('avg_lagrangian', [])[-1]):.6f}" if training_history.get("avg_lagrangian") else "Best epoch mean per-user Lagrangian: n/a")
                ),
                (
                    "Per-user last epoch avg rate over active rollout queries: "
                    f"{training_history.get('per_user_rate', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_rate', [])]}"
                ),
                (
                    "Per-user last epoch avg Lagrangian over active rollout queries: "
                    f"{training_history.get('per_user_lagrangian', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_lagrangian', [])]}"
                ),
                (
                    "Last epoch feasible rollout queries: "
                    f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
                    f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)} "
                    f"({float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f})"
                    if isinstance(post_training_summary, dict)
                    else "Last epoch feasible rollout queries: n/a"
                ),
            ]
        )
    else:
        lines.extend(["", "Training results", "Training metrics: n/a"])

    lines.extend([""])
    lines.extend(_build_per_user_test_lines(result))

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
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "",
                "Additional training details",
                f"Global active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}",
                f"Per-user active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}",
                f"Per-user active-user frontier rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}",
            ]
        )

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- channel episode: one (seed, block=0) block realization stored in the base dataset",
            "- active-user channel episode: one active user inside one stored channel episode",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
        ]
    )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline downlink Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="config_downlink_example.yaml", help="Path to a YAML config")
    parser.add_argument("--train_seeds", type=str, default=None, help="Explicit comma-separated training seeds")
    parser.add_argument(
        "--num_train_seeds",
        type=int,
        default=None,
        help="Build training seeds as 1..N excluding test_seed",
    )
    parser.add_argument("--test_seed", type=int, default=None, help="Deterministic Monte Carlo test seed")
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", "--policy_epochs", dest="precoder_net_epochs", type=int, default=None)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", "--policy_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", "--policy_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    verbose = not args.quiet
    system_params, sim_params, run_meta = load_config(args.cfg_name)
    run_started_at_local = current_local_timestamp()
    train_epochs = int(
        args.precoder_net_epochs
        if args.precoder_net_epochs is not None
        else sim_params.get("monte_carlo_training_max_epochs", sim_params.get("max_epochs", 100))
    )
    train_seeds, test_seed = resolve_monte_carlo_train_and_test_seeds(
        cli_train_seeds=args.train_seeds,
        cli_num_train_seeds=args.num_train_seeds,
        cli_test_seed=args.test_seed,
        config_train_seeds=sim_params.get("monte_carlo_train_seeds"),
        config_num_train_seeds=sim_params.get("monte_carlo_num_train_seeds"),
        config_test_seed=sim_params.get("monte_carlo_test_seed"),
    )
    configure_determinism(train_seeds[0] if train_seeds else 0)
    print(f"Resolved Monte Carlo train seeds: {train_seeds}")
    print(f"Resolved Monte Carlo test seed: {int(test_seed)}")
    training_scenario_summaries = [
        build_experiment_scenario_summary(scenario)
        for scenario in build_experiment_scenarios_for_seeds(system_params, sim_params, train_seeds)
    ]
    scope_name = sim_params.get("downlink_precoder_net_scope", "per_user_nets")
    scope_tag = compact_scope_tag(scope_name)
    shared_n_target_mode_tag = None
    if (
        str(scope_name) == "bs_shared_net"
        and str(sim_params.get("experiment_scenario_mode", "payload_completion")) == FIXED_BLOCK_TARGETS_MODE
    ):
        shared_n_target_mode_tag = compact_shared_n_target_mode_tag(
            sim_params.get("bs_shared_net_fixed_target_n_target_mode", "shared_n_targets")
        )
    result_tag = make_method_result_tag(
        join_compact_tag_parts(
            compact_method_tag("monte_carlo_precoder_net_train_test"),
            scope_tag,
            shared_n_target_mode_tag,
        ),
        run_meta["cfg_stem"],
        seed=int(test_seed),
    )
    output_dirs = build_downlink_result_dirs("Monte Carlo", result_tag)
    output_root = output_dirs["experiment_root"]

    training_started_at_local = current_local_timestamp()
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
        epochs=train_epochs,
        batch_size=args.precoder_net_batch_size,
        lr=args.precoder_net_lr,
        verbose=verbose,
    )
    training_wall_time_seconds = perf_counter() - training_start
    training_completed_at_local = current_local_timestamp()
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
    post_training_summary["run_started_at_local"] = str(run_started_at_local)
    post_training_summary["training_started_at_local"] = str(training_started_at_local)
    post_training_summary["training_completed_at_local"] = str(training_completed_at_local)

    configure_determinism(int(test_seed))
    testing_started_at_local = current_local_timestamp()
    testing_start = perf_counter()
    test_system = DownlinkSystem(system_params, seed=int(test_seed))
    test_scenario_summary = build_experiment_scenario_summary(
        build_experiment_scenario(system_params, sim_params, seed=int(test_seed))
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
    testing_completed_at_local = current_local_timestamp()
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(test_seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["training_dataset_summary"] = dataset_summary
    result["post_training_summary"] = post_training_summary
    result["experiment_scenario_mode"] = sim_params.get("experiment_scenario_mode", "payload_completion")
    result["experiment_scenario"] = test_scenario_summary
    result["training_experiment_scenarios"] = training_scenario_summaries
    result["training_objective"] = precoder_net_training_history.get(
        "training_objective",
        "lagrangian_sum_finite_blocklength_rate_with_online_full_block_anchor_bits",
    )
    result["experiment_cost"] = build_downlink_monte_carlo_total_cost(
        artifact,
        result.get("evaluation_cost_counters", {}),
        batch_size=args.precoder_net_batch_size,
        core_wall_time_seconds_training=training_wall_time_seconds,
        core_wall_time_seconds_testing=testing_wall_time_seconds,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = str(testing_completed_at_local)
    result["training_started_at_local"] = str(training_started_at_local)
    result["training_completed_at_local"] = str(training_completed_at_local)
    result["testing_started_at_local"] = str(testing_started_at_local)
    result["testing_completed_at_local"] = str(testing_completed_at_local)
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
        _build_summary_lines(result, run_meta["cfg_path"], int(test_seed)),
        os.path.join(output_dirs["test_data"], "summary.txt"),
    )
    save_json(test_scenario_summary, os.path.join(output_dirs["test_data"], "experiment_scenario.json"))
    save_text(
        build_experiment_scenario_summary_lines(test_scenario_summary),
        os.path.join(output_dirs["test_data"], "experiment_scenario.txt"),
    )
    mirror_paths = mirror_experiment_root_to_result_aliases(
        link_name="Downlink",
        scenario_mode=str(sim_params.get("experiment_scenario_mode", "payload_completion")),
        method_name="Monte Carlo",
        source_experiment_root=output_root,
    )
    print(f"Mirrored downlink precoder-net scenario results to: {mirror_paths['scenario_root']}")
    print(f"Saved downlink precoder-net method results to: {mirror_paths['method_root']}")


if __name__ == "__main__":
    main()
