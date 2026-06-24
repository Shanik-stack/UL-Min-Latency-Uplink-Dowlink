from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    plot_blocklength_sweep_curves,
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


def _build_dataset_summary_lines(dataset_summary: dict[str, object]) -> list[str]:
    return [
        "Downlink training dataset summary",
        f"Total training cases: {int(dataset_summary.get('total_training_cases', 0))}",
        f"Training cases by seed: {dataset_summary.get('training_cases_by_seed', {})}",
        f"Training cases by block: {dataset_summary.get('training_cases_by_block', {})}",
        f"Training cases by active user count: {dataset_summary.get('training_cases_by_active_user_count', {})}",
        f"Training cases by active mask: {dataset_summary.get('training_cases_by_active_mask', {})}",
        f"Active user-cases per user: {dataset_summary.get('active_user_cases_per_user', [])}",
        f"Global active user-cases by initial n_kl: {dataset_summary.get('global_active_user_cases_by_initial_n_kl', {})}",
        "Per-user active user-cases by initial n_kl:",
        f"{dataset_summary.get('per_user_active_user_cases_by_initial_n_kl', [])}",
        "",
        "Terminology",
        "- training case: one (seed, block, active-mask) item",
        "- active user-case: one active user inside one training case",
    ]


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    return [
        "Downlink post-training summary",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Train minimum required bits per active user-case: {int(post_training_summary.get('train_min_bits_required', 1))}",
        f"Final avg sum rate: {float(post_training_summary.get('final_avg_sum_rate', 0.0)):.6f}",
        f"Best avg sum rate: {float(post_training_summary.get('best_avg_sum_rate', 0.0)):.6f}",
        f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
        f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
        f"Final avg lagrangian: {float(post_training_summary.get('final_avg_lagrangian', 0.0)):.6f}",
        f"Best avg lagrangian: {float(post_training_summary.get('best_avg_lagrangian', 0.0)):.6f}",
        f"Final avg rate violation: {float(post_training_summary.get('final_avg_rate_violation', 0.0)):.6f}",
        f"Best avg rate violation: {float(post_training_summary.get('best_avg_rate_violation', 0.0)):.6f}",
        f"Final avg power violation: {float(post_training_summary.get('final_avg_power_violation', 0.0)):.6f}",
        f"Best avg power violation: {float(post_training_summary.get('best_avg_power_violation', 0.0)):.6f}",
        (
            "Final feasible training-case fraction: "
            f"{float(post_training_summary.get('final_feasible_training_case_fraction', 0.0)):.6f}"
        ),
        f"Total curriculum reduction events: {int(post_training_summary.get('total_curriculum_reduction_events', 0))}",
        f"Per-user final lagrangian: {post_training_summary.get('per_user_final_lagrangian', [])}",
        f"Per-user best lagrangian: {post_training_summary.get('per_user_best_lagrangian', [])}",
        f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
        f"Per-user final rate violation: {post_training_summary.get('per_user_final_rate_violation', [])}",
        f"Per-user final power violation: {post_training_summary.get('per_user_final_power_violation', [])}",
        "Global active user-case uses by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_training_uses_by_n_kl', {}).get('global_active_user_case_uses_by_n_kl_over_all_epochs', {})}",
        "Per-user active user-case uses by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_training_uses_by_n_kl', {}).get('per_user_active_user_case_uses_by_n_kl_over_all_epochs', [])}",
        "Global active user-cases by final n_kl:",
        f"{post_training_summary.get('final_training_case_n_kl_summary', {}).get('global_active_user_cases_by_final_n_kl', {})}",
        "Per-user active user-cases by final n_kl:",
        f"{post_training_summary.get('final_training_case_n_kl_summary', {}).get('per_user_active_user_cases_by_final_n_kl', [])}",
        "",
        "Terminology",
        "- training case: one (seed, block, active-mask) item",
        "- active user-case: one active user inside one training case",
        "- training use: one active user-case consumed once in one epoch at its current n_kl",
    ]


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
        f"Training active user-cases per user: {result.get('training_active_user_case_counts_per_user', result.get('training_dataset_sizes', []))}",
        f"Training dataset total cases: {int(dataset_summary.get('total_training_cases', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
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
                f"Train minimum required bits per active user-case: {int(post_training_summary.get('train_min_bits_required', 1))}" if isinstance(post_training_summary, dict) else "Train minimum required bits per active user-case: 1",
                f"Final epoch avg sum rate: {float(sum_rate_hist[-1]):.6f}",
                f"Final epoch avg user rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Final epoch avg user rate: n/a",
                f"Final epoch avg lagrangian: {float(training_history.get('avg_lagrangian', [])[-1]):.6f}" if training_history.get("avg_lagrangian") else "Final epoch avg lagrangian: n/a",
                f"Final epoch avg rate violation: {float(training_history.get('avg_rate_violation_over_users', [])[-1]):.6f}" if training_history.get("avg_rate_violation_over_users") else "Final epoch avg rate violation: n/a",
                f"Final epoch avg power violation: {float(training_history.get('avg_power_violation_over_users', [])[-1]):.6f}" if training_history.get("avg_power_violation_over_users") else "Final epoch avg power violation: n/a",
                f"Final epoch per-user rates: {training_history.get('per_user_rate', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_rate', [])]}",
                f"Final epoch per-user lagrangian: {training_history.get('per_user_lagrangian', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_lagrangian', [])]}",
                f"Final epoch per-user rate violation: {training_history.get('avg_rate_violation', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('avg_rate_violation', [])]}",
                f"Final epoch per-user power violation: {training_history.get('avg_power_violation', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('avg_power_violation', [])]}",
                (
                    f"Final feasible training-case fraction: "
                    f"{float(post_training_summary.get('final_feasible_training_case_fraction', 0.0)):.6f}"
                    if isinstance(post_training_summary, dict)
                    else "Final feasible training-case fraction: n/a"
                ),
                f"Global active user-case uses by n_kl over all epochs: {post_training_summary.get('cumulative_training_uses_by_n_kl', {}).get('global_active_user_case_uses_by_n_kl_over_all_epochs', {})}" if isinstance(post_training_summary, dict) else "Global active user-case uses by n_kl over all epochs: n/a",
                f"Per-user active user-case uses by n_kl over all epochs: {post_training_summary.get('cumulative_training_uses_by_n_kl', {}).get('per_user_active_user_case_uses_by_n_kl_over_all_epochs', [])}" if isinstance(post_training_summary, dict) else "Per-user active user-case uses by n_kl over all epochs: n/a",
                f"Per-user active user-cases by final n_kl: {post_training_summary.get('final_training_case_n_kl_summary', {}).get('per_user_active_user_cases_by_final_n_kl', [])}" if isinstance(post_training_summary, dict) else "Per-user active user-cases by final n_kl: n/a",
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
        f"Initial avg SINR (dB): {metrics['initial_avg_sinr_db']:.4f}",
        f"Final avg SINR (dB): {metrics['final_avg_sinr_db']:.4f}",
        "",
        "Per-user details",
    ])

    for row in metrics["per_user_summary"]:
        lines.append(
            " | ".join(
                [
                    f"User {row['user']}",
                    f"init_lat={row['initial_latency']:.6f}",
                    f"final_lat={row['final_latency']:.6f}",
                    f"lat_red={row['latency_reduction_percent']:.4f}%",
                    f"init_sinr={row['initial_sinr_db']:.4f} dB",
                    f"final_sinr={row['final_sinr_db']:.4f} dB",
                    f"blocks={row['blocks']}",
                    f"total_n={row['total_n']}",
                    f"served_bits={row['served_bits']}",
                    f"skipped_blocks={row['skipped_blocks']}",
                ]
            )
        )

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

    lines.extend(
        [
            "",
            "Terminology",
            "- training case: one (seed, block, active-mask) item",
            "- active user-case: one active user inside one training case",
            "- training use: one active user-case consumed once in one epoch at its current n_kl",
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
    result_tag = make_method_result_tag("monte_carlo_precoder_net_train_test", run_meta["cfg_stem"], seed=args.test_seed)
    output_dirs = build_downlink_result_dirs("Monte Carlo", result_tag)
    output_root = output_dirs["experiment_root"]

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
    dataset_summary = precoder_net_training_history.get("dataset_summary", {})
    post_training_summary = precoder_net_training_history.get("post_training_summary", {})

    test_system = DownlinkSystem(system_params, seed=int(args.test_seed))
    result = evaluate_downlink_precoder_net(
        test_system,
        sim_params,
        user_models,
        verbose=verbose,
        precoder_net_training_history=precoder_net_training_history,
        train_seeds=train_seeds,
        training_dataset_sizes=training_dataset_sizes,
    )
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(args.test_seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["training_dataset_summary"] = dataset_summary
    result["post_training_summary"] = post_training_summary
    result["training_objective"] = precoder_net_training_history.get(
        "training_objective",
        "lagrangian_sum_finite_blocklength_rate_with_fixed_min_bits_objective",
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
    plot_blocklength_sweep_curves(test_system, result, output_dirs["optimization_history"])
    plot_interference_before_after_heatmaps(result, output_dirs["interference"])
    plot_per_user_interference_before_after(result, output_dirs["interference"])
    plot_interference_heatmaps(test_system, output_dirs["interference"])
    plot_per_user_interference_profiles(test_system, output_dirs["interference"])

    artifact = build_precoder_net_artifact(
        system_params,
        sim_params,
        train_seeds,
        user_models,
        precoder_net_training_history,
        training_dataset_sizes,
    )
    artifact["training_dataset_summary"] = dataset_summary
    artifact["post_training_summary"] = post_training_summary
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
    save_json(result, os.path.join(output_dirs["test_data"], "result.json"))
    save_text(
        _build_summary_lines(result, run_meta["cfg_path"], int(args.test_seed)),
        os.path.join(output_dirs["test_data"], "summary.txt"),
    )
    print(f"Saved downlink precoder-net results to: {output_root}")


if __name__ == "__main__":
    main()
