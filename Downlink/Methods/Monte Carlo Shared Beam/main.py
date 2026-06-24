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
from experiment_runner import _compute_summary_metrics
from experiment_utils import make_method_result_tag, parse_seed_list
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
from policy_optimizer import (
    build_precoder_net_artifact,
    build_training_dataset,
    evaluate_shared_beam_precoder_net,
    train_shared_beam_precoder_net,
)
from project_paths import build_downlink_result_dirs
from utils import save_json, save_text


def _build_dataset_summary_lines(dataset_summary: dict[str, object]) -> list[str]:
    lines = [
        "Downlink shared-beam training dataset summary",
        f"Total training cases: {int(dataset_summary.get('total_training_cases', 0))}",
        f"Total n-level evaluations: {int(dataset_summary.get('total_n_level_evaluations', 0))}",
        f"Total active-user n evaluations: {int(dataset_summary.get('total_active_user_n_evaluations', 0))}",
        f"Training cases by seed: {dataset_summary.get('training_cases_by_seed', {})}",
        f"Training cases by block: {dataset_summary.get('training_cases_by_block', {})}",
        f"Training cases by active user count: {dataset_summary.get('training_cases_by_active_user_count', {})}",
        f"Training cases by active mask: {dataset_summary.get('training_cases_by_active_mask', {})}",
        f"Active user-cases per user: {dataset_summary.get('active_user_cases_per_user', [])}",
        f"Global active-user n evaluations by n_kl: {dataset_summary.get('global_active_user_n_evaluations_by_n_kl', {})}",
        "Per-user active-user n evaluations by n_kl:",
        f"{dataset_summary.get('per_user_active_user_n_evaluations_by_n_kl', [])}",
        "",
        "Terminology",
        "- training case: one (seed, block, active-mask) item",
        "- n-level evaluation: one candidate joint n_kl vector reused with the same predicted beams",
        "- active-user n evaluation: one active user counted once at one candidate n_kl inside one training case",
    ]
    return lines


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    return [
        "Downlink shared-beam post-training summary",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Final avg loss: {float(post_training_summary.get('final_avg_loss', 0.0)):.6f}",
        f"Best avg loss: {float(post_training_summary.get('best_avg_loss', 0.0)):.6f}",
        f"Final avg sum rate: {float(post_training_summary.get('final_avg_sum_rate', 0.0)):.6f}",
        f"Best avg sum rate: {float(post_training_summary.get('best_avg_sum_rate', 0.0)):.6f}",
        f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
        f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
        f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
        f"Per-user best rate: {post_training_summary.get('per_user_best_rate', [])}",
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
    ]

    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "Training summary",
                f"Final avg loss: {float(post_training_summary.get('final_avg_loss', 0.0)):.6f}",
                f"Best avg loss: {float(post_training_summary.get('best_avg_loss', 0.0)):.6f}",
                f"Final avg sum rate: {float(post_training_summary.get('final_avg_sum_rate', 0.0)):.6f}",
                f"Best avg sum rate: {float(post_training_summary.get('best_avg_sum_rate', 0.0)):.6f}",
                f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
                f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
                f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
                f"Per-user best rate: {post_training_summary.get('per_user_best_rate', [])}",
                "",
            ]
        )

    lines.extend(
        [
            "Testing summary",
            "",
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
        ]
    )

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
            "- active user-case: one user being active inside one training case",
            "- active-user n evaluation: one active user counted once at one candidate n_kl inside one training case",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline downlink Monte Carlo shared-beam train/test")
    parser.add_argument("--cfg_name", type=str, default="config_downlink_example.yaml", help="Path to a YAML config")
    parser.add_argument("--train_seeds", type=str, default="0,1,2", help="Comma-separated Monte Carlo training seeds")
    parser.add_argument("--test_seed", type=int, default=3, help="Deterministic test seed")
    parser.add_argument("--precoder_net_epochs", "--precoder_epochs", dest="precoder_net_epochs", type=int, default=40)
    parser.add_argument("--precoder_net_batch_size", "--precoder_batch_size", dest="precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", "--precoder_lr", dest="precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    parser.add_argument("--skip_test", action="store_true", help="Train only and skip the test pass")
    args = parser.parse_args()

    verbose = not args.quiet
    train_seeds = parse_seed_list(args.train_seeds)
    configure_determinism(train_seeds[0] if train_seeds else 0)

    system_params, sim_params, run_meta = load_config(args.cfg_name)
    result_tag = make_method_result_tag("monte_carlo_shared_beam_train_test", run_meta["cfg_stem"], seed=args.test_seed)
    output_dirs = build_downlink_result_dirs("Monte Carlo Shared Beam", result_tag)
    output_root = output_dirs["experiment_root"]

    training_cases = build_training_dataset(
        train_seeds,
        system_params,
        sim_params,
        verbose=verbose,
    )
    user_models, precoder_net_training_history, training_dataset_sizes = train_shared_beam_precoder_net(
        system_params,
        sim_params,
        training_cases,
        epochs=args.precoder_net_epochs,
        batch_size=args.precoder_net_batch_size,
        lr=args.precoder_net_lr,
        verbose=verbose,
    )
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

    if not args.skip_test:
        test_system = DownlinkSystem(system_params, seed=int(args.test_seed))
        result = evaluate_shared_beam_precoder_net(
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
            "average_shared_beam_sum_fbl_rate_over_candidate_n_grid",
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

        save_json(result, os.path.join(output_dirs["test_data"], "result.json"))
        save_text(
            _build_summary_lines(result, run_meta["cfg_path"], int(args.test_seed)),
            os.path.join(output_dirs["test_data"], "summary.txt"),
        )

    print(f"Saved downlink shared-beam results to: {output_root}")


if __name__ == "__main__":
    main()
