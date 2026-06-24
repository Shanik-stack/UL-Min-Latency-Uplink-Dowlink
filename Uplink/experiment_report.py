from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _pairwise_latency_diffs(latencies: Sequence[float]) -> tuple[list[list[float]], list[dict[str, float]], float]:
    arr = [float(x) for x in latencies]
    K = len(arr)
    matrix = [[abs(arr[i] - arr[j]) for j in range(K)] for i in range(K)]
    pair_details: list[dict[str, float]] = []
    async_sum = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            diff = float(matrix[i][j])
            async_sum += diff
            pair_details.append({"user_i": int(i), "user_j": int(j), "abs_latency_diff": diff})
    return matrix, pair_details, float(async_sum)


def compute_summary_metrics(result: dict[str, Any]) -> dict[str, Any]:
    initial_latency = [float(x) for x in result["initial_latency"]]
    final_latency = [float(x) for x in result["final_latency"]]
    K = len(final_latency)

    latency_reduction_per_user_percent: list[float] = []
    for init_val, final_val in zip(initial_latency, final_latency):
        if init_val > 0:
            reduction = ((init_val - final_val) / init_val) * 100.0
        else:
            reduction = 0.0
        latency_reduction_per_user_percent.append(float(reduction))

    initial_total_latency = float(sum(initial_latency))
    final_total_latency = float(sum(final_latency))
    if initial_total_latency > 0:
        total_latency_reduction_percent = ((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0
    else:
        total_latency_reduction_percent = 0.0

    initial_async_matrix, initial_async_pairs, initial_async_sum = _pairwise_latency_diffs(initial_latency)
    final_async_matrix, final_async_pairs, final_async_sum = _pairwise_latency_diffs(final_latency)
    if initial_async_sum > 0:
        async_reduction_percent = ((initial_async_sum - final_async_sum) / initial_async_sum) * 100.0
    else:
        async_reduction_percent = 0.0

    initial_snr_db = [float(x) for x in result.get("initial_snr_db", [])]
    final_snr_db = [float(x) for x in result.get("final_snr_db", [])]
    initial_sinr_db = [float(x) for x in result.get("initial_sinr_db", [])]
    final_sinr_db = [float(x) for x in result.get("final_sinr_db", [])]
    blocks_per_user = [len(v) for v in result.get("final_n_kl", [[] for _ in range(K)])]
    total_n = [int(x) for x in result.get("final_n", [0 for _ in range(K)])]
    served_bits = [int(sum(v)) for v in result.get("B_kl", [[] for _ in range(K)])]

    per_user_summary = []
    for k in range(K):
        per_user_summary.append(
            {
                "user": int(k),
                "initial_latency": initial_latency[k],
                "final_latency": final_latency[k],
                "latency_reduction_percent": latency_reduction_per_user_percent[k],
                "initial_snr_db": initial_snr_db[k] if k < len(initial_snr_db) else 0.0,
                "final_snr_db": final_snr_db[k] if k < len(final_snr_db) else 0.0,
                "initial_sinr_db": initial_sinr_db[k] if k < len(initial_sinr_db) else 0.0,
                "final_sinr_db": final_sinr_db[k] if k < len(final_sinr_db) else 0.0,
                "blocks": int(blocks_per_user[k]),
                "total_n": int(total_n[k]),
                "served_bits": int(served_bits[k]),
            }
        )

    return {
        "initial_total_latency": initial_total_latency,
        "final_total_latency": final_total_latency,
        "initial_avg_latency": float(initial_total_latency / max(K, 1)),
        "final_avg_latency": float(final_total_latency / max(K, 1)),
        "initial_max_latency": float(max(initial_latency) if initial_latency else 0.0),
        "final_max_latency": float(max(final_latency) if final_latency else 0.0),
        "initial_min_latency": float(min(initial_latency) if initial_latency else 0.0),
        "final_min_latency": float(min(final_latency) if final_latency else 0.0),
        "latency_reduction_per_user_percent": latency_reduction_per_user_percent,
        "total_latency_reduction_percent": float(total_latency_reduction_percent),
        "initial_asynchronality_matrix": initial_async_matrix,
        "final_asynchronality_matrix": final_async_matrix,
        "initial_asynchronality_pairs": initial_async_pairs,
        "final_asynchronality_pairs": final_async_pairs,
        "initial_asynchronality_sum": float(initial_async_sum),
        "final_asynchronality_sum": float(final_async_sum),
        "asynchronality_reduction_percent": float(async_reduction_percent),
        "initial_avg_snr_db": float(sum(initial_snr_db) / max(len(initial_snr_db), 1)),
        "final_avg_snr_db": float(sum(final_snr_db) / max(len(final_snr_db), 1)),
        "initial_avg_sinr_db": float(sum(initial_sinr_db) / max(len(initial_sinr_db), 1)),
        "final_avg_sinr_db": float(sum(final_sinr_db) / max(len(final_sinr_db), 1)),
        "per_user_summary": per_user_summary,
    }


def build_precoder_net_result(
    test_uplinksystem: Any,
    test_data_dict: dict[str, Any],
    *,
    method_name: str,
    cfg_path: str,
    test_seed: int,
    train_seeds: Sequence[int],
    train_artifact: dict[str, Any],
    initial_R_fbl: Sequence[Any],
    initial_n_kl: Sequence[Any],
    initial_n: Sequence[float],
    initial_latency: Sequence[float],
    initial_snr_db: Sequence[float],
    initial_sinr_db: Sequence[float],
    initial_bits_per_symbol: Sequence[float],
    initial_B_kl: Sequence[Sequence[int]] | None = None,
    initial_bits_per_symbol_by_block: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    _, final_snr_db = test_uplinksystem.get_SNR()
    _, final_sinr_db = test_uplinksystem.get_SINR()
    final_bits_per_symbol = [
        np.asarray(test_data_dict["B_kl_star_test"][user], dtype=np.float64)
        / np.asarray(test_uplinksystem.n_kl[user], dtype=np.float64)
        for user in range(test_uplinksystem.K)
    ]

    result = {
        "method_name": method_name,
        "cfg_path": cfg_path,
        "seed": int(test_seed),
        "train_seeds": [int(v) for v in train_seeds],
        "training_dataset_sizes": [int(v) for v in train_artifact.get("training_dataset_sizes", [])],
        "training_sample_counts_per_user": [
            int(v)
            for v in train_artifact.get(
                "training_sample_counts_per_user",
                train_artifact.get("training_dataset_sizes", []),
            )
        ],
        "training_dataset_summary": train_artifact.get("training_dataset_summary", {}),
        "post_training_summary": train_artifact.get("post_training_summary", {}),
        "precoder_net_training_losses": train_artifact.get(
            "precoder_net_training_losses",
            train_artifact.get(
                "precoder_training_losses",
                train_artifact.get("policy_training_losses", []),
            ),
        ),
        "precoder_net_training_history": train_artifact.get("precoder_net_training_history", {}),
        "precoder_parameterization": train_artifact.get("precoder_parameterization", "unknown"),
        "training_objective": train_artifact.get("training_objective", "unknown"),
        "user_model_specs": train_artifact.get("user_model_specs", []),
        "initial_latency": list(map(float, initial_latency)),
        "final_latency": list(map(float, test_uplinksystem.latency)),
        "initial_n": list(map(float, initial_n)),
        "final_n": list(map(int, test_uplinksystem.n)),
        "initial_n_kl": [list(map(float, np.atleast_1d(v))) for v in initial_n_kl],
        "final_n_kl": [list(map(int, v)) for v in test_uplinksystem.n_kl],
        "initial_B_kl": [list(map(int, values)) for values in (initial_B_kl or [[] for _ in range(test_uplinksystem.K)])],
        "initial_R_fbl": [np.asarray(v).tolist() for v in initial_R_fbl],
        "final_R_fbl": [np.asarray(v).tolist() for v in test_uplinksystem.R_fbl],
        "initial_snr_db": list(map(float, initial_snr_db)),
        "final_snr_db": list(map(float, final_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "final_sinr_db": list(map(float, final_sinr_db)),
        "initial_bits_per_symbol": list(map(float, initial_bits_per_symbol)),
        "initial_bits_per_symbol_by_block": [
            list(map(float, values))
            for values in (initial_bits_per_symbol_by_block or [[] for _ in range(test_uplinksystem.K)])
        ],
        "final_bits_per_symbol": [list(map(float, vals)) for vals in final_bits_per_symbol],
        "B_kl": [list(map(int, values)) for values in test_data_dict["B_kl_star_test"]],
        "n_kl": [list(map(int, values)) for values in test_data_dict["n_star_test"]],
        "R_fbl": [list(map(float, values)) for values in test_data_dict["R_star_test"]],
        "blocks_per_user": [len(v) for v in test_uplinksystem.n_kl],
        "initial_schedule_source": "random_precoder_baseline",
    }
    result["summary_metrics"] = compute_summary_metrics(result)
    return result


def build_precoder_result(*args, **kwargs):
    return build_precoder_net_result(*args, **kwargs)


def build_policy_result(*args, **kwargs):
    return build_precoder_net_result(*args, **kwargs)


def build_summary_lines(result: dict[str, Any]) -> list[str]:
    metrics = result["summary_metrics"]
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})
    lines = [
        "Uplink optimizer summary",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {result.get('cfg_path', 'unknown')}",
        f"Test seed: {int(result.get('seed', 0))}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Training sample counts per user: {result.get('training_sample_counts_per_user', result.get('training_dataset_sizes', []))}",
        f"Training dataset total examples: {int(dataset_summary.get('total_examples', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'unknown')}",
        "",
    ]
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "Training summary",
                f"Train minimum required bits per sample: {int(post_training_summary.get('train_min_bits_required', 1))}",
                f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
                f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
                f"Final avg lagrangian: {float(post_training_summary.get('final_avg_lagrangian', 0.0)):.6f}",
                f"Best avg lagrangian: {float(post_training_summary.get('best_avg_lagrangian', 0.0)):.6f}",
                f"Final avg rate violation: {float(post_training_summary.get('final_avg_rate_violation', 0.0)):.6f}",
                f"Final avg power violation: {float(post_training_summary.get('final_avg_power_violation', 0.0)):.6f}",
                f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
                f"Per-user final lagrangian: {post_training_summary.get('per_user_final_lagrangian', post_training_summary.get('per_user_final_loss', []))}",
                f"Per-user final rate violation: {post_training_summary.get('per_user_final_rate_violation', [])}",
                f"Per-user final power violation: {post_training_summary.get('per_user_final_power_violation', [])}",
                f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
                f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
                f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
                f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
                (
                    "Train-eval total latency reduction (%): "
                    f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
                ),
                (
                    f"Train-eval initial selected n_kl summary: "
                    f"{post_training_summary.get('train_eval_initial_selected_n_kl_summary', {})}"
                ),
                (
                    f"Train-eval final selected n_kl summary: "
                    f"{post_training_summary.get('train_eval_selected_n_kl_summary', {})}"
                ),
                "",
            ]
        )
    lines.extend([
        "Testing summary",
        f"Initial latency source: {result.get('initial_schedule_source', 'unknown')}",
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
    ])
    for row in metrics["per_user_summary"]:
        lines.append(
            " | ".join(
                [
                    f"User {row['user']}",
                    f"init_lat={row['initial_latency']:.6f}",
                    f"final_lat={row['final_latency']:.6f}",
                    f"lat_red={row['latency_reduction_percent']:.4f}%",
                    f"init_snr={row['initial_snr_db']:.4f} dB",
                    f"final_snr={row['final_snr_db']:.4f} dB",
                    f"init_sinr={row['initial_sinr_db']:.4f} dB",
                    f"final_sinr={row['final_sinr_db']:.4f} dB",
                    f"blocks={row['blocks']}",
                    f"total_n={row['total_n']}",
                    f"served_bits={row['served_bits']}",
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
            "- training sample: one (seed, user, block, n_kl) item used for one user-net update",
            "- initial schedule: the random-precoder baseline used for the before-optimization uplink latency",
        ]
    )
    return lines


def build_training_dataset_summary_lines(dataset_summary: dict[str, Any]) -> list[str]:
    lines = [
        "Uplink training dataset summary",
        f"Total training samples: {int(dataset_summary.get('total_examples', 0))}",
        f"Training samples by seed: {dataset_summary.get('examples_by_seed', {})}",
        f"Training samples by block: {dataset_summary.get('global_examples_by_block', {})}",
        f"Training samples by minimum required bits: {dataset_summary.get('global_examples_by_min_bits_required', {})}",
        f"Global training samples by n_kl: {dataset_summary.get('global_examples_by_n_kl', {})}",
        "",
        "Per-user training sample details",
    ]
    for user_summary in dataset_summary.get("per_user", []):
        lines.append(
            " | ".join(
                [
                    f"User {int(user_summary.get('user', 0))}",
                    f"training_samples={int(user_summary.get('total_examples', 0))}",
                    f"unique_n_kl={user_summary.get('unique_n_kl', [])}",
                    f"training_samples_by_block={user_summary.get('examples_by_block', {})}",
                    f"training_samples_by_min_bits_required={user_summary.get('examples_by_min_bits_required', {})}",
                    f"training_samples_by_n_kl={user_summary.get('examples_by_n_kl', {})}",
                    f"training_samples_by_seed={user_summary.get('examples_by_seed', {})}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "Terminology",
            "- training sample: one (seed, user, block, n_kl) item used for one user-net update",
        ]
    )
    return lines


def build_post_training_summary_lines(post_training_summary: dict[str, Any]) -> list[str]:
    lines = [
        "Uplink post-training summary",
        f"Train-eval seed: {int(post_training_summary.get('train_eval_seed', 0))}",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Train minimum required bits per sample: {int(post_training_summary.get('train_min_bits_required', 1))}",
        f"Per-user num epochs: {post_training_summary.get('per_user_num_epochs', [])}",
        f"Final avg user rate: {float(post_training_summary.get('final_avg_user_rate', 0.0)):.6f}",
        f"Best avg user rate: {float(post_training_summary.get('best_avg_user_rate', 0.0)):.6f}",
        f"Final avg lagrangian: {float(post_training_summary.get('final_avg_lagrangian', 0.0)):.6f}",
        f"Best avg lagrangian: {float(post_training_summary.get('best_avg_lagrangian', 0.0)):.6f}",
        f"Final avg rate violation: {float(post_training_summary.get('final_avg_rate_violation', 0.0)):.6f}",
        f"Best avg rate violation: {float(post_training_summary.get('best_avg_rate_violation', 0.0)):.6f}",
        f"Final avg power violation: {float(post_training_summary.get('final_avg_power_violation', 0.0)):.6f}",
        f"Best avg power violation: {float(post_training_summary.get('best_avg_power_violation', 0.0)):.6f}",
        f"Per-user final rate: {post_training_summary.get('per_user_final_rate', [])}",
        f"Per-user final lagrangian: {post_training_summary.get('per_user_final_lagrangian', post_training_summary.get('per_user_final_loss', []))}",
        f"Per-user best lagrangian: {post_training_summary.get('per_user_best_lagrangian', post_training_summary.get('per_user_best_loss', []))}",
        f"Per-user final rate violation: {post_training_summary.get('per_user_final_rate_violation', [])}",
        f"Per-user final power violation: {post_training_summary.get('per_user_final_power_violation', [])}",
        f"Train-eval initial latency: {post_training_summary.get('train_eval_initial_latency', [])}",
        f"Train-eval final latency: {post_training_summary.get('train_eval_final_latency', [])}",
        f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
        f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
        f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
        f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
        f"Train-eval initial served bits per user: {post_training_summary.get('train_eval_initial_served_bits_per_user', [])}",
        f"Train-eval final served bits per user: {post_training_summary.get('train_eval_served_bits_per_user', [])}",
        (
            "Train-eval total latency reduction (%): "
            f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
        ),
        "Train-eval initial selected n_kl summary:",
        f"{post_training_summary.get('train_eval_initial_selected_n_kl_summary', {})}",
        "Train-eval selected n_kl summary:",
        f"{post_training_summary.get('train_eval_selected_n_kl_summary', {})}",
        "",
        "Terminology",
        "- train-eval: evaluation of the trained precoder nets on the first training seed",
        "- initial schedule: the random-precoder baseline used for the before-training latency",
    ]
    return lines
