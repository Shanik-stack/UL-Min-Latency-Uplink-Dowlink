from __future__ import annotations

import os
from typing import Callable

import numpy as np

from config_loader import load_config
from determinism import configure_determinism
from downlink_system import DownlinkSystem
from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE
from experiment_utils import make_method_result_tag
from optimizer import (
    optimize_downlink_safe_sweep,
    resolve_safe_sweep_objective_mode,
    safe_sweep_objective_tag,
)
from plotting import (
    initialize_output_dirs,
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
from utils import save_json, save_text


OPTIMIZERS: dict[str, Callable] = {
    "greedy_safe_sweep": optimize_downlink_safe_sweep,
}


def build_result_tag(
    method_name: str,
    cfg_stem: str,
    seed: int,
    *,
    objective_mode: str | None = None,
) -> str:
    method_tag = method_name
    if objective_mode:
        method_tag = f"{method_name}_{safe_sweep_objective_tag(objective_mode)}"
    return make_method_result_tag(method_tag, cfg_stem, seed=seed)


def _pairwise_latency_diffs(latencies: list[float]) -> tuple[list[list[float]], list[dict[str, float]], float]:
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


def _mean_valid_rows(values: object, K: int) -> tuple[list[float], float, bool]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)

    per_user = [0.0 for _ in range(K)]
    global_values: list[float] = []
    has_any = False
    for k in range(K):
        row = arr[k] if k < arr.shape[0] else np.asarray([], dtype=float)
        valid = row[np.isfinite(row)]
        if valid.size > 0:
            per_user[k] = float(np.mean(valid))
            global_values.extend(valid.tolist())
            has_any = True

    global_mean = float(np.mean(global_values)) if global_values else 0.0
    return per_user, global_mean, has_any


def _compute_summary_metrics(result: dict) -> dict:
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

    skipped_blocks_per_user = [0 for _ in range(K)]
    for point in result.get("rate_points", []):
        if bool(point.get("skipped", False)):
            skipped_blocks_per_user[int(point["user"])] += 1

    n_totals = [int(sum(v)) for v in result.get("n_kl", [[] for _ in range(K)])]
    bits_totals = [int(sum(v)) for v in result.get("B_kl", [[] for _ in range(K)])]
    blocks_per_user = [int(v) for v in result.get("blocks_per_user", [0 for _ in range(K)])]
    final_sinr_db_raw = [float(x) for x in result.get("final_sinr_db", [0.0 for _ in range(K)])]
    initial_sinr_db_raw = [float(x) for x in result.get("initial_sinr_db", [0.0 for _ in range(K)])]
    initial_block_sinr_db, initial_avg_block_sinr_db, has_initial_block_sinr = _mean_valid_rows(
        result.get("initial_interference_diag", {}).get("sinr_db", []),
        K,
    )
    final_block_sinr_db, final_avg_block_sinr_db, has_final_block_sinr = _mean_valid_rows(
        result.get("final_interference_diag", {}).get("sinr_db", []),
        K,
    )
    if not has_initial_block_sinr:
        initial_block_sinr_db = list(initial_sinr_db_raw)
        initial_avg_block_sinr_db = float(sum(initial_sinr_db_raw) / max(len(initial_sinr_db_raw), 1))
    if not has_final_block_sinr:
        final_block_sinr_db = list(final_sinr_db_raw)
        final_avg_block_sinr_db = float(sum(final_sinr_db_raw) / max(len(final_sinr_db_raw), 1))
    scenario_mode = str(result.get("scenario_mode", ""))
    scenario_block_targets = np.asarray(result.get("scenario_block_targets", []), dtype=int)
    target_bits_per_user = [0 for _ in range(K)]
    unserved_bits_per_user = [0 for _ in range(K)]
    partially_served_blocks_per_user = [0 for _ in range(K)]
    zero_service_blocks_per_user = [0 for _ in range(K)]

    if scenario_mode == FIXED_BLOCK_TARGETS_MODE and scenario_block_targets.ndim == 2 and scenario_block_targets.shape[0] == K:
        target_bits_per_user = list(map(int, scenario_block_targets.sum(axis=1, dtype=int)))
        served_by_user = [list(map(int, result.get("B_kl", [[] for _ in range(K)])[k])) for k in range(K)]
        for k in range(K):
            targets = scenario_block_targets[int(k)].tolist()
            served = served_by_user[int(k)] if int(k) < len(served_by_user) else []
            unserved_bits_per_user[int(k)] = int(sum(max(int(t) - int(s), 0) for t, s in zip(targets, served)))
            partially_served_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, served) if int(t) > 0 and 0 < int(s) < int(t))
            )
            zero_service_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, served) if int(t) > 0 and int(s) <= 0)
            )

    per_user_summary = []
    for k in range(K):
        per_user_summary.append(
            {
                "user": int(k),
                "initial_latency": initial_latency[k],
                "final_latency": final_latency[k],
                "latency_reduction_percent": latency_reduction_per_user_percent[k],
                "initial_sinr_db": initial_block_sinr_db[k],
                "final_sinr_db": final_block_sinr_db[k],
                "initial_sinr_db_raw": initial_sinr_db_raw[k],
                "final_sinr_db_raw": final_sinr_db_raw[k],
                "blocks": blocks_per_user[k],
                "total_n": n_totals[k],
                "target_bits": int(target_bits_per_user[k]),
                "served_bits": bits_totals[k],
                "unserved_bits": int(unserved_bits_per_user[k]),
                "partially_served_blocks": int(partially_served_blocks_per_user[k]),
                "zero_service_blocks": int(zero_service_blocks_per_user[k]),
                "skipped_blocks": int(skipped_blocks_per_user[k]),
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
        "initial_avg_sinr_db": float(initial_avg_block_sinr_db),
        "final_avg_sinr_db": float(final_avg_block_sinr_db),
        "initial_avg_sinr_db_raw": float(sum(initial_sinr_db_raw) / max(len(initial_sinr_db_raw), 1)),
        "final_avg_sinr_db_raw": float(sum(final_sinr_db_raw) / max(len(final_sinr_db_raw), 1)),
        "initial_sinr_db_per_user": initial_block_sinr_db,
        "final_sinr_db_per_user": final_block_sinr_db,
        "initial_avg_snr_db": float(sum(result.get("initial_snr_db", [])) / max(len(result.get("initial_snr_db", [])), 1)),
        "final_avg_snr_db": float(sum(result.get("final_snr_db", [])) / max(len(result.get("final_snr_db", [])), 1)),
        "scenario_mode": scenario_mode,
        "target_bits_per_user": target_bits_per_user,
        "unserved_bits_per_user": unserved_bits_per_user,
        "partially_served_blocks_per_user": partially_served_blocks_per_user,
        "zero_service_blocks_per_user": zero_service_blocks_per_user,
        "skipped_blocks_per_user": skipped_blocks_per_user,
        "per_user_summary": per_user_summary,
    }


def run_downlink_experiment(
    method_name: str,
    cfg_name: str,
    seed: int,
    verbose: bool = True,
    *,
    output_root: str | None = None,
) -> dict:
    if method_name not in OPTIMIZERS:
        known = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown method '{method_name}'. Expected one of: {known}")

    configure_determinism(seed)
    system_params, sim_params, run_meta = load_config(cfg_name)
    objective_mode_tag = (
        resolve_safe_sweep_objective_mode(sim_params)
        if method_name == "greedy_safe_sweep"
        else None
    )
    result_tag = build_result_tag(
        method_name,
        run_meta["cfg_stem"],
        seed,
        objective_mode=objective_mode_tag,
    )
    if output_root is None:
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", result_tag)
    output_dirs = initialize_output_dirs(output_root)

    system = DownlinkSystem(system_params, seed=seed)
    result = OPTIMIZERS[method_name](system, sim_params, verbose=verbose)
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
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
    plot_blocklength_sweep_curves(system, result, output_dirs["optimization_history"])
    plot_interference_before_after_heatmaps(result, output_dirs["interference"])
    plot_per_user_interference_before_after(result, output_dirs["interference"])
    plot_interference_heatmaps(system, output_dirs["interference"])
    plot_per_user_interference_profiles(system, output_dirs["interference"])

    save_json(result, os.path.join(output_dirs["data"], "result.json"))

    metrics = result["summary_metrics"]
    lines = [
        "Downlink optimizer summary",
        f"Method: {method_name}",
        f"Config: {run_meta['cfg_path']}",
        f"Seed: {seed}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
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
        f"Initial avg block SINR (dB): {metrics['initial_avg_sinr_db']:.4f}",
        f"Final avg block SINR (dB): {metrics['final_avg_sinr_db']:.4f}",
        "",
        "Per-user details",
    ]
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

    save_text(lines, os.path.join(output_dirs["data"], "summary.txt"))
    print(f"Saved downlink results to: {output_root}")
    return result
