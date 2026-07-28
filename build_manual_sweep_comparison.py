from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
COMPARISON_ROOT = RESULTS_ROOT / "Decision Sweep Comparisons"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return None


def _sum_numeric_list(value: Any) -> float | None:
    if not isinstance(value, list):
        return None
    total = 0.0
    found = False
    for item in value:
        if isinstance(item, (int, float)) and not (isinstance(item, float) and (math.isnan(item) or math.isinf(item))):
            total += float(item)
            found = True
    return total if found else None


def _detect_link(result_path: Path) -> str:
    parts = {part.lower() for part in result_path.parts}
    if "uplink" in parts:
        return "uplink"
    if "downlink" in parts:
        return "downlink"
    raise ValueError(f"Could not detect link from path: {result_path}")


def _detect_method(result_path: Path) -> str:
    parts = [part.lower() for part in result_path.parts]
    if "method-monte carlo" in parts:
        return "monte_carlo"
    if "method-convergence per epoch" in parts:
        return "convergence"
    raise ValueError(f"Could not detect method from path: {result_path}")


def _run_name_from_path(result_path: Path) -> str:
    parts = list(result_path.parts)
    for idx, part in enumerate(parts):
        lower = part.lower()
        if lower in {"testing", "training"} and idx > 0:
            return parts[idx - 1]
    if result_path.parent.name == "data" and len(parts) >= 2:
        return parts[-3]
    return result_path.parent.parent.name


def _extract_direction(run_name: str) -> str:
    match = re.search(r"ndir[_-](asc|desc)", run_name)
    if match:
        return "ascending" if match.group(1) == "asc" else "descending"
    return "unknown"


def _normalize_objective(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    lowered = value.lower()
    if lowered in {"unweighted_sum_rate", "unweighted"}:
        return "unweighted_sum_rate"
    if lowered in {"inverse_cnr_weighted_sum_rate", "inverse_cnr", "weighted"}:
        return "inverse_cnr_weighted_sum_rate"
    return value


def _baseline_field(summary_metrics: dict[str, Any], baseline_name: str, field_name: str) -> Any:
    baseline = summary_metrics.get("baseline_comparison_metrics", {}) or {}
    entry = baseline.get(baseline_name, {}) or {}
    return entry.get(field_name)


def _extract_row(result_path: Path) -> dict[str, Any]:
    result = _load_json(result_path)
    summary = result.get("summary_metrics", {}) or {}
    cost = result.get("experiment_cost", {}) or {}
    post_training = result.get("post_training_summary", {}) or {}

    run_name = _run_name_from_path(result_path)
    link = _detect_link(result_path)
    method = _detect_method(result_path)

    objective_mode = _normalize_objective(
        result.get("uplink_objective_mode")
        or result.get("objective_mode")
        or post_training.get("uplink_objective_mode")
        or post_training.get("objective_mode")
    )

    row: dict[str, Any] = {
        "run_name": run_name,
        "result_json_path": str(result_path),
        "link": link,
        "method": method,
        "scenario_mode": result.get("scenario_mode") or summary.get("scenario_mode") or result.get("experiment_scenario_mode") or "",
        "objective_mode": objective_mode,
        "n_search_direction": _extract_direction(run_name),
        "seed": result.get("seed"),
        "train_seeds": _json_text(result.get("train_seeds")),
        "train_seed_count": len(result.get("train_seeds") or []),
        "initial_schedule_source": result.get("initial_schedule_source", ""),
        "precoder_parameterization": result.get("precoder_parameterization", ""),
        "convergence_precoder_update_mode": result.get("convergence_precoder_update_mode", ""),
        "downlink_precoder_net_scope": result.get("downlink_precoder_net_scope", ""),
        "weight_strategy": result.get("weight_strategy", ""),
        "rollout_query_weighting_mode": result.get("rollout_query_weighting_mode", ""),
        "run_started_at_local": result.get("run_started_at_local", ""),
        "run_completed_at_local": result.get("run_completed_at_local", ""),
        "training_started_at_local": result.get("training_started_at_local", ""),
        "training_completed_at_local": result.get("training_completed_at_local", ""),
        "testing_started_at_local": result.get("testing_started_at_local", ""),
        "testing_completed_at_local": result.get("testing_completed_at_local", ""),
        "initial_total_latency": summary.get("initial_total_latency"),
        "final_total_latency": summary.get("final_total_latency"),
        "total_latency_reduction_percent": summary.get("total_latency_reduction_percent"),
        "initial_avg_latency": summary.get("initial_avg_latency"),
        "final_avg_latency": summary.get("final_avg_latency"),
        "initial_max_latency": summary.get("initial_max_latency"),
        "final_max_latency": summary.get("final_max_latency"),
        "initial_min_latency": summary.get("initial_min_latency"),
        "final_min_latency": summary.get("final_min_latency"),
        "initial_avg_sinr_db": summary.get("initial_avg_sinr_db"),
        "final_avg_sinr_db": summary.get("final_avg_sinr_db"),
        "initial_avg_snr_db": summary.get("initial_avg_snr_db"),
        "final_avg_snr_db": summary.get("final_avg_snr_db"),
        "initial_asynchronality_sum": summary.get("initial_asynchronality_sum"),
        "final_asynchronality_sum": summary.get("final_asynchronality_sum"),
        "asynchronality_reduction_percent": summary.get("asynchronality_reduction_percent"),
        "initial_total_served_blocks": summary.get("initial_total_served_blocks"),
        "final_total_served_blocks": summary.get("final_total_served_blocks"),
        "unserved_bits_total": _sum_numeric_list(summary.get("unserved_bits_per_user")),
        "skipped_blocks_total": _sum_numeric_list(summary.get("skipped_blocks_per_user")),
        "zero_service_blocks_total": _sum_numeric_list(summary.get("zero_service_blocks_per_user")),
        "partially_served_blocks_total": _sum_numeric_list(summary.get("partially_served_blocks_per_user")),
        "target_bits_per_user": _json_text(summary.get("target_bits_per_user")),
        "latency_reduction_per_user_percent": _json_text(summary.get("latency_reduction_per_user_percent")),
        "unserved_bits_per_user": _json_text(summary.get("unserved_bits_per_user")),
        "skipped_blocks_per_user": _json_text(summary.get("skipped_blocks_per_user")),
        "zero_service_blocks_per_user": _json_text(summary.get("zero_service_blocks_per_user")),
        "partially_served_blocks_per_user": _json_text(summary.get("partially_served_blocks_per_user")),
        "baseline_random_total_latency_reduction_percent": _baseline_field(summary, "random_precoder_baseline", "total_latency_reduction_percent"),
        "baseline_random_asynchronality_reduction_percent": _baseline_field(summary, "random_precoder_baseline", "asynchronality_reduction_percent"),
        "baseline_naive_full_t_total_latency_reduction_percent": _baseline_field(summary, "naive_full_T_baseline", "total_latency_reduction_percent"),
        "baseline_naive_full_t_asynchronality_reduction_percent": _baseline_field(summary, "naive_full_T_baseline", "asynchronality_reduction_percent"),
        "core_wall_time_seconds_total": cost.get("core_wall_time_seconds_total"),
        "core_wall_time_seconds_training": cost.get("core_wall_time_seconds_training"),
        "core_wall_time_seconds_testing": cost.get("core_wall_time_seconds_testing"),
        "forward_backward_nn_flops": cost.get("estimated_nn_training_flops"),
        "forward_only_nn_flops": cost.get("estimated_nn_inference_flops"),
        "total_nn_flops": cost.get("estimated_nn_total_flops"),
        "optimizer_updates": cost.get("optimizer_steps"),
        "gradient_eval_checks": cost.get("training_forward_backward_sample_equivalents"),
        "forward_only_beam_evaluations": cost.get("inference_forward_calls"),
        "training_epochs_requested": post_training.get("epochs_requested") or post_training.get("configured_max_epochs"),
        "training_epochs_completed": post_training.get("epochs_completed"),
        "training_solve_status": post_training.get("training_solve_status", ""),
        "training_base_dataset_kind": post_training.get("base_dataset_kind", ""),
        "training_rollout_anchor_bits_mode": post_training.get("rollout_anchor_bits_mode", ""),
        "training_rollout_query_weighting_mode": post_training.get("rollout_query_weighting_mode", ""),
        "training_total_channel_episodes": post_training.get("total_training_channel_episodes"),
        "training_final_avg_sum_rate": post_training.get("final_avg_sum_rate"),
        "training_best_avg_sum_rate": post_training.get("best_avg_sum_rate"),
        "training_final_avg_user_rate": post_training.get("final_avg_user_rate"),
        "training_best_avg_user_rate": post_training.get("best_avg_user_rate"),
        "training_final_avg_lagrangian": post_training.get("final_avg_lagrangian"),
        "training_best_avg_lagrangian": post_training.get("best_avg_lagrangian"),
        "training_final_avg_rate_violation": post_training.get("final_avg_rate_violation"),
        "training_best_avg_rate_violation": post_training.get("best_avg_rate_violation"),
        "training_final_avg_power_violation": post_training.get("final_avg_power_violation") or post_training.get("final_avg_block_power_violation"),
        "training_best_avg_power_violation": post_training.get("best_avg_power_violation") or post_training.get("best_avg_block_power_violation"),
        "training_final_feasible_rollout_query_fraction": post_training.get("final_feasible_rollout_query_fraction"),
        "training_last_epoch_total_rollout_queries": post_training.get("last_epoch_total_rollout_queries"),
        "training_last_epoch_feasible_rollout_queries": post_training.get("last_epoch_feasible_rollout_queries"),
        "training_last_epoch_infeasible_rollout_queries": post_training.get("last_epoch_infeasible_rollout_queries"),
        "training_final_kkt_primal_residual": post_training.get("final_kkt_primal_residual"),
        "training_final_kkt_complementarity_residual": post_training.get("final_kkt_complementarity_residual"),
        "training_final_kkt_stationarity_residual": post_training.get("final_kkt_stationarity_residual"),
        "training_per_user_final_rate": _json_text(post_training.get("per_user_final_rate")),
        "training_per_user_final_lagrangian": _json_text(post_training.get("per_user_final_lagrangian")),
        "training_per_user_best_lagrangian": _json_text(post_training.get("per_user_best_lagrangian")),
    }

    return row


def _discover_result_json_paths(sweep_id: str) -> list[Path]:
    patterns = [
        RESULTS_ROOT / "Uplink" / "Method-Convergence per epoch",
        RESULTS_ROOT / "Uplink" / "Method-Monte Carlo",
        RESULTS_ROOT / "Downlink" / "Method-Convergence per epoch",
        RESULTS_ROOT / "Downlink" / "Method-Monte Carlo",
    ]
    paths: list[Path] = []
    for root in patterns:
        if not root.exists():
            continue
        for result_path in root.glob(f"*{sweep_id}*/**/result.json"):
            if result_path.is_file():
                paths.append(result_path)
    deduped = sorted(set(paths))
    return deduped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_text(row.get(key)) for key in fieldnames})


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _best_rows_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["link"]), str(row["method"]))].append(row)
    best: list[dict[str, Any]] = []
    for (link, method), group_rows in sorted(groups.items()):
        ranked = sorted(
            group_rows,
            key=lambda item: (
                -(item.get("total_latency_reduction_percent") or float("-inf")),
                -(item.get("asynchronality_reduction_percent") or float("-inf")),
                item.get("final_total_latency") or float("inf"),
            ),
        )
        winner = dict(ranked[0])
        winner["group_link"] = link
        winner["group_method"] = method
        best.append(winner)
    return best


def _decision_average_rows(rows: list[dict[str, Any]], decision_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["link"]), str(row["method"]), str(row.get(decision_key, "")))].append(row)
    averages: list[dict[str, Any]] = []
    metrics = [
        "total_latency_reduction_percent",
        "asynchronality_reduction_percent",
        "final_total_latency",
        "final_avg_sinr_db",
        "final_avg_snr_db",
        "unserved_bits_total",
        "skipped_blocks_total",
        "core_wall_time_seconds_total",
    ]
    for (link, method, decision_value), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "link": link,
            "method": method,
            decision_key: decision_value,
            "run_count": len(group_rows),
        }
        for metric in metrics:
            values = [row.get(metric) for row in group_rows if isinstance(row.get(metric), (int, float))]
            summary[f"avg_{metric}"] = sum(values) / len(values) if values else None
            summary[f"best_{metric}"] = max(values) if values and "reduction" in metric else (min(values) if values else None)
        averages.append(summary)
    return averages


def _report_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(_json_text(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _write_markdown_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda item: (
            -(item.get("total_latency_reduction_percent") or float("-inf")),
            -(item.get("asynchronality_reduction_percent") or float("-inf")),
        ),
    )
    best_rows = _best_rows_by_group(rows)
    objective_averages = _decision_average_rows(rows, "objective_mode")
    direction_averages = _decision_average_rows(rows, "n_search_direction")
    lines = [
        f"# Sweep Comparison: {path.parent.parent.name}",
        "",
        f"Total completed runs: {len(rows)}",
        "",
        "## Best by link and method",
        _report_table(
            best_rows,
            [
                "group_link",
                "group_method",
                "objective_mode",
                "n_search_direction",
                "total_latency_reduction_percent",
                "asynchronality_reduction_percent",
                "final_total_latency",
                "final_avg_sinr_db",
                "final_avg_snr_db",
                "core_wall_time_seconds_total",
                "run_name",
            ],
        ),
        "",
        "## All runs sorted by latency reduction",
        _report_table(
            sorted_rows,
            [
                "link",
                "method",
                "objective_mode",
                "n_search_direction",
                "total_latency_reduction_percent",
                "asynchronality_reduction_percent",
                "final_total_latency",
                "final_avg_sinr_db",
                "final_avg_snr_db",
                "unserved_bits_total",
                "core_wall_time_seconds_total",
                "run_name",
            ],
        ),
        "",
        "## Objective averages",
        _report_table(
            objective_averages,
            [
                "link",
                "method",
                "objective_mode",
                "run_count",
                "avg_total_latency_reduction_percent",
                "avg_asynchronality_reduction_percent",
                "avg_final_total_latency",
                "avg_final_avg_sinr_db",
                "avg_final_avg_snr_db",
                "avg_core_wall_time_seconds_total",
            ],
        ),
        "",
        "## Direction averages",
        _report_table(
            direction_averages,
            [
                "link",
                "method",
                "n_search_direction",
                "run_count",
                "avg_total_latency_reduction_percent",
                "avg_asynchronality_reduction_percent",
                "avg_final_total_latency",
                "avg_final_avg_sinr_db",
                "avg_final_avg_snr_db",
                "avg_core_wall_time_seconds_total",
            ],
        ),
        "",
        "## Notes",
        "- `objective_mode` is taken from the saved result JSON, not only the folder name.",
        "- `n_search_direction` is inferred from the run folder tag.",
        "- Wide CSV/JSON files contain the fuller metric set, including baseline comparisons, FLOPs, runtime, and Monte Carlo training summaries.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparison bundle from saved sweep result files.")
    parser.add_argument("--sweep_id", required=True, help="Sweep folder id, for example sweep_20260727_215427")
    args = parser.parse_args()

    result_paths = _discover_result_json_paths(args.sweep_id)
    if not result_paths:
        raise SystemExit(f"No result.json files found for {args.sweep_id}")

    rows = [_extract_row(path) for path in result_paths]
    rows = sorted(
        rows,
        key=lambda item: (str(item["link"]), str(item["method"]), str(item["objective_mode"]), str(item["n_search_direction"])),
    )

    comparison_root = COMPARISON_ROOT / args.sweep_id
    overall_root = comparison_root / "overall"
    overall_root.mkdir(parents=True, exist_ok=True)

    _write_csv(overall_root / "comparison_summary.csv", rows)
    _write_json(overall_root / "comparison_summary.json", rows)
    _write_markdown_report(overall_root / "comparison_report.md", rows)

    best_rows = _best_rows_by_group(rows)
    _write_csv(overall_root / "best_by_link_method.csv", best_rows)
    _write_json(overall_root / "best_by_link_method.json", best_rows)

    for decision_key in ("objective_mode", "n_search_direction"):
        decision_root = comparison_root / "by_decision" / decision_key
        averages = _decision_average_rows(rows, decision_key)
        _write_csv(decision_root / "decision_averages.csv", averages)
        _write_json(decision_root / "decision_averages.json", averages)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(decision_key, ""))].append(row)
        for decision_value, value_rows in sorted(grouped.items()):
            value_root = decision_root / decision_value
            _write_csv(value_root / "comparison_summary.csv", value_rows)
            _write_json(value_root / "comparison_summary.json", value_rows)
            _write_markdown_report(value_root / "comparison_report.md", value_rows)

    group_root = comparison_root / "by_group"
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['link']}__{row['method']}"
        grouped_runs[key].append(row)
    for group_name, group_rows in sorted(grouped_runs.items()):
        value_root = group_root / group_name
        _write_csv(value_root / "comparison_summary.csv", group_rows)
        _write_json(value_root / "comparison_summary.json", group_rows)
        _write_markdown_report(value_root / "comparison_report.md", group_rows)

    print(f"Comparison CSV: {overall_root / 'comparison_summary.csv'}")
    print(f"Comparison JSON: {overall_root / 'comparison_summary.json'}")
    print(f"Comparison report: {overall_root / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
