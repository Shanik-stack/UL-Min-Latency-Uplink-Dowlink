from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import run_decision_sweep as sweep


def _load_rows(comparison_root: Path) -> list[dict]:
    rows_path = comparison_root / "overall" / "comparison_summary.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    for row in rows:
        try:
            row["_effective_decisions"] = json.loads(row.get("effective_decisions_json") or "{}")
        except Exception:
            row["_effective_decisions"] = {}
    return rows


def _write_request_summary(
    *,
    comparison_root: Path,
    manifest_path: Path,
    rows: list[dict],
) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in completed:
        groups[(str(row.get("link")), str(row.get("method")))].append(row)

    lines = [
        "# Payload Direction and Weighting Sweep Summary",
        "",
        f"- Sweep: `{comparison_root.name}`",
        f"- Manifest: `{manifest_path}`",
        f"- Completed runs: `{len(completed)}/{len(rows)}`",
        "- Objective names below use the cleaned public labels such as `unweighted_sum_rate` and `inverse_cnr_weighted_sum_rate`.",
        "- Asynchronality is the sum of pairwise latency differences across users, so smaller is better.",
    ]

    for group_key in sorted(groups):
        link_name, method_name = group_key
        group_rows = sorted(
            groups[group_key],
            key=lambda row: (
                float(row.get("final_total_latency") or 1e18),
                -float(row.get("total_latency_reduction_percent") or -1e18),
                str(row.get("decision_tag", "")),
            ),
        )
        best_latency = min(group_rows, key=lambda row: float(row.get("final_total_latency") or 1e18))
        best_async = min(group_rows, key=lambda row: float(row.get("final_asynchronality_sum") or 1e18))
        lines.extend(
            [
                "",
                f"## {link_name.capitalize()} | {method_name}",
                "",
                "| Variant | Configuration | Final latency | Latency reduction % | Final asynchronality | Asynchronality reduction % | Final avg SNR (dB) | Final avg SINR (dB) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        sweep._normalize_report_text(row.get("decision_tag", "")),
                        sweep._normalize_report_text(row.get("highlighted_configuration", "")),
                        sweep._metric_text(row.get("final_total_latency"), digits=6),
                        sweep._metric_text(row.get("total_latency_reduction_percent"), digits=4),
                        sweep._metric_text(row.get("final_asynchronality_sum"), digits=6),
                        sweep._metric_text(row.get("asynchronality_reduction_percent"), digits=4),
                        sweep._metric_text(row.get("final_avg_snr_db"), digits=4),
                        sweep._metric_text(row.get("final_avg_sinr_db"), digits=4),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Best latency: `{sweep._normalize_report_text(best_latency.get('decision_tag', ''))}` -> `{sweep._metric_text(best_latency.get('final_total_latency'), digits=6)}`",
                f"- Best latency configuration: `{sweep._normalize_report_text(best_latency.get('highlighted_configuration', ''))}`",
                f"- Best asynchronality: `{sweep._normalize_report_text(best_async.get('decision_tag', ''))}` -> `{sweep._metric_text(best_async.get('final_asynchronality_sum'), digits=6)}`",
                f"- Best asynchronality configuration: `{sweep._normalize_report_text(best_async.get('highlighted_configuration', ''))}`",
            ]
        )
        if best_latency.get("decision_tag") != best_async.get("decision_tag"):
            lines.append(
                "- Interpretation: best latency and best asynchronality differ here, so this group shows a throughput-versus-fairness tradeoff."
            )

    summary_path = comparison_root / "overall" / "request_compiled_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild saved decision-sweep markdown reports from comparison_summary.json.")
    parser.add_argument("--comparison_root", required=True, help="Path to Results/Decision Sweep Comparisons/<sweep_id>")
    parser.add_argument("--manifest", required=True, help="Manifest YAML path used for the sweep")
    parser.add_argument("--run_root", required=True, help="Path to Decision Sweep Runs/<sweep_id>")
    args = parser.parse_args()

    comparison_root = Path(args.comparison_root)
    manifest_path = Path(args.manifest)
    run_root = Path(args.run_root)
    rows = _load_rows(comparison_root)

    sweep._write_detailed_comparison_results(
        comparison_root=comparison_root,
        manifest_path=manifest_path,
        run_root=run_root,
        rows=rows,
        generated_at_local=sweep.current_local_timestamp(),
    )
    _write_request_summary(comparison_root=comparison_root, manifest_path=manifest_path, rows=rows)
    print(f"Rebuilt comparison reports under: {comparison_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
