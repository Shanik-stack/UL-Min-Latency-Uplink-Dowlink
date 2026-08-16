from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
COMPARISON_ROOT = RESULTS_ROOT / "Small Exhaustive Comparisons"

RUNNER_SPECS = {
    "uplink": {
        "module_path": PROJECT_ROOT / "Uplink" / "Methods" / "Convergence per sweep" / "exhaustive_payload_compare.py",
        "default_cfg": "uplink_payload_completion_exhaustive_small.yaml",
    },
    "downlink": {
        "module_path": PROJECT_ROOT / "Downlink" / "Methods" / "Convergence per sweep" / "exhaustive_payload_compare.py",
        "default_cfg": "downlink_payload_completion_exhaustive_small.yaml",
    },
}

MODULE_PURGE_NAMES = {
    "Optimizer_per_block",
    "UplinkSystem",
    "advanced_methods_common",
    "config_loader",
    "downlink_system",
    "experiment_scenarios",
    "experiment_utils",
    "optimizer",
    "precoder_models",
    "project_paths",
    "uplink_rate_model",
}


def _current_local_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d__%H%M%S")


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return sorted({int(part.strip()) for part in str(args.seeds).split(",") if part.strip()})
    if args.seed_start is not None and args.seed_end is not None:
        start = int(args.seed_start)
        end = int(args.seed_end)
        if end < start:
            raise ValueError("--seed_end must be >= --seed_start")
        return list(range(start, end + 1))
    raise ValueError("Provide either --seeds or both --seed_start and --seed_end")


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _load_runner(link: str):
    for module_name in MODULE_PURGE_NAMES:
        sys.modules.pop(module_name, None)
    spec_info = RUNNER_SPECS[str(link)]
    module_path = spec_info["module_path"]
    module_name = f"small_exhaustive_{link}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module for {link}: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = getattr(module, "run_exhaustive_payload_compare", None)
    if runner is None:
        raise RuntimeError(f"Module {module_path} does not expose run_exhaustive_payload_compare")
    return runner


def _strategy_key(strategy: dict[str, Any]) -> tuple:
    completed = bool(strategy.get("all_completed", False))
    latency = int(strategy.get("global_latency_sum", 10**12))
    remaining_total = int(sum(max(int(v), 0) for v in strategy.get("per_user_remaining_bits", []) or []))
    return (not completed, latency, remaining_total)


def _evaluate_case(link: str, cfg_name: str, seed: int, include_catalogs: bool) -> dict[str, Any]:
    runner = _load_runner(str(link))
    result = runner(cfg_name=str(cfg_name), seed=int(seed), include_catalogs=bool(include_catalogs))
    online = result.get("online_strategy", {}) or {}
    exhaustive = result.get("exhaustive_strategy", {}) or {}
    online_key = _strategy_key(online)
    exhaustive_key = _strategy_key(exhaustive)
    latency_gap = int(online.get("global_latency_sum", 0)) - int(exhaustive.get("global_latency_sum", 0))
    remaining_gap = int(sum(max(int(v), 0) for v in online.get("per_user_remaining_bits", []) or [])) - int(
        sum(max(int(v), 0) for v in exhaustive.get("per_user_remaining_bits", []) or [])
    )
    counterexample = exhaustive_key < online_key
    return {
        "link": str(link),
        "cfg_name": str(cfg_name),
        "seed": int(seed),
        "counterexample": bool(counterexample),
        "online_key": list(online_key),
        "exhaustive_key": list(exhaustive_key),
        "latency_gap": int(latency_gap),
        "remaining_bits_gap": int(remaining_gap),
        "result": result,
    }


def _build_row(case: dict[str, Any]) -> dict[str, Any]:
    result = case["result"]
    online = result.get("online_strategy", {}) or {}
    exhaustive = result.get("exhaustive_strategy", {}) or {}
    online_remaining_total = int(sum(max(int(v), 0) for v in online.get("per_user_remaining_bits", []) or []))
    exhaustive_remaining_total = int(sum(max(int(v), 0) for v in exhaustive.get("per_user_remaining_bits", []) or []))
    online_latency = int(online.get("global_latency_sum", 0))
    exhaustive_latency = int(exhaustive.get("global_latency_sum", 0))
    latency_gap_pct = 100.0 * float(online_latency - exhaustive_latency) / max(float(online_latency), 1.0)
    return {
        "link": case["link"],
        "seed": case["seed"],
        "cfg_name": case["cfg_name"],
        "counterexample": bool(case["counterexample"]),
        "online_completed": bool(online.get("all_completed", False)),
        "exhaustive_completed": bool(exhaustive.get("all_completed", False)),
        "online_latency_sum": online_latency,
        "exhaustive_latency_sum": exhaustive_latency,
        "latency_gap": int(case["latency_gap"]),
        "latency_gap_percent": latency_gap_pct,
        "online_remaining_bits_total": online_remaining_total,
        "exhaustive_remaining_bits_total": exhaustive_remaining_total,
        "remaining_bits_gap": int(case["remaining_bits_gap"]),
        "online_served_bits": _json_text(online.get("per_user_served_bits")),
        "exhaustive_served_bits": _json_text(exhaustive.get("per_user_served_bits")),
        "online_remaining_bits": _json_text(online.get("per_user_remaining_bits")),
        "exhaustive_remaining_bits": _json_text(exhaustive.get("per_user_remaining_bits")),
        "online_per_user_latency": _json_text(online.get("per_user_latency")),
        "exhaustive_per_user_latency": _json_text(exhaustive.get("per_user_latency")),
        "online_wall_time_seconds": float(online.get("wall_time_seconds", 0.0)),
        "exhaustive_wall_time_seconds": float(exhaustive.get("wall_time_seconds", 0.0)),
        "case_wall_time_seconds_total": float(result.get("core_wall_time_seconds_total", 0.0)),
        "run_started_at_local": result.get("run_started_at_local", ""),
        "run_completed_at_local": result.get("run_completed_at_local", ""),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_lines(
    *,
    link_mode: str,
    seeds: list[int],
    max_workers: int,
    include_catalogs: bool,
    rows: list[dict[str, Any]],
    started_at: str,
    completed_at: str,
) -> list[str]:
    counterexamples = [row for row in rows if bool(row.get("counterexample", False))]
    lines = [
        "Small exhaustive payload-completion seed sweep",
        f"Run started (local): {started_at}",
        f"Run completed (local): {completed_at}",
        f"Link selection: {link_mode}",
        f"Seeds: {seeds}",
        f"Max workers: {max_workers}",
        f"Include exhaustive catalogs in per-seed outputs: {include_catalogs}",
        "",
        f"Total evaluated cases: {len(rows)}",
        f"Counterexamples found: {len(counterexamples)}",
        "",
        "Counterexample definition",
        "  - Exhaustive is considered better if it wins on the same lexicographic rule used in the exhaustive ranking:",
        "    completed first, then lower total latency sum, then lower total remaining bits.",
        "",
    ]
    if counterexamples:
        best_rows = sorted(
            counterexamples,
            key=lambda row: (
                -int(row.get("latency_gap", 0)),
                -int(row.get("remaining_bits_gap", 0)),
                str(row.get("link", "")),
                int(row.get("seed", 0)),
            ),
        )
        lines.append("Top counterexamples")
        for row in best_rows[:10]:
            lines.append(
                "  - "
                f"{row['link']} seed={row['seed']} | latency {row['online_latency_sum']} -> {row['exhaustive_latency_sum']} "
                f"(gap {row['latency_gap']}) | remaining bits {row['online_remaining_bits_total']} -> "
                f"{row['exhaustive_remaining_bits_total']}"
            )
    else:
        lines.append("Top counterexamples")
        lines.append("  - None found in this sweep.")
    return lines


def _run_cases_with_executor(executor_factory, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    with executor_factory() as executor:
        future_to_case = {
            executor.submit(
                _evaluate_case,
                str(case["link"]),
                str(case["cfg_name"]),
                int(case["seed"]),
                bool(case["include_catalogs"]),
            ): case
            for case in cases
        }
        completed = 0
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            completed += 1
            result = future.result()
            print(
                f"[{completed}/{len(cases)}] Finished {case['link']} seed={case['seed']} | "
                f"counterexample={result['counterexample']} | latency_gap={result['latency_gap']}",
                flush=True,
            )
            evaluated.append(result)
    return evaluated


def _run_cases_sequential(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] Running {case['link']} seed={case['seed']} cfg={case['cfg_name']}",
            flush=True,
        )
        evaluated.append(
            _evaluate_case(
                link=str(case["link"]),
                cfg_name=str(case["cfg_name"]),
                seed=int(case["seed"]),
                include_catalogs=bool(case["include_catalogs"]),
            )
        )
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep seeds for the small exhaustive payload validators.")
    parser.add_argument("--link", type=str, default="both", choices=["uplink", "downlink", "both"])
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, for example 0,1,2,3")
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_end", type=int, default=None)
    parser.add_argument("--uplink_cfg", type=str, default=RUNNER_SPECS["uplink"]["default_cfg"])
    parser.add_argument("--downlink_cfg", type=str, default=RUNNER_SPECS["downlink"]["default_cfg"])
    parser.add_argument("--include_catalogs", action="store_true", help="Save full exhaustive catalogs in each per-seed result")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel worker count across seed/link cases")
    args = parser.parse_args()

    seeds = _parse_seeds(args)
    links = ["uplink", "downlink"] if str(args.link) == "both" else [str(args.link)]
    cfg_by_link = {
        "uplink": str(args.uplink_cfg),
        "downlink": str(args.downlink_cfg),
    }
    started_at = _current_local_timestamp()
    run_tag = f"seed_sweep__{args.link}__{_timestamp_tag()}"
    run_root = COMPARISON_ROOT / run_tag
    per_seed_root = run_root / "per_seed_results"
    per_seed_root.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "link": str(link),
            "cfg_name": str(cfg_by_link[link]),
            "seed": int(seed),
            "include_catalogs": bool(args.include_catalogs),
        }
        for link in links
        for seed in seeds
    ]
    print(f"Saving sweep results under: {run_root}", flush=True)
    print(f"Evaluating {len(cases)} cases: {[(case['link'], case['seed']) for case in cases]}", flush=True)

    evaluated: list[dict[str, Any]] = []
    max_workers = max(1, int(args.max_workers))
    if max_workers == 1:
        evaluated = _run_cases_sequential(cases)
    else:
        try:
            evaluated = _run_cases_with_executor(lambda: ProcessPoolExecutor(max_workers=max_workers), cases)
        except (PermissionError, OSError) as exc:
            print(
                f"Process-based parallelism unavailable ({exc}). Falling back to thread-based execution.",
                flush=True,
            )
            unique_links = {str(case["link"]) for case in cases}
            if len(unique_links) > 1:
                print(
                    "Mixed uplink/downlink cases share conflicting module names in one interpreter. "
                    "Falling back to sequential execution for safety.",
                    flush=True,
                )
                evaluated = _run_cases_sequential(cases)
            else:
                try:
                    evaluated = _run_cases_with_executor(lambda: ThreadPoolExecutor(max_workers=max_workers), cases)
                except Exception as thread_exc:
                    print(
                        f"Thread-based execution also failed ({thread_exc}). Falling back to sequential execution.",
                        flush=True,
                    )
                    evaluated = _run_cases_sequential(cases)

    rows = [_build_row(case) for case in evaluated]
    rows.sort(key=lambda row: (str(row["link"]), int(row["seed"])))

    for case in evaluated:
        result = case["result"]
        link_dir = per_seed_root / str(case["link"])
        link_dir.mkdir(parents=True, exist_ok=True)
        seed_path = link_dir / f"seed_{int(case['seed']):04d}.json"
        seed_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    comparison_summary = {
        "run_started_at_local": started_at,
        "run_completed_at_local": _current_local_timestamp(),
        "link_mode": str(args.link),
        "seeds": seeds,
        "max_workers": max_workers,
        "include_catalogs": bool(args.include_catalogs),
        "rows": rows,
    }

    counterexample_rows = [row for row in rows if bool(row.get("counterexample", False))]
    counterexample_rows.sort(
        key=lambda row: (
            -int(row.get("latency_gap", 0)),
            -int(row.get("remaining_bits_gap", 0)),
            str(row.get("link", "")),
            int(row.get("seed", 0)),
        )
    )

    summary_lines = _summary_lines(
        link_mode=str(args.link),
        seeds=seeds,
        max_workers=max_workers,
        include_catalogs=bool(args.include_catalogs),
        rows=rows,
        started_at=started_at,
        completed_at=comparison_summary["run_completed_at_local"],
    )

    (run_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (run_root / "summary.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    _write_csv(rows, run_root / "summary.csv")
    _write_csv(counterexample_rows, run_root / "counterexamples.csv")

    print("\n".join(summary_lines))
    print(f"\nSaved sweep comparison to: {run_root}")


if __name__ == "__main__":
    main()
