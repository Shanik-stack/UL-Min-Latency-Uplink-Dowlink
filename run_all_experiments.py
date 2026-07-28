from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "Experiment Configs"

LINK_ALIASES = {
    "ul": "uplink",
    "uplink": "uplink",
    "dl": "downlink",
    "downlink": "downlink",
}

METHOD_ALIASES = {
    "convergence": "convergence",
    "conv": "convergence",
    "convergence_per_epoch": "convergence",
    "monte_carlo": "monte_carlo",
    "monte-carlo": "monte_carlo",
    "mc": "monte_carlo",
}

METHOD_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    (
        "uplink",
        "convergence",
    ): {
        "display_name": "Uplink | Convergence per epoch",
        "script_path": PROJECT_ROOT / "Uplink" / "Methods" / "Convergence per epoch" / "main.py",
        "supports_quiet": False,
    },
    (
        "uplink",
        "monte_carlo",
    ): {
        "display_name": "Uplink | Monte Carlo",
        "script_path": PROJECT_ROOT / "Uplink" / "Methods" / "Monte Carlo" / "main.py",
        "supports_quiet": False,
    },
    (
        "downlink",
        "convergence",
    ): {
        "display_name": "Downlink | Convergence per epoch",
        "script_path": PROJECT_ROOT / "Downlink" / "Methods" / "Convergence per epoch" / "main.py",
        "supports_quiet": True,
    },
    (
        "downlink",
        "monte_carlo",
    ): {
        "display_name": "Downlink | Monte Carlo",
        "script_path": PROJECT_ROOT / "Downlink" / "Methods" / "Monte Carlo" / "main.py",
        "supports_quiet": True,
    },
}


def _resolve_manifest_path(manifest_name: str) -> Path:
    raw = str(manifest_name).strip()
    candidate = Path(raw)
    if candidate.suffix.lower() != ".yaml":
        candidate = Path(f"{raw}.yaml")

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / candidate,
                EXPERIMENT_CONFIGS_DIR / candidate,
                Path.cwd() / candidate,
            ]
        )

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find batch manifest '{manifest_name}'. Searched:\n{searched}")


def _normalize_link(link_name: str) -> str:
    key = str(link_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LINK_ALIASES:
        raise ValueError(f"Unknown link '{link_name}'. Expected uplink or downlink.")
    return str(LINK_ALIASES[key])


def _normalize_method(method_name: str) -> str:
    key = str(method_name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unknown method '{method_name}'. Expected convergence or monte_carlo.")
    return str(METHOD_ALIASES[key])


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_int_list(value: Any) -> list[int]:
    return [int(item) for item in _as_string_list(value)]


def _as_extra_arg_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _discover_config_names_for_link(link_name: str) -> list[str]:
    prefix = "uplink_" if link_name == "uplink" else "downlink_"
    config_names = sorted(
        path.name
        for path in EXPERIMENT_CONFIGS_DIR.glob(f"{prefix}*.yaml")
        if path.is_file()
        and "fixed_block_targets" not in path.name.lower()
    )
    if len(config_names) == 0:
        raise FileNotFoundError(f"No config files found for link '{link_name}' under {EXPERIMENT_CONFIGS_DIR}.")
    return config_names


def _expand_cfg_names(raw_cfg_names: Any, *, link_name: str) -> list[str]:
    cfg_names = _as_string_list(raw_cfg_names)
    if len(cfg_names) == 0:
        return _discover_config_names_for_link(link_name)
    if any(name.lower() in {"all", "*"} for name in cfg_names):
        return _discover_config_names_for_link(link_name)
    return cfg_names


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not interpret boolean value: {value!r}")


def _normalize_launch_mode(value: Any) -> str:
    if value is None:
        return "parallel_terminals"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"parallel", "parallel_terminal", "parallel_terminals", "new_terminal", "new_terminals"}:
        return "parallel_terminals"
    if text in {"sequential", "same_terminal"}:
        return "sequential"
    raise ValueError(
        f"Unknown launch_mode {value!r}. Expected 'parallel_terminals' or 'sequential'."
    )


def _build_command(
    *,
    python_executable: str,
    link_name: str,
    method_name: str,
    cfg_name: str,
    seed: int | None,
    train_seeds: list[int],
    num_train_seeds: int | None,
    test_seed: int | None,
    quiet: bool,
    skip_test: bool,
    extra_args: list[str],
) -> list[str]:
    spec = METHOD_SPECS[(link_name, method_name)]
    cmd = [python_executable, str(spec["script_path"]), "--cfg_name", str(cfg_name)]
    if method_name == "convergence":
        if seed is None:
            raise ValueError(f"{link_name} convergence requires 'seed'.")
        cmd.extend(["--seed", str(int(seed))])
    else:
        if len(train_seeds) > 0:
            cmd.extend(["--train_seeds", ",".join(str(int(v)) for v in train_seeds)])
        elif num_train_seeds is not None:
            cmd.extend(["--num_train_seeds", str(int(num_train_seeds))])
        if test_seed is not None:
            cmd.extend(["--test_seed", str(int(test_seed))])
        if skip_test:
            cmd.append("--skip_test")
    if quiet and bool(spec.get("supports_quiet", False)):
        cmd.append("--quiet")
    cmd.extend(extra_args)
    return cmd


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Batch manifest must be a mapping with 'defaults' and 'runs'.")
    defaults = payload.get("defaults", {})
    runs = payload.get("runs", [])
    if not isinstance(defaults, dict):
        raise ValueError("Manifest 'defaults' must be a mapping.")
    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("Manifest must contain a non-empty 'runs' list.")
    normalized_runs: list[dict[str, Any]] = []
    for idx, raw_run in enumerate(runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Manifest run at index {idx} must be a mapping.")
        normalized_runs.append(dict(raw_run))
    return defaults, normalized_runs


def _build_invocations(
    *,
    defaults: dict[str, Any],
    runs: list[dict[str, Any]],
    python_override: str | None,
) -> list[dict[str, Any]]:
    default_python = str(defaults.get("python_executable", sys.executable))
    invocations: list[dict[str, Any]] = []
    for raw_run in runs:
        link_name = _normalize_link(raw_run.get("link"))
        method_name = _normalize_method(raw_run.get("method"))
        if not _normalize_bool(raw_run.get("enabled", True), default=True):
            continue

        cfg_names = _expand_cfg_names(raw_run.get("cfg_names"), link_name=link_name)
        seed = raw_run.get("seed", defaults.get("seed"))
        test_seed = raw_run.get("test_seed", defaults.get("test_seed"))
        train_seeds = _as_int_list(raw_run.get("train_seeds", defaults.get("train_seeds", [])))
        raw_num_train_seeds = raw_run.get("num_train_seeds", defaults.get("num_train_seeds"))
        num_train_seeds = int(raw_num_train_seeds) if raw_num_train_seeds is not None else None
        quiet = _normalize_bool(raw_run.get("quiet", defaults.get("quiet", False)), default=False)
        skip_test = _normalize_bool(raw_run.get("skip_test", defaults.get("skip_test", False)), default=False)
        extra_args = _as_extra_arg_list(raw_run.get("extra_args", []))
        python_executable = str(python_override or raw_run.get("python_executable", default_python))

        for cfg_name in cfg_names:
            command = _build_command(
                python_executable=python_executable,
                link_name=link_name,
                method_name=method_name,
                cfg_name=cfg_name,
                seed=int(seed) if seed is not None else None,
                train_seeds=train_seeds,
                num_train_seeds=num_train_seeds,
                test_seed=int(test_seed) if test_seed is not None else None,
                quiet=quiet,
                skip_test=skip_test,
                extra_args=extra_args,
            )
            invocations.append(
                {
                    "link": link_name,
                    "method": method_name,
                    "cfg_name": str(cfg_name),
                    "display_name": str(METHOD_SPECS[(link_name, method_name)]["display_name"]),
                    "command": command,
                }
            )
    return invocations


def _normalize_filter_set(raw_value: str | None, *, kind: str) -> set[str]:
    values = _as_string_list(raw_value)
    if len(values) == 0:
        return set()
    if kind == "link":
        return {_normalize_link(value) for value in values}
    if kind == "method":
        return {_normalize_method(value) for value in values}
    return {str(value).strip() for value in values}


def _filter_invocations(
    invocations: list[dict[str, Any]],
    *,
    link_filter: set[str],
    method_filter: set[str],
    cfg_filter: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for invocation in invocations:
        cfg_name = str(invocation["cfg_name"])
        cfg_stem = Path(cfg_name).stem
        if len(link_filter) > 0 and str(invocation["link"]) not in link_filter:
            continue
        if len(method_filter) > 0 and str(invocation["method"]) not in method_filter:
            continue
        if len(cfg_filter) > 0 and cfg_name not in cfg_filter and cfg_stem not in cfg_filter:
            continue
        filtered.append(invocation)
    return filtered


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _print_plan(invocations: list[dict[str, Any]], *, launch_mode: str) -> None:
    print("")
    print("Batch experiment plan")
    print(f"Launch mode: {launch_mode}")
    print(f"Total runs: {len(invocations)}")
    for idx, invocation in enumerate(invocations, start=1):
        print(
            f"{idx:02d}. {invocation['display_name']} | cfg={invocation['cfg_name']} | "
            f"cmd={_format_command(invocation['command'])}"
        )


def _quote_powershell_literal(text: str) -> str:
    return str(text).replace("'", "''")


def _format_powershell_invocation(command: list[str]) -> str:
    return "& " + " ".join(f"'{_quote_powershell_literal(part)}'" for part in command)


def _build_terminal_script(invocation: dict[str, Any], *, index: int, total: int) -> str:
    label = f"[{index}/{total}] {invocation['display_name']} | cfg={invocation['cfg_name']}"
    label_q = _quote_powershell_literal(label)
    cwd_q = _quote_powershell_literal(str(PROJECT_ROOT))
    run_expr = _format_powershell_invocation(invocation["command"])
    return (
        f"$runLabel = '{label_q}'; "
        f"$Host.UI.RawUI.WindowTitle = $runLabel; "
        f"Set-Location '{cwd_q}'; "
        f"Write-Host $runLabel -ForegroundColor Cyan; "
        f"{run_expr}; "
        f"$exitCode = $LASTEXITCODE; "
        f"Write-Host ''; "
        f"if ($exitCode -eq 0) {{ "
        f"Write-Host 'Experiment finished successfully.' -ForegroundColor Green "
        f"}} else {{ "
        f"Write-Host ('Experiment failed with exit code ' + $exitCode) -ForegroundColor Red "
        f"}}; "
        f"Read-Host 'Press Enter to close'; "
        f"exit $exitCode"
    )


def _launch_in_new_terminal(invocation: dict[str, Any], *, index: int, total: int) -> subprocess.Popen[Any]:
    if os.name != "nt":
        raise RuntimeError("parallel_terminals launch mode currently supports Windows only.")
    terminal_script = _build_terminal_script(invocation, index=index, total=total)
    powershell_cmd = [
        "powershell.exe",
        "-NoLogo",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        terminal_script,
    ]
    return subprocess.Popen(
        powershell_cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all uplink/downlink experiment configs and methods from one manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="all_experiments.yaml",
        help="Batch manifest YAML name or path. Defaults to Experiment Configs/all_experiments.yaml",
    )
    parser.add_argument("--python_exe", type=str, default=None, help="Override Python executable used for child runs.")
    parser.add_argument("--links", type=str, default=None, help="Optional filter, e.g. uplink or uplink,downlink")
    parser.add_argument("--methods", type=str, default=None, help="Optional filter, e.g. convergence or convergence,monte_carlo")
    parser.add_argument("--configs", type=str, default=None, help="Optional filter by cfg filename or stem, comma-separated")
    parser.add_argument(
        "--launch_mode",
        type=str,
        default=None,
        help="Launch mode: parallel_terminals or sequential. Defaults to manifest/default parallel_terminals.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue_on_error", action="store_true", help="Keep running remaining experiments after a failure.")
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.manifest)
    defaults, runs = _load_manifest(manifest_path)
    launch_mode = _normalize_launch_mode(args.launch_mode or defaults.get("launch_mode", "parallel_terminals"))
    invocations = _build_invocations(
        defaults=defaults,
        runs=runs,
        python_override=args.python_exe,
    )
    invocations = _filter_invocations(
        invocations,
        link_filter=_normalize_filter_set(args.links, kind="link"),
        method_filter=_normalize_filter_set(args.methods, kind="method"),
        cfg_filter=_normalize_filter_set(args.configs, kind="cfg"),
    )
    if len(invocations) == 0:
        print("No experiment runs matched the manifest and filters.")
        return 1

    print(f"Loaded batch manifest: {manifest_path}")
    _print_plan(invocations, launch_mode=launch_mode)
    if args.dry_run:
        return 0

    if launch_mode == "parallel_terminals":
        launched: list[dict[str, Any]] = []
        for idx, invocation in enumerate(invocations, start=1):
            print("")
            print("=" * 100)
            print(f"Launching [{idx}/{len(invocations)}] {invocation['display_name']} | cfg={invocation['cfg_name']}")
            print(_format_command(invocation["command"]))
            print("=" * 100)
            process = _launch_in_new_terminal(invocation, index=idx, total=len(invocations))
            launched.append(
                {
                    "index": idx,
                    "display_name": invocation["display_name"],
                    "cfg_name": invocation["cfg_name"],
                    "pid": int(process.pid),
                }
            )

        print("")
        print("Batch experiment launch summary")
        print(f"Requested runs: {len(invocations)}")
        print(f"Launched terminal windows: {len(launched)}")
        print("Each experiment is now running in its own PowerShell window.")
        print("Those windows stay open after completion so you can inspect the logs.")
        for item in launched:
            print(
                f"- run {item['index']}: {item['display_name']} | "
                f"cfg={item['cfg_name']} | terminal_pid={item['pid']}"
            )
        return 0

    failures: list[dict[str, Any]] = []
    for idx, invocation in enumerate(invocations, start=1):
        print("")
        print("=" * 100)
        print(f"[{idx}/{len(invocations)}] {invocation['display_name']} | cfg={invocation['cfg_name']}")
        print(_format_command(invocation["command"]))
        print("=" * 100)
        completed = subprocess.run(
            invocation["command"],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if completed.returncode != 0:
            failures.append(
                {
                    "index": idx,
                    "display_name": invocation["display_name"],
                    "cfg_name": invocation["cfg_name"],
                    "returncode": int(completed.returncode),
                }
            )
            print(
                f"FAILED [{idx}/{len(invocations)}] {invocation['display_name']} | "
                f"cfg={invocation['cfg_name']} | returncode={completed.returncode}"
            )
            if not args.continue_on_error:
                break

    print("")
    print("Batch experiment summary")
    print(f"Requested runs: {len(invocations)}")
    print(f"Failed runs: {len(failures)}")
    if len(failures) == 0:
        print("All runs completed successfully.")
        return 0

    for failure in failures:
        print(
            f"- run {failure['index']}: {failure['display_name']} | "
            f"cfg={failure['cfg_name']} | returncode={failure['returncode']}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
