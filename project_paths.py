from __future__ import annotations

import re
import shutil
from pathlib import Path

from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE, PAYLOAD_COMPLETION_MODE

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"


def _sanitize(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def build_experiment_root(link_name: str, method_name: str, experiment_name: str) -> Path:
    return RESULTS_ROOT / link_name / method_name / _sanitize(experiment_name)


def _scenario_folder_name(scenario_mode: str) -> str:
    mode = str(scenario_mode).strip().lower()
    if mode == PAYLOAD_COMPLETION_MODE:
        return "Payload completion"
    if mode == FIXED_BLOCK_TARGETS_MODE:
        return "Fixed block targets"
    return _sanitize(mode or PAYLOAD_COMPLETION_MODE)


def build_scenario_experiment_root(
    link_name: str,
    scenario_mode: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return RESULTS_ROOT / link_name / _scenario_folder_name(scenario_mode) / method_name / _sanitize(experiment_name)


def mirror_experiment_root_to_scenario_layout(
    *,
    link_name: str,
    scenario_mode: str,
    method_name: str,
    source_experiment_root: str | Path,
) -> str:
    source_root = Path(source_experiment_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Cannot mirror missing experiment root: {source_root}")

    target_root = build_scenario_experiment_root(
        link_name,
        scenario_mode,
        method_name,
        source_root.name,
    )
    if source_root.resolve() == target_root.resolve():
        return str(target_root)

    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    return str(target_root)


def build_uplink_result_dirs(method_name: str, experiment_name: str) -> dict[str, str]:
    root = build_experiment_root("Uplink", method_name, experiment_name)
    training_root = root / "training"
    testing_root = root / "testing"
    dirs = {
        "experiment_root": root,
        "training_root": training_root,
        "testing_root": testing_root,
        "train_data": training_root / "data",
        "test_data": testing_root / "data",
        "train_optimization_history": training_root / "optimization_history",
        "test_optimization_history": testing_root / "optimization_history",
        "train_result": training_root / "optimization_history",
        "test_result": testing_root / "optimization_history",
        "test_user_config": testing_root / "user_config",
        "test_latency_asynchronality": testing_root / "latency_asynchronality",
        "test_link_quality": testing_root / "link_quality",
        "test_interference": testing_root / "interference",
        "test_schedule_details": testing_root / "schedule_details",
        "data": testing_root / "data",
        "optimization_history": testing_root / "optimization_history",
        "user_config": testing_root / "user_config",
        "latency_asynchronality": testing_root / "latency_asynchronality",
        "link_quality": testing_root / "link_quality",
        "interference": testing_root / "interference",
        "schedule_details": testing_root / "schedule_details",
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in dirs.items()}


def build_uplink_convergence_result_dirs(method_name: str, experiment_name: str) -> dict[str, str]:
    root = build_experiment_root("Uplink", method_name, experiment_name)
    data_root = root / "data"
    optimization_root = root / "optimization_history"
    user_config_root = root / "user_config"
    latency_root = root / "latency_asynchronality"
    link_root = root / "link_quality"
    interference_root = root / "interference"
    schedule_root = root / "schedule_details"
    dirs = {
        "experiment_root": root,
        "training_root": root,
        "testing_root": root,
        "train_data": data_root,
        "test_data": data_root,
        "data": data_root,
        "train_optimization_history": optimization_root,
        "test_optimization_history": optimization_root,
        "optimization_history": optimization_root,
        "train_result": optimization_root,
        "test_result": optimization_root,
        "test_user_config": user_config_root,
        "user_config": user_config_root,
        "test_latency_asynchronality": latency_root,
        "latency_asynchronality": latency_root,
        "test_link_quality": link_root,
        "link_quality": link_root,
        "test_interference": interference_root,
        "interference": interference_root,
        "test_schedule_details": schedule_root,
        "schedule_details": schedule_root,
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in dirs.items()}


def build_downlink_result_dirs(method_name: str, experiment_name: str) -> dict[str, str]:
    root = build_experiment_root("Downlink", method_name, experiment_name)
    training_root = root / "training"
    testing_root = root / "testing"
    dirs = {
        "experiment_root": root,
        "training_root": training_root,
        "testing_root": testing_root,
        "train_data": training_root / "data",
        "train_optimization_history": training_root / "optimization_history",
        "test_data": testing_root / "data",
        "test_user_config": testing_root / "user_config",
        "test_latency_asynchronality": testing_root / "latency_asynchronality",
        "test_link_quality": testing_root / "link_quality",
        "test_optimization_history": testing_root / "optimization_history",
        "test_schedule_details": testing_root / "schedule_details",
        "test_interference": testing_root / "interference",
        "data": testing_root / "data",
        "user_config": testing_root / "user_config",
        "latency_asynchronality": testing_root / "latency_asynchronality",
        "link_quality": testing_root / "link_quality",
        "optimization_history": testing_root / "optimization_history",
        "schedule_details": testing_root / "schedule_details",
        "interference": testing_root / "interference",
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in dirs.items()}
