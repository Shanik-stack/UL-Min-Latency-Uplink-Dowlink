from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FORWARD_BACKWARD_FLOP_FACTOR = 3.0


def _layer_flops(in_dim: int, out_dim: int) -> int:
    return int((2 * int(in_dim) * int(out_dim)) + int(out_dim))


def _relu_flops(width: int) -> int:
    return int(width)


def _hidden_dims(out_dim: int) -> tuple[int, int, int]:
    out_dim = int(out_dim)
    return (
        max(256, 8 * out_dim),
        max(128, 4 * out_dim),
        max(64, 2 * out_dim),
    )


def _mlp_forward_flops(in_dim: int, out_dim: int) -> int:
    h1, h2, h3 = _hidden_dims(int(out_dim))
    total = 0
    total += _layer_flops(in_dim, h1) + _relu_flops(h1)
    total += _layer_flops(h1, h2) + _relu_flops(h2)
    total += _layer_flops(h2, h3) + _relu_flops(h3)
    total += _layer_flops(h3, int(out_dim))
    return int(total)


def _sum_numeric_mapping(values: Mapping[str, Any] | None) -> int:
    if not isinstance(values, Mapping):
        return 0
    total = 0
    for value in values.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return int(total)


def _sum_numeric_sequence_mappings(values: Sequence[Mapping[str, Any]] | None) -> list[int]:
    if not isinstance(values, Sequence):
        return []
    return [_sum_numeric_mapping(value) for value in values if isinstance(value, Mapping)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summed_nested_state_count(all_user_block_results: Sequence[Sequence[Sequence[dict[str, Any]]]]) -> list[int]:
    per_user_counts: list[int] = []
    for user_blocks in all_user_block_results:
        count = 0
        for block_states in user_blocks:
            count += len(block_states)
        per_user_counts.append(int(count))
    return per_user_counts


def _summed_uplink_solver_epochs(
    all_user_block_results: Sequence[Sequence[Sequence[dict[str, Any]]]],
) -> tuple[list[int], int]:
    per_user_epochs: list[int] = []
    total_solver_calls = 0
    for user_blocks in all_user_block_results:
        user_epochs = 0
        for block_states in user_blocks:
            for state in block_states:
                if isinstance(state, Mapping) and isinstance(state.get("kkt_history"), Sequence):
                    epochs = len(state.get("kkt_history", []))
                    if epochs > 0:
                        total_solver_calls += 1
                    user_epochs += int(epochs)
        per_user_epochs.append(int(user_epochs))
    return per_user_epochs, int(total_solver_calls)


def _uplink_model_forward_flops_from_spec(spec: Mapping[str, Any]) -> int:
    nr = int(spec.get("Nr", spec.get("nr", 0)))
    nt = int(spec.get("Nt", spec.get("nb", spec.get("NT", 0))))
    dk = int(spec.get("dk", 0))
    input_mode = str(spec.get("input_mode", "channel_only")).strip().lower()
    out_dim = 2 * nt * dk

    if input_mode == "channel_sigma_epsilon":
        in_dim = (2 * nr * nt) + 2
    elif input_mode == "channel_noise_epsilon":
        in_dim = (2 * nr * nt) + (2 * nr * nr) + 1
    elif input_mode == "channel_noise_epsilon_n":
        in_dim = (2 * nr * nt) + (2 * nr * nr) + 2
    elif input_mode == "channel_sigma_epsilon_n":
        in_dim = (2 * nr * nt) + 3
    else:
        in_dim = 2 * nr * nt
    return _mlp_forward_flops(in_dim, out_dim)


def _downlink_model_forward_flops_from_spec(spec: Mapping[str, Any]) -> int:
    nr = int(spec.get("nr", spec.get("Nr", 0)))
    nb = int(spec.get("nb", spec.get("Nt", spec.get("Nb", 0))))
    dk = int(spec.get("dk", 0))
    input_mode = str(spec.get("input_mode", "channel_only")).strip().lower()
    out_dim = 2 * nb * dk

    if input_mode == "block_context_noise_epsilon":
        k_count = int(spec.get("context_k", 1))
        max_nr = int(spec.get("context_max_nr", nr))
        max_nb = int(spec.get("context_max_nb", nb))
        in_dim = (2 * k_count * max_nr * max_nb) + k_count + (2 * max_nr * max_nr) + 1
    elif input_mode == "block_context_noise_epsilon_n":
        k_count = int(spec.get("context_k", 1))
        max_nr = int(spec.get("context_max_nr", nr))
        max_nb = int(spec.get("context_max_nb", nb))
        in_dim = (2 * k_count * max_nr * max_nb) + k_count + (2 * max_nr * max_nr) + 2
    else:
        in_dim = 2 * nr * nb
    return _mlp_forward_flops(in_dim, out_dim)


def build_uplink_sigma_context_specs(system_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    nr = list(system_params.get("NR", []))
    nt = list(system_params.get("NT", []))
    dk = list(system_params.get("dk", []))
    return [
        {
            "Nr": int(nr[k]),
            "Nt": int(nt[k]),
            "dk": int(dk[k]),
            "input_mode": "channel_sigma_epsilon",
        }
        for k in range(len(nr))
    ]


def build_downlink_channel_only_specs(system_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    nr = list(system_params.get("Nr", []))
    nb = list(system_params.get("Nb", []))
    dk = list(system_params.get("dk", []))
    return [
        {
            "nr": int(nr[k]),
            "nb": int(nb[k]),
            "dk": int(dk[k]),
            "input_mode": "channel_only",
        }
        for k in range(len(nr))
    ]


def build_forward_flops_per_user(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    link: str,
) -> list[int]:
    if str(link).strip().lower() == "uplink":
        return [_uplink_model_forward_flops_from_spec(spec) for spec in model_specs]
    return [_downlink_model_forward_flops_from_spec(spec) for spec in model_specs]


def _format_large_number(value: float) -> str:
    return f"{float(value):.3e}"


def format_experiment_cost_lines(experiment_cost: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(experiment_cost, Mapping) or len(experiment_cost) == 0:
        return []

    lines = [
        "",
        "Experiment cost",
        f"Core wall time total (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_total')):.6f}",
        f"Core wall time training (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_training')):.6f}",
        f"Testing-phase wall time (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_testing')):.6f}",
        f"Forward+backward NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_training_flops')))}",
        f"Forward-only NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_inference_flops')))}",
        f"Total NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_total_flops')))}",
        (
            "Forward+backward NN evaluations: "
            f"{_safe_int(experiment_cost.get('training_forward_backward_sample_equivalents'))}"
        ),
        f"Forward-only NN evaluations: {_safe_int(experiment_cost.get('inference_forward_calls'))}",
    ]

    if "actual_optimizer_updates" in experiment_cost:
        lines.append(
            f"Actual optimizer updates: {_safe_int(experiment_cost.get('actual_optimizer_updates'))}"
        )
    else:
        lines.append(f"Optimizer steps: {_safe_int(experiment_cost.get('optimizer_steps'))}")

    if "extra_gradient_evaluations" in experiment_cost:
        lines.append(
            "Extra gradient evaluations used to check the current constrained joint state: "
            f"{_safe_int(experiment_cost.get('extra_gradient_evaluations'))}"
        )

    if "forward_only_beam_evaluations" in experiment_cost:
        lines.append(
            f"Forward-only beam evaluations: {_safe_int(experiment_cost.get('forward_only_beam_evaluations'))}"
        )

    workload = experiment_cost.get("workload_counters", {})
    if isinstance(workload, Mapping) and len(workload) > 0:
        lines.append("Workload counters:")
        lines.append(str(workload))

    notes = experiment_cost.get("notes", [])
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)) and len(notes) > 0:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return lines


def build_uplink_convergence_cost(
    system_params: Mapping[str, Any],
    convergence_data: Mapping[str, Any],
    *,
    core_wall_time_seconds_total: float,
) -> dict[str, Any]:
    model_specs = build_uplink_sigma_context_specs(system_params)
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="uplink")
    block_results = convergence_data.get("all_user_block_results_train", [])
    solver_epochs_per_user, solver_calls = _summed_uplink_solver_epochs(block_results)
    visited_states_per_user = _summed_nested_state_count(block_results)

    training_forward_backward_equivalents = int(sum(solver_epochs_per_user))
    inference_forward_calls = int(sum(visited_states_per_user))
    estimated_training_flops = float(
        sum(
            FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * solver_epochs_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(solver_epochs_per_user)))
        )
    )
    estimated_inference_flops = float(
        sum(
            per_user_forward_flops[k] * visited_states_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(visited_states_per_user)))
        )
    )

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(training_forward_backward_equivalents),
        "inference_forward_calls": int(inference_forward_calls),
        "optimizer_steps": int(training_forward_backward_equivalents),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "solver_calls": int(solver_calls),
            "solver_epochs_per_user": [int(v) for v in solver_epochs_per_user],
            "visited_candidate_n_states_per_user": [int(v) for v in visited_states_per_user],
        },
        "notes": [
            "Forward+backward NN FLOPs count the online solver epochs that optimize the uplink precoder net.",
            "Forward-only NN FLOPs count the forward-only beam evaluations inside the same convergence routine.",
            "Testing-phase wall time is 0.0 for convergence because there is no separate train/test split.",
        ],
    }


def build_downlink_convergence_cost(
    system_params: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    core_wall_time_seconds_total: float,
) -> dict[str, Any]:
    model_specs = build_downlink_channel_only_specs(system_params)
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="downlink")
    epoch_history = result.get("epoch_history", [])
    user_update_steps = max(1, int(sim_params.get("user_update_steps", 1)))

    training_forward_backward_equivalents = 0
    inference_forward_calls = 0
    optimizer_steps = 0
    extra_gradient_evaluations = 0
    forward_only_beam_evaluations = 0
    constrained_epochs = 0
    unconstrained_epochs = 0
    estimated_training_flops = 0.0
    estimated_inference_flops = 0.0

    for entry in epoch_history:
        if not isinstance(entry, Mapping):
            continue
        updated_user_ids = [int(v) for v in entry.get("updated_user_ids", entry.get("user_ids", []))]
        if "kkt_primal_residual" in entry:
            constrained_epochs += 1
            extra_joint_eval_factor = 1
        else:
            unconstrained_epochs += 1
            extra_joint_eval_factor = 0
        for user_id in updated_user_ids:
            if user_id < 0 or user_id >= len(per_user_forward_flops):
                continue
            forward_flops = per_user_forward_flops[user_id]
            training_equivalents = user_update_steps + extra_joint_eval_factor
            training_forward_backward_equivalents += int(training_equivalents)
            optimizer_steps += int(user_update_steps)
            extra_gradient_evaluations += int(extra_joint_eval_factor)
            inference_forward_calls += 1
            forward_only_beam_evaluations += 1
            estimated_training_flops += FORWARD_BACKWARD_FLOP_FACTOR * float(forward_flops) * float(training_equivalents)
            estimated_inference_flops += float(forward_flops)

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(training_forward_backward_equivalents),
        "inference_forward_calls": int(inference_forward_calls),
        "optimizer_steps": int(optimizer_steps),
        "actual_optimizer_updates": int(optimizer_steps),
        "extra_gradient_evaluations": int(extra_gradient_evaluations),
        "forward_only_beam_evaluations": int(forward_only_beam_evaluations),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "joint_block_epochs": int(len(epoch_history)),
            "constrained_joint_block_epochs": int(constrained_epochs),
            "unconstrained_joint_block_epochs": int(unconstrained_epochs),
            "user_update_steps_per_epoch": int(user_update_steps),
            "updated_user_calls": int(sum(_safe_int(entry.get("updated_users", 0)) for entry in epoch_history if isinstance(entry, Mapping))),
        },
        "notes": [
            "Forward+backward NN FLOPs count the optimizer updates plus the extra gradient evaluations used to check the current constrained joint state.",
            "Forward-only NN FLOPs count the forward-only beam evaluations inside the same convergence routine.",
            "Testing-phase wall time is 0.0 for convergence because there is no separate train/test split.",
        ],
    }


def build_uplink_monte_carlo_training_cost(
    train_artifact: Mapping[str, Any],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
) -> dict[str, Any]:
    model_specs = list(train_artifact.get("user_model_specs", []))
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="uplink")
    training_history = train_artifact.get("precoder_net_training_history", {})
    rollout_counts_per_user = _sum_numeric_sequence_mappings(
        training_history.get("cumulative_rollout_queries_by_n_kl", {}).get(
            "per_user_rollout_queries_by_n_kl_over_all_epochs",
            [],
        )
    )
    rollout_summaries_per_user = training_history.get("rollout_query_summaries_per_user", [])
    optimizer_steps = 0
    for user_summaries in rollout_summaries_per_user:
        if not isinstance(user_summaries, Sequence):
            continue
        for epoch_summary in user_summaries:
            if not isinstance(epoch_summary, Mapping):
                continue
            total_queries = int(epoch_summary.get("total_rollout_queries", 0))
            optimizer_steps += int(math.ceil(total_queries / max(int(batch_size), 1)))

    train_eval_states_per_user = _summed_nested_state_count(train_artifact.get("all_user_block_results_train", []))
    estimated_training_flops = float(
        sum(
            FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * rollout_counts_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(rollout_counts_per_user)))
        )
    )
    estimated_inference_flops = float(
        sum(
            per_user_forward_flops[k] * train_eval_states_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(train_eval_states_per_user)))
        )
    )

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(sum(rollout_counts_per_user)),
        "inference_forward_calls": int(sum(train_eval_states_per_user)),
        "optimizer_steps": int(optimizer_steps),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "training_rollout_queries_per_user": [int(v) for v in rollout_counts_per_user],
            "train_eval_candidate_n_states_per_user": [int(v) for v in train_eval_states_per_user],
            "batch_size": int(batch_size),
        },
        "notes": [
            "Forward+backward NN FLOPs count the rollout-query training passes that optimize the uplink precoder net.",
            "Forward-only NN FLOPs count the forward-only beam evaluations used in the post-training train-eval pass.",
        ],
    }


def build_uplink_monte_carlo_total_cost(
    train_artifact: Mapping[str, Any],
    test_state_counts_per_user: Sequence[int],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
    core_wall_time_seconds_testing: float,
) -> dict[str, Any]:
    training_cost = build_uplink_monte_carlo_training_cost(
        train_artifact,
        batch_size=batch_size,
        core_wall_time_seconds_training=core_wall_time_seconds_training,
    )
    per_user_forward_flops = [int(v) for v in training_cost.get("per_user_forward_flops", [])]
    test_states_per_user = [int(v) for v in test_state_counts_per_user]
    testing_inference_flops = float(
        sum(
            per_user_forward_flops[k] * test_states_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(test_states_per_user)))
        )
    )

    combined_workload = dict(training_cost.get("workload_counters", {}))
    combined_workload["test_candidate_n_states_per_user"] = [int(v) for v in test_states_per_user]

    return {
        **training_cost,
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training + core_wall_time_seconds_testing),
        "core_wall_time_seconds_testing": float(core_wall_time_seconds_testing),
        "estimated_nn_inference_flops": float(training_cost.get("estimated_nn_inference_flops", 0.0) + testing_inference_flops),
        "estimated_nn_total_flops": float(
            training_cost.get("estimated_nn_training_flops", 0.0)
            + training_cost.get("estimated_nn_inference_flops", 0.0)
            + testing_inference_flops
        ),
        "inference_forward_calls": int(training_cost.get("inference_forward_calls", 0) + sum(test_states_per_user)),
        "workload_counters": combined_workload,
        "notes": [
            "Training FLOPs are estimated from rollout-query counts and precoder MLP shapes; plotting and file export are excluded from wall time.",
            "Inference forward calls include both the post-training train-eval pass and the held-out test pass.",
        ],
    }


def build_downlink_monte_carlo_training_cost(
    artifact: Mapping[str, Any],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
) -> dict[str, Any]:
    model_specs = list(artifact.get("user_model_specs", []))
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="downlink")
    training_history = artifact.get("precoder_net_training_history", {})
    rollout_counts_per_user = _sum_numeric_sequence_mappings(
        training_history.get("cumulative_rollout_queries_by_n_kl", {}).get(
            "per_user_active_user_rollout_queries_by_n_kl_over_all_epochs",
            [],
        )
    )
    epoch_rollout_summaries = training_history.get("epoch_rollout_query_summaries", [])
    optimizer_steps = 0
    for epoch_summary in epoch_rollout_summaries:
        if not isinstance(epoch_summary, Mapping):
            continue
        total_queries = int(epoch_summary.get("total_rollout_queries", 0))
        optimizer_steps += int(math.ceil(total_queries / max(int(batch_size), 1)))

    estimated_training_flops = float(
        sum(
            FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * rollout_counts_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(rollout_counts_per_user)))
        )
    )

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": 0.0,
        "estimated_nn_total_flops": float(estimated_training_flops),
        "training_forward_backward_sample_equivalents": int(sum(rollout_counts_per_user)),
        "inference_forward_calls": 0,
        "optimizer_steps": int(optimizer_steps),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "training_active_user_rollout_queries_per_user": [int(v) for v in rollout_counts_per_user],
            "batch_size": int(batch_size),
        },
        "notes": [
            "Forward+backward NN FLOPs count the active-user rollout-query training passes that optimize the downlink precoder nets.",
            "Forward-only NN FLOPs are 0.0 here because the downlink training phase does not include a separate train-eval pass.",
        ],
    }


def build_downlink_monte_carlo_total_cost(
    artifact: Mapping[str, Any],
    evaluation_cost_counters: Mapping[str, Any] | None,
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
    core_wall_time_seconds_testing: float,
) -> dict[str, Any]:
    training_cost = build_downlink_monte_carlo_training_cost(
        artifact,
        batch_size=batch_size,
        core_wall_time_seconds_training=core_wall_time_seconds_training,
    )
    per_user_forward_flops = [int(v) for v in training_cost.get("per_user_forward_flops", [])]
    per_user_eval_calls = [
        int(v)
        for v in (evaluation_cost_counters or {}).get(
            "per_user_forward_calls",
            [],
        )
    ]
    eval_inference_flops = float(
        sum(
            per_user_forward_flops[k] * per_user_eval_calls[k]
            for k in range(min(len(per_user_forward_flops), len(per_user_eval_calls)))
        )
    )

    combined_workload = dict(training_cost.get("workload_counters", {}))
    combined_workload["test_forward_calls_per_user"] = [int(v) for v in per_user_eval_calls]
    combined_workload["test_total_forward_calls"] = int((evaluation_cost_counters or {}).get("total_forward_calls", 0))

    return {
        **training_cost,
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training + core_wall_time_seconds_testing),
        "core_wall_time_seconds_testing": float(core_wall_time_seconds_testing),
        "estimated_nn_inference_flops": float(eval_inference_flops),
        "estimated_nn_total_flops": float(training_cost.get("estimated_nn_training_flops", 0.0) + eval_inference_flops),
        "inference_forward_calls": int((evaluation_cost_counters or {}).get("total_forward_calls", 0)),
        "workload_counters": combined_workload,
        "notes": [
            "Forward+backward NN FLOPs count the active-user rollout-query training passes that optimize the downlink precoder nets.",
            "Forward-only NN FLOPs count the forward-only beam evaluations in the held-out downlink test pass.",
        ],
    }
