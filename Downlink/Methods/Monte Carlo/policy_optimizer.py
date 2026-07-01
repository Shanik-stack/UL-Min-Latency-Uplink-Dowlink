from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from determinism import configure_determinism
from downlink_system import DownlinkSystem
from experiment_scenarios import (
    FIXED_BLOCK_TARGETS_MODE,
    PAYLOAD_COMPLETION_MODE,
    build_experiment_scenario,
)
from optimizer import (
    _build_precoder_snapshot_from_models,
    _build_user_precoder_models,
    _clone_precoders,
    _collect_interference_diagnostics,
    _compute_user_link_budget,
    _ensure_user_block,
    _evaluate_block_candidate,
    _expand_precoders_for_plan,
    estimate_initial_latency_from_random_precoders_for_scenario as shared_estimate_initial_latency_from_random_precoders_for_scenario,
    _power_to_db,
    _zero_block_precoder,
)
from precoder_models import (
    DEVICE,
    build_shared_bs_precoder_net_with_blocklength,
    build_user_precoder_net_with_blocklength,
    export_user_model_specs,
    export_user_model_states,
    infer_raw_bs_precoders_numpy_with_blocklength,
    infer_raw_bs_precoders_torch_with_blocklength,
    infer_raw_precoder_numpy_with_blocklength,
    infer_raw_precoder_torch_with_blocklength,
    model_outputs_full_bs_precoder,
    resolve_downlink_precoder_net_scope,
)
from terminal_logging import format_log_line, format_latency_log_line, format_progress_log_line

LOG2E_SQ = float(np.log2(np.e) ** 2)
CONSTRAINT_LOSS_FORMS = {"plain_lagrangian", "augmented_lagrangian"}


def _to_complex_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.complex64, copy=False)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


def _clone_model_states(models: Sequence[torch.nn.Module]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        for model in models
    ]


def _relative_model_state_change(
    models: Sequence[torch.nn.Module],
    previous_states: Sequence[dict[str, torch.Tensor]] | None,
) -> float:
    if previous_states is None:
        return float("inf")
    delta_norm_sq = 0.0
    reference_norm_sq = 0.0
    for model, model_state in zip(models, previous_states):
        current_state = model.state_dict()
        for key, current_value in current_state.items():
            current_cpu = current_value.detach().cpu()
            previous_cpu = model_state[key]
            delta_norm_sq += float(torch.sum((current_cpu - previous_cpu).pow(2)).item())
            reference_norm_sq += float(torch.sum(previous_cpu.pow(2)).item())
    delta_norm = float(np.sqrt(max(delta_norm_sq, 0.0)))
    reference_norm = float(np.sqrt(max(reference_norm_sq, 0.0)))
    return float(delta_norm / max(reference_norm, 1e-12))


def _serialize_nested_history(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_nested_history(val) for key, val in value.items()}
    if isinstance(value, np.ndarray):
        return _serialize_nested_history(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_serialize_nested_history(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bool)):
        try:
            scalar = value.item()
        except Exception:
            return value
        if isinstance(scalar, (int, float, str, bool)):
            return scalar
    return value


def _rate_to_max_bits(n_kl: int, rate: float) -> int:
    return int(np.floor(float(n_kl) * float(rate)))


def _zero_downlink_precoder(system: DownlinkSystem, user: int) -> np.ndarray:
    k = int(user)
    return np.zeros((int(system.Nb[k]), int(system.dk[k])), dtype=np.complex128)


def _downlink_monte_carlo_precoder_parameterization(model_scope: str) -> str:
    scope = resolve_downlink_precoder_net_scope(model_scope)
    if scope == "bs_shared_net":
        return "bs_shared_block_context_to_full_precoder_mlp"
    return "per_user_block_context_to_precoder_mlp"


def _build_training_user_models(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
) -> list[torch.nn.Module]:
    K = int(system_params["K"])
    max_nr = int(np.max(system_params["Nr"]))
    max_nb = int(np.max(system_params["Nb"]))
    max_dk = int(np.max(system_params["dk"]))
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    if model_scope == "bs_shared_net":
        shared_model = build_shared_bs_precoder_net_with_blocklength(
            k_count=K,
            max_nr=max_nr,
            max_nb=max_nb,
            max_dk=max_dk,
            device=DEVICE,
        )
        return [shared_model for _ in range(K)]

    return [
        build_user_precoder_net_with_blocklength(
            int(system_params["Nr"][k]),
            int(system_params["Nb"][k]),
            int(system_params["dk"][k]),
            k_count=K,
            max_nr=max_nr,
            max_nb=max_nb,
            device=DEVICE,
        )
        for k in range(K)
    ]


def _unique_trainable_parameters(models: Sequence[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen_model_ids: set[int] = set()
    for model in models:
        model_id = id(model)
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        params.extend(list(model.parameters()))
    return params


def _models_output_full_bs_precoder(models: Sequence[torch.nn.Module]) -> bool:
    return len(models) > 0 and model_outputs_full_bs_precoder(models[0])


def _q_inv_torch(epsilon: float, device: torch.device = DEVICE) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    p = torch.tensor(1.0 - float(epsilon), device=device, dtype=torch.float64)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return normal.icdf(p).to(dtype=torch.float32)


def _compute_r_fbl_torch(
    H: torch.Tensor,
    Fmat: torch.Tensor,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor,
) -> torch.Tensor:
    Nr = H.shape[0]
    I = torch.eye(Nr, dtype=torch.complex64, device=H.device)
    noise_cov = noise_plus_interference_cov.to(device=H.device, dtype=torch.complex64)
    noise_cov = 0.5 * (noise_cov + noise_cov.conj().transpose(1, 0))
    noise_cov = noise_cov + (1e-6 * I)

    HF = H @ Fmat
    chol = torch.linalg.cholesky(noise_cov)
    G = torch.linalg.solve(chol, HF)
    A = G @ G.conj().transpose(1, 0)
    A = 0.5 * (A + A.conj().transpose(1, 0))

    sign, logdet = torch.linalg.slogdet(I + A)
    if torch.any(torch.abs(sign) <= 1e-12):
        raise RuntimeError("Non-positive logdet sign while evaluating downlink Monte Carlo rate.")

    C = (logdet / np.log(2.0)).real
    eigvals = torch.linalg.eigvalsh(A)
    V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2).real * LOG2E_SQ
    R = C - torch.sqrt(V / float(max(int(n_kl), 1))) * _q_inv_torch(float(epsilon), device=H.device)
    return R.real


def _resolve_constraint_loss_form(sim_params: dict[str, Any]) -> str:
    mode = str(sim_params.get("constraint_loss_form", "plain_lagrangian")).strip().lower()
    if mode not in CONSTRAINT_LOSS_FORMS:
        known = ", ".join(sorted(CONSTRAINT_LOSS_FORMS))
        raise ValueError(f"Unknown constraint loss form '{mode}'. Expected one of: {known}")
    return mode


def _constraint_violation_activation(value: torch.Tensor, loss_form: str) -> torch.Tensor:
    if loss_form == "plain_lagrangian":
        return F.leaky_relu(value)
    return torch.relu(value)


def _zero_precoder(system: DownlinkSystem, user: int) -> np.ndarray:
    return np.zeros((int(system.Nb[int(user)]), int(system.dk[int(user)])), dtype=np.complex128)


def _context_channels_for_block(system: DownlinkSystem, block: int) -> list[np.ndarray]:
    channels: list[np.ndarray] = []
    for k in range(system.K):
        if int(block) < len(system.H[k]):
            channels.append(np.asarray(system.H[k][int(block)], dtype=np.complex64))
        else:
            channels.append(np.zeros((int(system.Nr[k]), int(system.Nb[k])), dtype=np.complex64))
    return channels


def _shared_n_targets_for_block(
    system: DownlinkSystem,
    active_mask: Sequence[int | float],
    *,
    candidate_user: int | None = None,
    candidate_n_kl: int | None = None,
) -> list[int]:
    n_targets: list[int] = []
    for k in range(system.K):
        if float(active_mask[int(k)]) <= 0.5:
            n_targets.append(0)
            continue
        if candidate_user is not None and int(k) == int(candidate_user):
            n_targets.append(int(candidate_n_kl if candidate_n_kl is not None else system.T[int(k)]))
        else:
            n_targets.append(int(system.T[int(k)]))
    return n_targets


def _shared_precoder_snapshot_for_targets(
    system: DownlinkSystem,
    model: torch.nn.Module,
    block: int,
    n_targets: Sequence[int],
    active_mask: Sequence[int | float],
    inference_counters: dict[str, Any] | None = None,
) -> list[list[np.ndarray]]:
    active_users = [int(k) for k, flag in enumerate(active_mask) if float(flag) > 0.5]
    if inference_counters is not None:
        inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
        per_user = inference_counters.get("per_user_forward_calls")
        if isinstance(per_user, list):
            for k in active_users:
                if 0 <= int(k) < len(per_user):
                    per_user[int(k)] = int(per_user[int(k)]) + 1

    beams = infer_raw_bs_precoders_numpy_with_blocklength(
        model,
        _context_channels_for_block(system, int(block)),
        list(n_targets),
        active_mask,
        np.asarray(system.sigma2, dtype=np.float32),
        np.asarray(system.epsilon, dtype=np.float32),
        system.Nb,
        system.dk,
        device=DEVICE,
    )
    snapshot = system.clone_precoders()
    for k in active_users:
        snapshot[int(k)][int(block)] = np.asarray(beams[int(k)], dtype=np.complex128)
    system.project_block_precoders_to_power(snapshot, int(block), active_users=active_users)
    return snapshot


def _masked_precoder_snapshot(
    system: DownlinkSystem,
    working_F: list[list[np.ndarray]],
    block: int,
    active_mask: Sequence[int | float],
) -> list[list[np.ndarray]]:
    snapshot = _clone_precoders(working_F)
    for k in range(system.K):
        if int(block) < len(snapshot[k]) and float(active_mask[int(k)]) <= 0.5:
            snapshot[k][int(block)] = _zero_precoder(system, k)
    return snapshot


def _scenario_input_noise_covariances(
    system: DownlinkSystem,
    snapshot: list[list[np.ndarray]],
    block: int,
    active_mask: Sequence[int | float],
) -> list[np.ndarray]:
    covariances: list[np.ndarray] = []
    for k in range(system.K):
        if float(active_mask[int(k)]) > 0.5:
            cov = system.get_interference_plus_noise_covariance(int(k), int(block), F_override=snapshot)
        else:
            cov = np.asarray(float(system.sigma2[int(k)]) * np.eye(int(system.Nr[int(k)])), dtype=np.complex128)
        covariances.append(np.asarray(cov, dtype=np.complex128))
    return covariances


def _project_predicted_beams_to_block_power_torch(
    predicted_beams: Sequence[torch.Tensor],
    active_mask: Sequence[int | float] | np.ndarray,
    block_power_budget: float,
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    active = np.asarray(active_mask, dtype=np.float32)
    total_power = torch.zeros((), dtype=torch.float32, device=DEVICE)
    for k, beam in enumerate(predicted_beams):
        if k >= len(active) or float(active[k]) <= 0.5:
            continue
        total_power = total_power + (torch.linalg.norm(beam, ord="fro") ** 2).real
    if float(total_power.detach().cpu()) <= float(eps):
        return [beam for beam in predicted_beams]

    scale = torch.sqrt(
        torch.tensor(float(block_power_budget), dtype=torch.float32, device=DEVICE) / (total_power + eps)
    )
    projected: list[torch.Tensor] = []
    for k, beam in enumerate(predicted_beams):
        if k < len(active) and float(active[k]) > 0.5:
            projected.append(beam * scale.to(beam.dtype))
        else:
            projected.append(beam)
    return projected


def _scenario_forward_pass(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
    anchor_bits: Sequence[int] | None = None,
) -> dict[str, Any]:
    K = int(system_params["K"])
    active_mask = np.asarray(scenario["active_mask"], dtype=np.float32)
    n_targets_list = [int(v) for v in n_targets]
    anchor_bits_list = (
        [int(v) for v in anchor_bits]
        if anchor_bits is not None
        else [int(v) for v in scenario.get("rollout_anchor_bits", [0 for _ in range(K)])]
    )
    if len(anchor_bits_list) < K:
        anchor_bits_list = anchor_bits_list + [0 for _ in range(K - len(anchor_bits_list))]
    active_mask_t = torch.tensor(active_mask, dtype=torch.float32, device=DEVICE)
    H_block_t = [
        torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=DEVICE)
        for H_kl in scenario["H_block"]
    ]
    predicted_beams: list[torch.Tensor] = []
    rates: list[torch.Tensor | None] = [None for _ in range(K)]
    powers: list[torch.Tensor | None] = [None for _ in range(K)]
    required_rates = [0.0 for _ in range(K)]
    sum_rate = torch.zeros((), dtype=torch.float32, device=DEVICE)

    if _models_output_full_bs_precoder(user_models):
        sigma2_t = torch.tensor(np.asarray(scenario["sigma2"]), dtype=torch.float32, device=DEVICE)
        epsilon_t = torch.tensor(np.asarray(scenario["epsilon"]), dtype=torch.float32, device=DEVICE)
        predicted_beams = infer_raw_bs_precoders_torch_with_blocklength(
            user_models[0],
            H_block_t,
            torch.tensor(n_targets_list, dtype=torch.float32, device=DEVICE),
            active_mask_t,
            sigma2_t,
            epsilon_t,
            system_params["Nb"],
            system_params["dk"],
        )
        for k in range(K):
            if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
                predicted_beams[k] = torch.zeros(
                    (int(system_params["Nb"][k]), int(system_params["dk"][k])),
                    dtype=torch.complex64,
                    device=DEVICE,
                )
    else:
        for k in range(K):
            if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
                predicted_beams.append(
                    torch.zeros(
                        (int(system_params["Nb"][k]), int(system_params["dk"][k])),
                        dtype=torch.complex64,
                        device=DEVICE,
                    )
                )
                continue

            noise_cov_input_t = torch.tensor(
                np.asarray(scenario["input_noise_covariances"][k]),
                dtype=torch.complex64,
                device=DEVICE,
            )
            predicted_beams.append(
                infer_raw_precoder_torch_with_blocklength(
                    user_models[k],
                    H_block_t,
                    int(n_targets_list[k]),
                    active_mask_t,
                    noise_cov_input_t,
                    float(scenario["epsilon"][k]),
                    int(system_params["Nb"][k]),
                    int(system_params["dk"][k]),
                    user_index=int(k),
                )
            )

    predicted_beams = _project_predicted_beams_to_block_power_torch(
        predicted_beams,
        active_mask,
        float(scenario["block_power_budget"]),
    )

    block_power = torch.zeros((), dtype=torch.float32, device=DEVICE)

    for k in range(K):
        if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
            continue
        noise_cov_joint = _joint_noise_covariance_torch(
            H_block_t,
            predicted_beams,
            float(scenario["sigma2"][k]),
            k,
            active_mask,
        )
        rate = _compute_r_fbl_torch(
            H_block_t[k],
            predicted_beams[k],
            epsilon=float(scenario["epsilon"][k]),
            n_kl=int(n_targets_list[k]),
            noise_plus_interference_cov=noise_cov_joint,
        )
        power = (torch.linalg.norm(predicted_beams[k], ord="fro") ** 2).real
        required_bits = int(anchor_bits_list[k]) if int(k) < len(anchor_bits_list) else 0
        required_rate = (
            float(required_bits) / float(max(int(n_targets_list[k]), 1))
            if required_bits > 0
            else 0.0
        )
        rates[k] = rate
        powers[k] = power
        required_rates[k] = required_rate
        sum_rate = sum_rate + rate
        block_power = block_power + power

    block_power_gap = block_power - torch.tensor(
        float(scenario["block_power_budget"]),
        dtype=torch.float32,
        device=DEVICE,
    )
    block_power_violation = torch.relu(block_power_gap)

    return {
        "active_mask": active_mask,
        "n_targets": n_targets_list,
        "predicted_beams": predicted_beams,
        "rates": rates,
        "powers": powers,
        "rollout_anchor_bits": [int(v) for v in anchor_bits_list],
        "required_rates": required_rates,
        "sum_rate": sum_rate,
        "block_power": block_power,
        "block_power_gap": block_power_gap,
        "block_power_violation": block_power_violation,
    }


def _resolve_downlink_rollout_anchor_bits_from_forward(forward: dict[str, Any]) -> list[int]:
    anchor_bits = [0 for _ in range(len(forward["n_targets"]))]
    for k, rate_t in enumerate(forward["rates"]):
        if rate_t is None:
            continue
        if float(forward["active_mask"][k]) <= 0.5 or int(forward["n_targets"][k]) <= 0:
            continue
        achievable_bits = int(
            np.floor(
                max(float(rate_t.detach().cpu()), 0.0)
                * float(max(int(forward["n_targets"][k]), 1))
            )
        )
        anchor_bits[int(k)] = max(1, achievable_bits)
    return anchor_bits


def _scenario_metrics_from_forward(forward: dict[str, Any]) -> dict[str, Any]:
    K = int(len(forward["rates"]))
    rate_values = [0.0 for _ in range(K)]
    required_rates = [0.0 for _ in range(K)]
    rate_margins = [0.0 for _ in range(K)]
    active_users: list[int] = []
    feasible = True

    for k in range(K):
        rate_t = forward["rates"][k]
        if rate_t is None:
            continue
        active_users.append(int(k))
        rate_val = float(rate_t.detach().cpu())
        required_rate = float(forward["required_rates"][k])
        margin = float(rate_val - required_rate)
        rate_values[k] = rate_val
        required_rates[k] = required_rate
        rate_margins[k] = margin
        if margin < 0.0:
            feasible = False
    if float(forward["block_power_gap"].detach().cpu()) > 0.0:
        feasible = False

    active_margins = [rate_margins[k] for k in active_users]
    return {
        "feasible": bool(feasible),
        "active_users": active_users,
        "rate_values": rate_values,
        "required_rates": required_rates,
        "rate_margins": rate_margins,
        "rollout_anchor_bits": [int(v) for v in forward.get("rollout_anchor_bits", [0 for _ in range(K)])],
        "min_rate_margin": float(min(active_margins)) if active_margins else 0.0,
        "sum_rate": float(forward["sum_rate"].detach().cpu()),
        "block_power_gap": float(forward["block_power_gap"].detach().cpu()),
    }


def _scenario_metrics_with_models(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
    anchor_bits: Sequence[int] | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        forward = _scenario_forward_pass(system_params, scenario, user_models, n_targets, anchor_bits=anchor_bits)
    return _scenario_metrics_from_forward(forward)


def _best_joint_n_target_transition(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    current_n_targets: Sequence[int],
    anchor_bits: Sequence[int],
    *,
    n_min: int,
    n_step: int,
) -> dict[str, Any]:
    candidate_users = [
        int(k)
        for k, active in enumerate(scenario["active_mask"])
        if int(active) > 0 and int(current_n_targets[int(k)]) - int(n_step) >= int(n_min)
    ]
    if len(candidate_users) == 0:
        return {"accepted": None, "rejected": None}

    best_rejected: dict[str, Any] | None = None
    best_rejected_key: tuple[float, float] | None = None

    for subset_size in range(len(candidate_users), 0, -1):
        best_feasible: dict[str, Any] | None = None
        best_feasible_key: tuple[float, float] | None = None
        for subset in combinations(candidate_users, subset_size):
            candidate_n_targets = [int(v) for v in current_n_targets]
            for k in subset:
                candidate_n_targets[int(k)] -= int(n_step)
            metrics = _scenario_metrics_with_models(
                system_params,
                scenario,
                user_models,
                candidate_n_targets,
                anchor_bits=anchor_bits,
            )
            candidate = {
                "reduced_users": [int(k) for k in subset],
                "candidate_n_targets": [int(v) for v in candidate_n_targets],
                "metrics": metrics,
            }
            rejected_key = (float(metrics["min_rate_margin"]), float(metrics["sum_rate"]))
            if best_rejected_key is None or rejected_key > best_rejected_key:
                best_rejected_key = rejected_key
                best_rejected = candidate
            if not bool(metrics["feasible"]):
                continue
            feasible_key = (float(metrics["sum_rate"]), float(metrics["min_rate_margin"]))
            if best_feasible_key is None or feasible_key > best_feasible_key:
                best_feasible_key = feasible_key
                best_feasible = candidate
        if best_feasible is not None:
            return {"accepted": best_feasible, "rejected": best_rejected}

    return {"accepted": None, "rejected": best_rejected}


def _build_rollout_query_from_downlink_state(
    scenario: dict[str, Any],
    n_targets: Sequence[int],
    metrics: dict[str, Any],
    rollout_anchor_bits: Sequence[int],
    *,
    rollout_stage: str,
    frontier_query: bool,
) -> dict[str, Any]:
    query_weight = 2.0 if bool(frontier_query) else (1.25 if not bool(metrics["feasible"]) else 1.0)
    return {
        **scenario,
        "n_targets": [int(v) for v in n_targets],
        "rollout_anchor_bits": [int(v) for v in rollout_anchor_bits],
        "rollout_stage": str(rollout_stage),
        "frontier_query": bool(frontier_query),
        "rollout_feasible": bool(metrics["feasible"]),
        "rollout_min_rate_margin": float(metrics["min_rate_margin"]),
        "rollout_sum_rate": float(metrics["sum_rate"]),
        "query_weight": float(query_weight),
    }


def _generate_rollout_queries_for_downlink(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    user_models: Sequence[torch.nn.Module],
) -> list[dict[str, Any]]:
    n_min = int(sim_params["n_kl_min"])
    fine_step = max(1, int(sim_params["n_kl_step"]))
    coarse_step = max(fine_step, int(sim_params.get("monte_carlo_training_n_kl_coarse_step", fine_step)))
    phases = [("coarse", int(coarse_step))]
    if int(fine_step) < int(coarse_step):
        phases.append(("fine", int(fine_step)))

    rollout_queries: list[dict[str, Any]] = []
    for episode in training_episodes:
        visited_states: set[tuple[int, ...]] = set()
        episode_queries: list[dict[str, Any]] = []
        current_n_targets = [int(v) for v in episode["max_n_targets"]]
        initial_forward = _scenario_forward_pass(
            system_params,
            episode,
            user_models,
            current_n_targets,
            anchor_bits=None,
        )
        rollout_anchor_bits = _resolve_downlink_rollout_anchor_bits_from_forward(initial_forward)
        initial_metrics = _scenario_metrics_with_models(
            system_params,
            episode,
            user_models,
            current_n_targets,
            anchor_bits=rollout_anchor_bits,
        )
        state_key = tuple(int(v) for v in current_n_targets)
        visited_states.add(state_key)
        episode_queries.append(
            _build_rollout_query_from_downlink_state(
                episode,
                current_n_targets,
                initial_metrics,
                rollout_anchor_bits,
                rollout_stage="coarse",
                frontier_query=not bool(initial_metrics["feasible"]),
            )
        )
        last_feasible_idx = 0 if bool(initial_metrics["feasible"]) else None

        if bool(initial_metrics["feasible"]):
            for stage_name, step_size in phases:
                while True:
                    transition = _best_joint_n_target_transition(
                        system_params,
                        episode,
                        user_models,
                        current_n_targets,
                        rollout_anchor_bits,
                        n_min=n_min,
                        n_step=int(step_size),
                    )
                    accepted = transition.get("accepted")
                    if accepted is None:
                        rejected = transition.get("rejected")
                        if rejected is not None:
                            rejected_key = tuple(int(v) for v in rejected["candidate_n_targets"])
                            if rejected_key not in visited_states:
                                visited_states.add(rejected_key)
                                episode_queries.append(
                                    _build_rollout_query_from_downlink_state(
                                        episode,
                                        rejected["candidate_n_targets"],
                                        rejected["metrics"],
                                        rollout_anchor_bits,
                                        rollout_stage=stage_name,
                                        frontier_query=True,
                                    )
                                )
                        break

                    current_n_targets = [int(v) for v in accepted["candidate_n_targets"]]
                    state_key = tuple(int(v) for v in current_n_targets)
                    if state_key in visited_states:
                        break
                    visited_states.add(state_key)
                    episode_queries.append(
                        _build_rollout_query_from_downlink_state(
                            episode,
                            current_n_targets,
                            accepted["metrics"],
                            rollout_anchor_bits,
                            rollout_stage=stage_name,
                            frontier_query=False,
                        )
                    )
                    last_feasible_idx = len(episode_queries) - 1

        if last_feasible_idx is not None:
            episode_queries[int(last_feasible_idx)]["frontier_query"] = True
            episode_queries[int(last_feasible_idx)]["query_weight"] = max(
                float(episode_queries[int(last_feasible_idx)]["query_weight"]),
                2.0,
            )

        rollout_queries.extend(episode_queries)

    return rollout_queries


def _summarize_downlink_rollout_queries(rollout_queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_training_cases_with_n_kl(
        rollout_queries,
        n_key="n_targets",
        global_n_key="global_active_user_rollout_queries_by_n_kl",
        per_user_n_key="per_user_active_user_rollout_queries_by_n_kl",
    )
    frontier_queries = [query for query in rollout_queries if bool(query.get("frontier_query", False))]
    frontier_summary = _summarize_training_cases_with_n_kl(
        frontier_queries,
        n_key="n_targets",
        global_n_key="global_active_user_frontier_rollout_queries_by_n_kl",
        per_user_n_key="per_user_active_user_frontier_rollout_queries_by_n_kl",
    )
    feasible_queries = int(sum(1 for query in rollout_queries if bool(query.get("rollout_feasible", False))))
    infeasible_queries = int(len(rollout_queries) - feasible_queries)
    return {
        **summary,
        **frontier_summary,
        "total_rollout_queries": int(len(rollout_queries)),
        "feasible_rollout_queries": int(feasible_queries),
        "infeasible_rollout_queries": int(infeasible_queries),
        "frontier_rollout_queries": int(len(frontier_queries)),
    }


def _joint_noise_covariance_torch(
    H_block: Sequence[torch.Tensor],
    predicted_beams: Sequence[torch.Tensor],
    sigma2: float,
    user: int,
    active_mask: Sequence[int | float],
) -> torch.Tensor:
    k = int(user)
    Hk = H_block[k]
    Nrk = int(Hk.shape[0])
    cov = float(sigma2) * torch.eye(Nrk, dtype=torch.complex64, device=Hk.device)
    for j, Fj in enumerate(predicted_beams):
        if int(j) == k or float(active_mask[int(j)]) <= 0.5:
            continue
        HFj = Hk @ Fj
        cov = cov + (HFj @ HFj.conj().transpose(1, 0))
    cov = 0.5 * (cov + cov.conj().transpose(1, 0))
    cov = cov + (1e-6 * torch.eye(Nrk, dtype=torch.complex64, device=Hk.device))
    return cov


def _empty_case_count_summary(global_n_key: str, per_user_n_key: str) -> dict[str, Any]:
    return {
        "total_training_cases": 0,
        "training_cases_by_seed": {},
        "training_cases_by_active_user_count": {},
        "training_cases_by_active_mask": {},
        "active_user_cases_per_user": [],
        global_n_key: {},
        per_user_n_key: [],
    }


def _serialize_n_kl_case_counts(
    global_counts: dict[int, int],
    per_user_counts: Sequence[dict[int, int]],
    *,
    global_key: str,
    per_user_key: str,
) -> dict[str, Any]:
    return {
        global_key: {str(int(k)): int(v) for k, v in sorted(global_counts.items())},
        per_user_key: [
            {str(int(k)): int(v) for k, v in sorted(user_counts.items())}
            for user_counts in per_user_counts
        ],
    }


def _summarize_channel_episode_structure(training_episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(training_episodes) == 0:
        return {
            "total_channel_episodes": 0,
            "channel_episodes_by_seed": {},
            "channel_episodes_by_block": {},
            "channel_episodes_by_active_user_count": {},
            "channel_episodes_by_active_mask": {},
            "channel_episodes_per_user": [],
        }

    K = len(training_episodes[0]["active_mask"])
    channel_episodes_by_seed: dict[int, int] = {}
    channel_episodes_by_block: dict[int, int] = {}
    channel_episodes_by_active_user_count: dict[int, int] = {}
    channel_episodes_by_active_mask: dict[str, int] = {}
    channel_episodes_per_user = [0 for _ in range(K)]

    for episode in training_episodes:
        seed = int(episode["seed"])
        block = int(episode.get("block", 0))
        active_mask = [int(v) for v in episode["active_mask"]]
        active_users = [int(k) for k, is_active in enumerate(active_mask) if int(is_active) > 0]
        active_count = len(active_users)
        mask_key = "".join(str(int(v)) for v in active_mask)

        channel_episodes_by_seed[seed] = channel_episodes_by_seed.get(seed, 0) + 1
        channel_episodes_by_block[block] = channel_episodes_by_block.get(block, 0) + 1
        channel_episodes_by_active_user_count[active_count] = (
            channel_episodes_by_active_user_count.get(active_count, 0) + 1
        )
        channel_episodes_by_active_mask[mask_key] = channel_episodes_by_active_mask.get(mask_key, 0) + 1
        for k in active_users:
            channel_episodes_per_user[int(k)] += 1

    return {
        "total_channel_episodes": int(len(training_episodes)),
        "channel_episodes_by_seed": {str(int(k)): int(v) for k, v in sorted(channel_episodes_by_seed.items())},
        "channel_episodes_by_block": {str(int(k)): int(v) for k, v in sorted(channel_episodes_by_block.items())},
        "channel_episodes_by_active_user_count": {
            str(int(k)): int(v) for k, v in sorted(channel_episodes_by_active_user_count.items())
        },
        "channel_episodes_by_active_mask": {
            str(k): int(v) for k, v in sorted(channel_episodes_by_active_mask.items())
        },
        "channel_episodes_per_user": [int(v) for v in channel_episodes_per_user],
    }


def _count_active_user_cases_by_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
) -> tuple[dict[int, int], list[dict[int, int]]]:
    if len(training_cases) == 0:
        return {}, []

    K = len(training_cases[0]["active_mask"])
    global_counts: dict[int, int] = {}
    per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]

    for training_case in training_cases:
        active_mask = [int(v) for v in training_case["active_mask"]]
        n_targets = training_case[n_key]
        for k, is_active in enumerate(active_mask):
            if int(is_active) <= 0:
                continue
            n_val = int(n_targets[int(k)])
            per_user_counts[int(k)][n_val] = per_user_counts[int(k)].get(n_val, 0) + 1
            global_counts[n_val] = global_counts.get(n_val, 0) + 1

    return global_counts, per_user_counts


def _summarize_training_cases_with_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
    global_n_key: str,
    per_user_n_key: str,
) -> dict[str, Any]:
    if len(training_cases) == 0:
        return _empty_case_count_summary(global_n_key, per_user_n_key)

    summary = _summarize_channel_episode_structure(training_cases)
    global_counts, per_user_counts = _count_active_user_cases_by_n_kl(training_cases, n_key=n_key)
    summary.update(
        _serialize_n_kl_case_counts(
            global_counts,
            per_user_counts,
            global_key=global_n_key,
            per_user_key=per_user_n_key,
        )
    )
    return summary


def summarize_training_dataset(training_episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_channel_episode_structure(training_episodes)
    summary["base_dataset_kind"] = "channel_episodes_only"
    summary["scenario_modes"] = sorted(
        {str(case.get("scenario_mode", PAYLOAD_COMPLETION_MODE)) for case in training_episodes}
    )
    return summary


def _accumulate_active_user_case_uses_by_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
    global_counts: dict[int, int],
    per_user_counts: list[dict[int, int]],
) -> None:
    if len(training_cases) == 0:
        return
    K = len(training_cases[0]["active_mask"])
    if len(per_user_counts) == 0:
        per_user_counts.extend({} for _ in range(K))

    current_global_counts, current_per_user_counts = _count_active_user_cases_by_n_kl(training_cases, n_key=n_key)
    for n_val, count in current_global_counts.items():
        global_counts[int(n_val)] = global_counts.get(int(n_val), 0) + int(count)
    for k in range(K):
        for n_val, count in current_per_user_counts[k].items():
            per_user_counts[k][int(n_val)] = per_user_counts[k].get(int(n_val), 0) + int(count)


def build_training_dataset(
    train_seeds: Sequence[int],
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    K = int(system_params["K"])
    scenario_mode = str(sim_params.get("experiment_scenario_mode", PAYLOAD_COMPLETION_MODE))
    episodes: list[dict[str, Any]] = []
    blocks_per_seed = max(1, int(sim_params.get("monte_carlo_training_blocks_per_seed", 1)))
    if verbose and blocks_per_seed != 1:
        print(
            "[DL Monte Carlo Train] Base dataset now uses one joint channel episode per seed; "
            "ignoring monte_carlo_training_blocks_per_seed during training-data construction."
        )

    for seed in train_seeds:
        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Dataset]",
                    phase="collect",
                    seed=int(seed),
                    base_dataset="channel_episode_only",
                )
            )
        configure_determinism(int(seed))
        system = DownlinkSystem(system_params, seed=int(seed))
        block = 0
        active_mask = np.ones(K, dtype=np.float32)
        for k in range(K):
            system.ensure_block(k, int(block))
        H_block = _context_channels_for_block(system, int(block))
        input_snapshot = system.clone_precoders()
        for k in range(K):
            if int(block) < len(input_snapshot[k]):
                input_snapshot[k][int(block)] = _zero_precoder(system, k)
        episodes.append(
            {
                "seed": int(seed),
                "block": int(block),
                "H_block": [np.asarray(H_kl, dtype=np.complex64) for H_kl in H_block],
                "active_mask": [int(v > 0.5) for v in active_mask.tolist()],
                "max_n_targets": [int(system.T[int(k)]) for k in range(K)],
                "block_power_budget": float(system.block_power_budget),
                "P": [float(v) for v in system.P.tolist()],
                "sigma2": [float(v) for v in system.sigma2.tolist()],
                "epsilon": [float(v) for v in system.epsilon.tolist()],
                "input_noise_covariances": _scenario_input_noise_covariances(
                    system,
                    input_snapshot,
                    int(block),
                    active_mask,
                ),
                "scenario_mode": scenario_mode,
            }
        )

    return episodes


def train_blocklength_aware_precoder_net(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple[list[torch.nn.Module], dict[str, Any], list[int]]:
    K = int(system_params["K"])
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    dataset_summary = summarize_training_dataset(training_episodes)
    models = _build_training_user_models(system_params, sim_params)
    optimizer = torch.optim.Adam(
        _unique_trainable_parameters(models),
        lr=float(lr),
    )
    training_history = {
        "per_user_lagrangian": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "sum_rate": [],
        "avg_user_rate": [],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_block_power_violation": [],
        "avg_lagrangian": [],
        "avg_rate_violation_over_users": [],
        "dataset_summary": dataset_summary,
        "epoch_rollout_query_summaries": [],
        "downlink_precoder_net_scope": str(model_scope),
        "training_objective": "rollout_lagrangian_sum_finite_blocklength_rate_with_online_full_block_anchor_bits",
    }
    dataset_sizes = [
        int(dataset_summary.get("channel_episodes_per_user", [0 for _ in range(K)])[k])
        for k in range(K)
    ]
    constraint_loss_form = _resolve_constraint_loss_form(sim_params)
    augmented_lagrangian_rho_rate = float(sim_params.get("augmented_lagrangian_rho_rate", 0.0))
    augmented_lagrangian_rho_power = float(sim_params.get("augmented_lagrangian_rho_power", 0.0))
    lambda_rate = np.full(K, float(sim_params.get("initial_lambda_rate_constraint", 0.1)), dtype=float)
    lambda_power_block = float(sim_params.get("initial_lambda_power_constraint", 0.01))
    lr_rate = float(sim_params.get("lr_rate_constraint", 1e-2))
    lr_power = float(sim_params.get("lr_power_constraint", 1e-3))

    if verbose:
        print(
            format_log_line(
                "[DL Monte Carlo Train]",
                phase="start",
                scope="joint",
                channel_episodes=int(len(training_episodes)),
                channel_episodes_per_user=[int(v) for v in dataset_sizes],
                epochs=int(epochs),
                batch_size=int(batch_size),
                precoder_net_scope=str(model_scope),
            )
        )

    if len(training_episodes) == 0:
        return [model.eval() for model in models], training_history, dataset_sizes

    rng = np.random.default_rng(1000)
    cumulative_rollout_query_global_counts: dict[int, int] = {}
    cumulative_rollout_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    cumulative_frontier_query_global_counts: dict[int, int] = {}
    cumulative_frontier_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    final_epoch_rollout_summary: dict[str, Any] = {}
    previous_epoch_model_states: list[dict[str, torch.Tensor]] | None = None

    for epoch in range(int(epochs)):
        for model in models:
            model.eval()
        rollout_queries = _generate_rollout_queries_for_downlink(
            system_params,
            sim_params,
            training_episodes,
            models,
        )
        final_epoch_rollout_summary = _summarize_downlink_rollout_queries(rollout_queries)
        final_epoch_rollout_summary["epoch"] = int(epoch + 1)
        training_history["epoch_rollout_query_summaries"].append(final_epoch_rollout_summary)

        for n_key, count in final_epoch_rollout_summary.get("global_active_user_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_rollout_query_global_counts[n_val] = (
                cumulative_rollout_query_global_counts.get(n_val, 0) + int(count)
            )
        for k, user_counts in enumerate(
            final_epoch_rollout_summary.get("per_user_active_user_rollout_queries_by_n_kl", [])
        ):
            for n_key, count in user_counts.items():
                n_val = int(n_key)
                cumulative_rollout_query_per_user_counts[k][n_val] = (
                    cumulative_rollout_query_per_user_counts[k].get(n_val, 0) + int(count)
                )
        for n_key, count in final_epoch_rollout_summary.get(
            "global_active_user_frontier_rollout_queries_by_n_kl",
            {},
        ).items():
            n_val = int(n_key)
            cumulative_frontier_query_global_counts[n_val] = (
                cumulative_frontier_query_global_counts.get(n_val, 0) + int(count)
            )
        for k, user_counts in enumerate(
            final_epoch_rollout_summary.get("per_user_active_user_frontier_rollout_queries_by_n_kl", [])
        ):
            for n_key, count in user_counts.items():
                n_val = int(n_key)
                cumulative_frontier_query_per_user_counts[k][n_val] = (
                    cumulative_frontier_query_per_user_counts[k].get(n_val, 0) + int(count)
                )

        for model in models:
            model.train()
        indices = np.arange(len(rollout_queries))
        rng.shuffle(indices)
        epoch_term_sums = np.zeros(K, dtype=float)
        epoch_term_counts = np.zeros(K, dtype=float)
        epoch_rate_sums = np.zeros(K, dtype=float)
        epoch_sum_rate_sums = 0.0
        epoch_sum_rate_weight = 0.0
        epoch_rate_violation_sums = np.zeros(K, dtype=float)
        epoch_block_power_violation_sum = 0.0

        if len(rollout_queries) == 0:
            for k in range(K):
                training_history["per_user_lagrangian"][k].append(0.0)
                training_history["per_user_rate"][k].append(0.0)
                training_history["avg_rate_violation"][k].append(0.0)
            training_history["sum_rate"].append(0.0)
            training_history["avg_user_rate"].append(0.0)
            training_history["avg_lagrangian"].append(0.0)
            training_history["avg_rate_violation_over_users"].append(0.0)
            training_history["avg_block_power_violation"].append(0.0)
            if verbose:
                print(
                    format_progress_log_line(
                        "[DL Monte Carlo]",
                        phase="train",
                        method="monte_carlo",
                        scope="joint",
                        epoch=f"{epoch + 1}/{int(epochs)}",
                        objective=0.0,
                        sum_rate=0.0,
                        avg_user_rate=0.0,
                        r_p=0.0,
                        r_c=0.0,
                        r_s=0.0,
                        status="no_rollout_queries",
                    )
                )
            continue

        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start : start + max(int(batch_size), 1)]
            optimizer.zero_grad()

            loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
            batch_rate_violation = np.zeros(K, dtype=float)
            batch_active_counts = np.zeros(K, dtype=float)
            batch_block_power_violation = 0.0
            batch_block_count = 0.0
            total_active_weight = 0.0

            for idx in batch_idx:
                scenario = rollout_queries[int(idx)]
                query_weight = float(scenario.get("query_weight", 1.0))
                forward = _scenario_forward_pass(
                    system_params,
                    scenario,
                    models,
                    scenario["n_targets"],
                )
                active_mask = forward["active_mask"]
                scenario_term = torch.zeros((), dtype=torch.float32, device=DEVICE)
                scenario_has_active_user = False
                for k in range(K):
                    rate = forward["rates"][k]
                    power = forward["powers"][k]
                    if rate is None or power is None:
                        continue

                    required_rate = float(forward["required_rates"][k])
                    rate_violation = torch.tensor(required_rate, dtype=torch.float32, device=DEVICE) - rate
                    rate_violation_pos = _constraint_violation_activation(rate_violation, constraint_loss_form)
                    term = (
                        -rate
                        + float(lambda_rate[k]) * rate_violation_pos
                    )
                    if constraint_loss_form == "augmented_lagrangian":
                        term = term + 0.5 * augmented_lagrangian_rho_rate * rate_violation_pos.pow(2)
                    scenario_term = scenario_term + term
                    batch_rate_violation[k] += float(query_weight) * float(rate_violation_pos.detach().cpu())
                    batch_active_counts[k] += float(query_weight)
                    total_active_weight += float(query_weight)
                    epoch_term_sums[k] += float(query_weight) * float(term.detach().cpu())
                    epoch_term_counts[k] += float(query_weight)
                    epoch_rate_sums[k] += float(query_weight) * float(rate.detach().cpu())
                    epoch_rate_violation_sums[k] += float(query_weight) * float(rate_violation_pos.detach().cpu())
                    scenario_has_active_user = True
                block_power_violation = forward["block_power_violation"]
                scenario_term = scenario_term + float(lambda_power_block) * block_power_violation
                if constraint_loss_form == "augmented_lagrangian":
                    scenario_term = (
                        scenario_term
                        + 0.5 * augmented_lagrangian_rho_power * block_power_violation.pow(2)
                    )
                loss = loss + (float(query_weight) * scenario_term)
                if scenario_has_active_user:
                    batch_block_power_violation += float(query_weight) * float(block_power_violation.detach().cpu())
                    batch_block_count += float(query_weight)
                    epoch_block_power_violation_sum += float(query_weight) * float(block_power_violation.detach().cpu())
                if float(np.sum(active_mask)) > 0.0:
                    epoch_sum_rate_sums += float(query_weight) * float(forward["sum_rate"].detach().cpu())
                    epoch_sum_rate_weight += float(query_weight)

            if total_active_weight <= 0.0:
                continue

            loss = loss / float(total_active_weight)
            loss.backward()
            for model in models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for k in range(K):
                if batch_active_counts[k] <= 0.0:
                    continue
                lambda_rate[k] = max(0.0, float(lambda_rate[k]) + lr_rate * (batch_rate_violation[k] / batch_active_counts[k]))
            if batch_block_count > 0.0:
                lambda_power_block = max(
                    0.0,
                    float(lambda_power_block) + lr_power * (batch_block_power_violation / batch_block_count),
                )

        for model in models:
            model.eval()
        epoch_lagrangians = []
        epoch_rates = []
        epoch_rate_violations = []
        avg_block_power_violation = float(epoch_block_power_violation_sum / max(epoch_sum_rate_weight, 1.0))
        for k in range(K):
            avg_term = float(epoch_term_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate = float(epoch_rate_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate_violation = float(epoch_rate_violation_sums[k] / max(epoch_term_counts[k], 1.0))
            training_history["per_user_lagrangian"][k].append(avg_term)
            training_history["per_user_rate"][k].append(avg_rate)
            training_history["avg_rate_violation"][k].append(avg_rate_violation)
            epoch_lagrangians.append(avg_term)
            epoch_rates.append(avg_rate)
            epoch_rate_violations.append(avg_rate_violation)
        avg_sum_rate = float(epoch_sum_rate_sums / max(epoch_sum_rate_weight, 1.0))
        training_history["sum_rate"].append(avg_sum_rate)
        training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)
        training_history["avg_lagrangian"].append(float(np.mean(epoch_lagrangians)) if epoch_lagrangians else 0.0)
        training_history["avg_rate_violation_over_users"].append(
            float(np.mean(epoch_rate_violations)) if epoch_rate_violations else 0.0
        )
        training_history["avg_block_power_violation"].append(avg_block_power_violation)
        epoch_r_p = float(max(max(epoch_rate_violations, default=0.0), max(avg_block_power_violation, 0.0)))
        epoch_r_c = float(
            max(
                max(
                    (
                        abs(float(lambda_rate[k]) * max(float(epoch_rate_violations[k]), 0.0))
                        for k in range(min(K, len(epoch_rate_violations)))
                    ),
                    default=0.0,
                ),
                abs(float(lambda_power_block) * max(avg_block_power_violation, 0.0)),
            )
        )
        epoch_r_s = _relative_model_state_change(models, previous_epoch_model_states)
        previous_epoch_model_states = _clone_model_states(models)
        epoch_status = "epoch_budget_reached" if int(epoch + 1) >= int(epochs) else "running"
        if verbose:
            print(
                format_progress_log_line(
                    "[DL Monte Carlo]",
                    phase="train",
                    method="monte_carlo",
                    scope="joint",
                    epoch=f"{epoch + 1}/{int(epochs)}",
                    objective=float(training_history["avg_lagrangian"][-1]),
                    sum_rate=float(avg_sum_rate),
                    avg_user_rate=float(training_history["avg_user_rate"][-1]),
                    r_p=epoch_r_p,
                    r_c=epoch_r_c,
                    r_s=epoch_r_s,
                    status=epoch_status,
                )
            )

    training_history["post_training_summary"] = {
        "epochs_requested": int(epochs),
        "downlink_precoder_net_scope": str(model_scope),
        "base_dataset_kind": dataset_summary.get("base_dataset_kind", "channel_episodes_only"),
        "rollout_anchor_bits_mode": "derived_online_from_current_joint_full_block_rate",
        "per_user_final_lagrangian": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_lagrangian"]
        ],
        "per_user_best_lagrangian": [
            float(min(history)) if len(history) > 0 else 0.0 for history in training_history["per_user_lagrangian"]
        ],
        "per_user_final_rate": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_rate"]
        ],
        "per_user_final_rate_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["avg_rate_violation"]
        ],
        "final_avg_sum_rate": float(training_history["sum_rate"][-1]) if training_history["sum_rate"] else 0.0,
        "best_avg_sum_rate": float(max(training_history["sum_rate"])) if training_history["sum_rate"] else 0.0,
        "final_avg_user_rate": (
            float(training_history["avg_user_rate"][-1]) if training_history["avg_user_rate"] else 0.0
        ),
        "best_avg_user_rate": float(max(training_history["avg_user_rate"])) if training_history["avg_user_rate"] else 0.0,
        "final_avg_lagrangian": (
            float(training_history["avg_lagrangian"][-1]) if training_history["avg_lagrangian"] else 0.0
        ),
        "best_avg_lagrangian": float(min(training_history["avg_lagrangian"])) if training_history["avg_lagrangian"] else 0.0,
        "final_avg_rate_violation": (
            float(training_history["avg_rate_violation_over_users"][-1])
            if training_history["avg_rate_violation_over_users"]
            else 0.0
        ),
        "best_avg_rate_violation": (
            float(min(training_history["avg_rate_violation_over_users"]))
            if training_history["avg_rate_violation_over_users"]
            else 0.0
        ),
        "final_avg_block_power_violation": (
            float(training_history["avg_block_power_violation"][-1])
            if training_history["avg_block_power_violation"]
            else 0.0
        ),
        "best_avg_block_power_violation": (
            float(min(training_history["avg_block_power_violation"]))
            if training_history["avg_block_power_violation"]
            else 0.0
        ),
        "final_feasible_rollout_query_fraction": (
            float(final_epoch_rollout_summary.get("feasible_rollout_queries", 0))
            / float(max(int(final_epoch_rollout_summary.get("total_rollout_queries", 0)), 1))
        ),
        "cumulative_rollout_queries_by_n_kl": _serialize_n_kl_case_counts(
            cumulative_rollout_query_global_counts,
            cumulative_rollout_query_per_user_counts,
            global_key="global_active_user_rollout_queries_by_n_kl_over_all_epochs",
            per_user_key="per_user_active_user_rollout_queries_by_n_kl_over_all_epochs",
        ),
        "cumulative_frontier_rollout_queries_by_n_kl": _serialize_n_kl_case_counts(
            cumulative_frontier_query_global_counts,
            cumulative_frontier_query_per_user_counts,
            global_key="global_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs",
            per_user_key="per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs",
        ),
        "final_epoch_rollout_query_summary": final_epoch_rollout_summary,
    }

    return [model.eval() for model in models], training_history, dataset_sizes


def _precoder_net_beam_for_n(
    system: DownlinkSystem,
    model: torch.nn.Module,
    user: int,
    block: int,
    n_kl: int,
    active_mask: Sequence[int | float],
    input_precoders: list[list[np.ndarray]],
    inference_counters: dict[str, Any] | None = None,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    if model_outputs_full_bs_precoder(model):
        snapshot = _shared_precoder_snapshot_for_targets(
            system,
            model,
            l,
            _shared_n_targets_for_block(
                system,
                active_mask,
                candidate_user=k,
                candidate_n_kl=int(n_kl),
            ),
            active_mask,
            inference_counters=inference_counters,
        )
        return np.asarray(snapshot[k][l], dtype=np.complex128)

    if inference_counters is not None:
        inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
        per_user = inference_counters.get("per_user_forward_calls")
        if isinstance(per_user, list) and 0 <= k < len(per_user):
            per_user[k] = int(per_user[k]) + 1
    H_block = _context_channels_for_block(system, l)
    input_noise_cov = system.get_interference_plus_noise_covariance(k, l, F_override=input_precoders)
    return infer_raw_precoder_numpy_with_blocklength(
        model,
        H_block,
        int(n_kl),
        active_mask,
        np.asarray(input_noise_cov, dtype=np.complex128),
        float(system.epsilon[k]),
        nb=int(system.Nb[k]),
        dk=int(system.dk[k]),
        device=DEVICE,
        user_index=int(k),
    )


def _allocate_fixed_target_for_user_block_snapshot(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    user: int,
    block: int,
    target_bits: int,
    sim_params: dict[str, Any],
    *,
    allow_infeasible_zero: bool = False,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    zero_beam = _zero_downlink_precoder(system, k)

    if int(target_bits) <= 0:
        return 0, 0, 0.0, zero_beam

    F_fixed = np.asarray(frozen_F[k][l], dtype=np.complex128)
    snapshot = _clone_precoders(frozen_F)
    snapshot[k][l] = np.array(F_fixed, copy=True)
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    B_used = int(min(int(target_bits), B_max))
    if int(B_used) <= 0 and allow_infeasible_zero:
        return 0, T_k, 0.0, zero_beam

    chosen_n = int(T_k)
    chosen_R = float(R_T)
    if int(B_used) >= int(target_bits) and int(target_bits) > 0:
        candidate = T_k - n_step
        while candidate >= n_min:
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=snapshot))
            if (float(target_bits) / float(max(int(candidate), 1))) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                candidate -= int(n_step)
            else:
                break
    return int(B_used), int(chosen_n), float(chosen_R), np.array(F_fixed, copy=True)


def _allocate_bits_for_user_block_precoder_net(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    model: torch.nn.Module,
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    *,
    allow_infeasible_zero: bool = False,
    inference_counters: dict[str, Any] | None = None,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    F_T = _precoder_net_beam_for_n(
        system,
        model,
        k,
        l,
        T_k,
        active_mask,
        frozen_F,
        inference_counters=inference_counters,
    )
    snapshot_T = _clone_precoders(frozen_F)
    snapshot_T[k][l] = np.array(F_T, copy=True)
    active_users = [int(user_id) for user_id, flag in enumerate(active_mask) if float(flag) > 0.5]
    system.project_block_precoders_to_power(snapshot_T, l, active_users=active_users)
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot_T))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    if B_max <= 0:
        if allow_infeasible_zero:
            return 0, T_k, R_T, np.zeros_like(F_T)
        raise RuntimeError(
            f"Precoder-net user {k} block {l} infeasible at n=T={T_k}; R_T={R_T:.6f}, B_max={B_max}."
        )

    B_used = int(min(int(remaining_bits), B_max))
    chosen_n = int(T_k)
    chosen_R = float(R_T)
    chosen_F = np.array(F_T, copy=True)

    if int(remaining_bits) <= B_max:
        candidate = T_k - n_step
        while candidate >= n_min:
            F_candidate = _precoder_net_beam_for_n(
                system,
                model,
                k,
                l,
                int(candidate),
                active_mask,
                frozen_F,
                inference_counters=inference_counters,
            )
            candidate_snapshot = _clone_precoders(frozen_F)
            candidate_snapshot[k][l] = np.array(F_candidate, copy=True)
            system.project_block_precoders_to_power(candidate_snapshot, l, active_users=active_users)
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=candidate_snapshot))
            if (float(B_used) / float(candidate)) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                chosen_F = np.array(F_candidate, copy=True)
                candidate -= n_step
            else:
                break

    return int(B_used), int(chosen_n), float(chosen_R), chosen_F


def _allocate_fixed_target_for_user_block_precoder_net(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    model: torch.nn.Module,
    user: int,
    block: int,
    target_bits: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    *,
    allow_infeasible_zero: bool = False,
    inference_counters: dict[str, Any] | None = None,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    zero_beam = _zero_downlink_precoder(system, k)

    if int(target_bits) <= 0:
        return 0, 0, 0.0, zero_beam

    F_T = _precoder_net_beam_for_n(
        system,
        model,
        k,
        l,
        T_k,
        active_mask,
        frozen_F,
        inference_counters=inference_counters,
    )
    snapshot_T = _clone_precoders(frozen_F)
    snapshot_T[k][l] = np.array(F_T, copy=True)
    active_users = [int(user_id) for user_id, flag in enumerate(active_mask) if float(flag) > 0.5]
    system.project_block_precoders_to_power(snapshot_T, l, active_users=active_users)
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot_T))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    B_used = int(min(int(target_bits), B_max))
    if int(B_used) <= 0 and allow_infeasible_zero:
        return 0, T_k, 0.0, zero_beam

    chosen_n = int(T_k)
    chosen_R = float(R_T)
    chosen_F = np.array(F_T, copy=True)
    if int(B_used) >= int(target_bits) and int(target_bits) > 0:
        candidate = T_k - n_step
        while candidate >= n_min:
            F_candidate = _precoder_net_beam_for_n(
                system,
                model,
                k,
                l,
                int(candidate),
                active_mask,
                frozen_F,
                inference_counters=inference_counters,
            )
            candidate_snapshot = _clone_precoders(frozen_F)
            candidate_snapshot[k][l] = np.array(F_candidate, copy=True)
            system.project_block_precoders_to_power(candidate_snapshot, l, active_users=active_users)
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=candidate_snapshot))
            if (float(target_bits) / float(max(int(candidate), 1))) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                chosen_F = np.array(F_candidate, copy=True)
                candidate -= int(n_step)
            else:
                break
    return int(B_used), int(chosen_n), float(chosen_R), chosen_F


def _estimate_initial_latency_from_random_precoders_fixed_block_targets(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline_system = DownlinkSystem(system.sc, seed=system.seed)
    baseline_models = _build_user_precoder_models(
        baseline_system,
        init_seed=int(system.seed),
        model_scope="per_user_nets",
    )
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    n_plan: list[list[int]] = [[] for _ in range(baseline_system.K)]
    B_plan: list[list[int]] = [[] for _ in range(baseline_system.K)]
    R_plan: list[list[float]] = [[] for _ in range(baseline_system.K)]
    skipped_blocks_per_user = [0 for _ in range(baseline_system.K)]
    working_F = _build_precoder_snapshot_from_models(baseline_system, baseline_models)

    for block in range(num_blocks):
        for k in range(baseline_system.K):
            _ensure_user_block(baseline_system, working_F, k, block, use_previous_as_template=False)
        working_F = _build_precoder_snapshot_from_models(baseline_system, baseline_models)

        for k in range(baseline_system.K):
            target_bits = int(block_targets[k, block])

            B_used, n_used, R_used, F_used = _allocate_fixed_target_for_user_block_snapshot(
                baseline_system,
                working_F,
                int(k),
                int(block),
                int(target_bits),
                sim_params,
                allow_infeasible_zero=True,
            )
            working_F[int(k)][int(block)] = np.array(F_used, copy=True)
            if B_used <= 0:
                skipped_blocks_per_user[int(k)] += 1
                n_plan[k].append(int(baseline_system.T[k]))
                B_plan[k].append(0)
                R_plan[k].append(float(R_used))
                continue

            n_plan[k].append(int(n_used))
            B_plan[k].append(int(B_used))
            R_plan[k].append(float(R_used))

    initial_F = _expand_precoders_for_plan(baseline_system, working_F, n_plan)
    baseline_system.apply_solution(initial_F, n_plan)
    latency = baseline_system.latency.tolist()
    initial_plan = {
        "n_kl": n_plan,
        "B_kl": B_plan,
        "R_alloc": R_plan,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "block_bit_targets": block_targets.tolist(),
        "blocks_per_user": [int(len(v)) for v in n_plan],
    }
    return latency, initial_plan, _collect_interference_diagnostics(baseline_system)


def _estimate_initial_latency_from_random_precoders_for_scenario(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    return shared_estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
    )


def _evaluate_downlink_precoder_net_fixed_block_targets(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    *,
    verbose: bool,
    method_name: str,
    scenario: dict[str, Any],
    precoder_net_training_history: dict[str, Any] | None,
    train_seeds: Sequence[int] | None,
    training_dataset_sizes: Sequence[int] | None,
) -> dict[str, Any]:
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="fixed_block_targets",
                method="monte_carlo",
            )
        )

    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    n_plan: list[list[int]] = [[] for _ in range(system.K)]
    B_plan: list[list[int]] = [[] for _ in range(system.K)]
    R_plan: list[list[float]] = [[] for _ in range(system.K)]
    working_F = system.clone_precoders()
    epoch_history: list[dict[str, Any]] = []
    outer_history: list[dict[str, Any]] = []
    rate_points: list[dict[str, Any]] = []
    skipped_blocks_per_user = [0 for _ in range(system.K)]
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(system.K)],
        "total_forward_calls": 0,
    }

    for block in range(num_blocks):
        active_users = list(range(system.K))
        for k in active_users:
            _ensure_user_block(system, working_F, k, block)
        active_mask = [1 for _ in range(system.K)]
        if _models_output_full_bs_precoder(user_models):
            joint_snapshot = _shared_precoder_snapshot_for_targets(
                system,
                user_models[0],
                int(block),
                _shared_n_targets_for_block(system, active_mask),
                active_mask,
                inference_counters=evaluation_cost_counters,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(joint_snapshot[int(k)][int(block)], dtype=np.complex128)
        else:
            input_snapshot = _masked_precoder_snapshot(system, working_F, block, active_mask)
            for k in active_users:
                working_F[k][block] = _precoder_net_beam_for_n(
                    system,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(system.T[int(k)]),
                    active_mask,
                    input_snapshot,
                    inference_counters=evaluation_cost_counters,
                )
        system.project_block_precoders_to_power(working_F, int(block), active_users=[int(k) for k in active_users])
        for k in range(system.K):
            if int(block_targets[k, block]) <= 0 and int(block) < len(working_F[k]):
                _zero_block_precoder(system, working_F, k, block)
        system.project_block_precoders_to_power(
            working_F,
            int(block),
            active_users=[int(k) for k in active_users if int(block_targets[k, block]) > 0],
        )

        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    target_bits=int(np.sum(block_targets[:, block])),
                    mode="fixed_block_targets",
                )
            )

        allocation_snapshot = _clone_precoders(working_F)
        block_plans: dict[int, dict[str, Any]] = {}
        for k in range(system.K):
            target_bits = int(block_targets[k, block])
            B_used, n_used, R_used, F_used = _allocate_fixed_target_for_user_block_precoder_net(
                system,
                allocation_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                int(target_bits),
                sim_params,
                active_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            block_plans[int(k)] = {
                "B_used": int(B_used),
                "n_used": int(n_used if B_used > 0 else int(system.T[int(k)])),
                "R_used": float(R_used),
                "F_used": np.array(F_used, copy=True),
                "skipped": bool(target_bits > 0 and B_used <= 0),
                "target_bits": int(target_bits),
            }

        committed_snapshot = _clone_precoders(working_F)
        for k in range(system.K):
            committed_snapshot[int(k)][block] = np.array(block_plans[int(k)]["F_used"], copy=True)

        corrected_plans: dict[int, dict[str, Any]] = {}
        for k in range(system.K):
            plan = block_plans[int(k)]
            if bool(plan["skipped"]):
                corrected_plans[int(k)] = plan
                continue

            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            F_used = np.array(plan["F_used"], copy=True)
            required_rate = float(B_used) / float(max(n_used, 1))
            actual_rate = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))
            if actual_rate >= required_rate:
                corrected_plans[int(k)] = {
                    **plan,
                    "R_used": float(actual_rate),
                    "F_used": F_used,
                    "skipped": False,
                }
                continue

            B_fix, n_fix, R_fix, F_fix = _allocate_fixed_target_for_user_block_precoder_net(
                system,
                committed_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                int(plan["target_bits"]),
                sim_params,
                active_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            corrected_plans[int(k)] = {
                "B_used": int(B_fix),
                "n_used": int(n_fix if B_fix > 0 else int(system.T[int(k)])),
                "R_used": float(R_fix),
                "F_used": np.array(F_fix, copy=True),
                "skipped": bool(B_fix <= 0),
                "target_bits": int(plan["target_bits"]),
            }

        for k in range(system.K):
            final_plan = corrected_plans[int(k)]
            committed_snapshot[int(k)][block] = np.array(final_plan["F_used"], copy=True)

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot))
            signal_power, interference_power, _, sinr_db = _compute_user_link_budget(
                system,
                allocation_snapshot,
                int(k),
                int(block),
            )
            user_rates.append(rate)
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(_power_to_db(interference_power))
            user_signal_db.append(_power_to_db(signal_power))

        epoch_history.append(
            {
                "block": int(block),
                "epoch": 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [1.0 for _ in active_users],
                "max_precoder_delta": 0.0,
                "sum_rate": float(sum(user_rates)),
                "unweighted_sum_rate": float(sum(user_rates)),
                "blended_objective": float(sum(user_rates)),
                "objective_mode": "unweighted_sum_rate",
            }
        )

        block_bits = 0
        block_unserved_bits = 0
        for k in range(system.K):
            final_plan = corrected_plans[int(k)]
            working_F[int(k)][block] = np.array(final_plan["F_used"], copy=True)
            B_used = int(final_plan["B_used"])
            n_used = int(final_plan["n_used"])
            R_used = float(
                system.compute_block_rate(int(k), int(block), max(int(n_used), 1), F_override=committed_snapshot)
            ) if B_used > 0 else 0.0

            n_plan[int(k)].append(int(n_used))
            B_plan[int(k)].append(int(B_used))
            R_plan[int(k)].append(float(R_used))
            block_bits += int(B_used)
            unserved_bits = max(int(final_plan.get("target_bits", B_used)) - int(B_used), 0)
            block_unserved_bits += int(unserved_bits)
            if bool(final_plan.get("skipped", False)):
                skipped_blocks_per_user[int(k)] += 1

            required_rate = float(B_used) / float(max(n_used, 1)) if B_used > 0 else 0.0
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "target_bits": int(final_plan.get("target_bits", B_used)),
                    "unserved_bits": int(unserved_bits),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": float(R_used) - required_rate,
                    "queue_weight": 1.0,
                    "skipped": bool(final_plan.get("skipped", False)),
                    "partially_served": bool(0 < int(B_used) < int(final_plan.get("target_bits", B_used))),
                }
            )
            if verbose:
                if bool(final_plan.get("skipped", False)):
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                            target_bits=int(final_plan.get("target_bits", 0)),
                        )
                    )
                else:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            target_bits=int(final_plan.get("target_bits", 0)),
                            served_bits=int(B_used),
                            unserved_bits=int(unserved_bits),
                            n_kl=int(n_used),
                            required_rate=float(required_rate),
                            achieved_rate=float(R_used),
                        )
                    )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "skipped_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] <= 0)),
                "allocated_bits": int(block_bits),
                "target_bits": int(np.sum(block_targets[:, block])),
                "unserved_bits": int(block_unserved_bits),
                "future_target_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "remaining_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "feasible_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "min_max_bits": int(min([corrected_plans[int(k)]["B_used"] for k in active_users], default=0)),
                "queue_weights": {int(k): 1.0 for k in active_users},
                "final_precoder_delta": 0.0,
            }
        )

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))

    result = {
        "method_name": method_name,
        "objective_mode": "unweighted_sum_rate",
        "allocation_mode": "fixed_block_targets",
        "weight_strategy": "none",
        "precoder_parameterization": _downlink_monte_carlo_precoder_parameterization(model_scope),
        "downlink_precoder_net_scope": str(model_scope),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
            model_scope=model_scope,
        ),
        "n_kl": [list(map(int, v)) for v in n_plan],
        "B_kl": [list(map(int, v)) for v in B_plan],
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": [list(map(float, v)) for v in R_plan],
        "initial_latency": list(map(float, initial_latency)),
        "initial_plan": initial_plan,
        "initial_interference_diag": initial_interference_diag,
        "final_latency": system.latency.tolist(),
        "initial_snr_db": initial_snr_db,
        "final_snr_db": final_snr_db,
        "initial_sinr_db": initial_sinr_db,
        "final_sinr_db": final_sinr_db,
        "final_interference_diag": final_interference_diag,
        "outer_history": outer_history,
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in ((precoder_net_training_history or {}).get("per_user_lagrangian", []))
        ],
        "precoder_net_training_history": _serialize_nested_history(precoder_net_training_history or {}),
        "train_seeds": [int(v) for v in (train_seeds or [])],
        "training_dataset_sizes": [int(v) for v in (training_dataset_sizes or [])],
        "training_channel_episode_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "evaluation_cost_counters": evaluation_cost_counters,
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }
    return result


def evaluate_downlink_precoder_net(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    *,
    verbose: bool = True,
    method_name: str = "monte_carlo_precoder_net_train_test",
    precoder_net_training_history: dict[str, Any] | None = None,
    train_seeds: Sequence[int] | None = None,
    training_dataset_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    scenario = build_experiment_scenario(system.sc, sim_params, seed=int(system.seed))
    if str(scenario["mode"]) == FIXED_BLOCK_TARGETS_MODE:
        return _evaluate_downlink_precoder_net_fixed_block_targets(
            system,
            sim_params,
            user_models,
            verbose=verbose,
            method_name=method_name,
            scenario=scenario,
            precoder_net_training_history=precoder_net_training_history,
            train_seeds=train_seeds,
            training_dataset_sizes=training_dataset_sizes,
        )
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="payload_completion",
                method="monte_carlo",
            )
        )

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: list[list[int]] = [[] for _ in range(system.K)]
    B_plan: list[list[int]] = [[] for _ in range(system.K)]
    R_plan: list[list[float]] = [[] for _ in range(system.K)]
    working_F = system.clone_precoders()
    epoch_history: list[dict[str, Any]] = []
    outer_history: list[dict[str, Any]] = []
    rate_points: list[dict[str, Any]] = []
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(system.K)],
        "total_forward_calls": 0,
    }
    max_blocks = int(sim_params.get("max_total_blocks", 256))
    block = 0
    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Precoder-net evaluation hit max_total_blocks={max_blocks} with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(system.K) if int(remaining[k]) > 0]
        for k in active_users:
            _ensure_user_block(system, working_F, k, block)
        active_mask = [1 if k in active_users else 0 for k in range(system.K)]
        if _models_output_full_bs_precoder(user_models):
            joint_snapshot = _shared_precoder_snapshot_for_targets(
                system,
                user_models[0],
                int(block),
                _shared_n_targets_for_block(system, active_mask),
                active_mask,
                inference_counters=evaluation_cost_counters,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(joint_snapshot[int(k)][int(block)], dtype=np.complex128)
        else:
            input_snapshot = _masked_precoder_snapshot(system, working_F, block, active_mask)
            for k in active_users:
                working_F[k][block] = _precoder_net_beam_for_n(
                    system,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(system.T[int(k)]),
                    active_mask,
                    input_snapshot,
                    inference_counters=evaluation_cost_counters,
                )
        system.project_block_precoders_to_power(working_F, int(block), active_users=[int(k) for k in active_users])

        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    remaining_bits=int(np.sum(remaining)),
                    mode="payload_completion",
                )
            )

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        skipped_user_rates: dict[int, float] = {}
        while len(transmit_users) > 0:
            current_eval = _evaluate_block_candidate(system, working_F, transmit_users, block)
            infeasible_users = [
                int(user_id)
                for user_id, max_bits in zip(current_eval["user_ids"], current_eval["user_max_bits"])
                if int(max_bits) <= 0
            ]
            if len(infeasible_users) == 0:
                break
            current_eval_rates = {
                int(user_id): float(rate_val)
                for user_id, rate_val in zip(current_eval["user_ids"], current_eval["user_rates"])
            }
            for k in infeasible_users:
                skipped_user_rates[int(k)] = float(current_eval_rates.get(int(k), 0.0))
                _zero_block_precoder(system, working_F, k, block)
                skipped_users.append(int(k))
            transmit_users = [k for k in transmit_users if int(k) not in infeasible_users]
            if verbose:
                print(
                    format_log_line(
                        "[DL Monte Carlo Eval]",
                        block=int(block),
                        skipped_users=[int(k) for k in infeasible_users],
                    )
                )

        allocation_snapshot = _clone_precoders(working_F)
        transmit_mask = [1 if k in transmit_users else 0 for k in range(system.K)]
        block_plans: dict[int, dict[str, Any]] = {}
        for k in active_users:
            if int(k) in skipped_users:
                block_plans[int(k)] = {
                    "B_used": 0,
                    "n_used": int(system.T[int(k)]),
                    "R_used": float(skipped_user_rates.get(int(k), 0.0)),
                    "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                    "skipped": True,
                }
                continue

            B_used, n_used, R_used, F_used = _allocate_bits_for_user_block_precoder_net(
                system,
                allocation_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                int(remaining[int(k)]),
                sim_params,
                transmit_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            block_plans[int(k)] = {
                "B_used": int(B_used),
                "n_used": int(n_used),
                "R_used": float(R_used),
                "F_used": np.array(F_used, copy=True),
                "skipped": bool(B_used <= 0),
            }

        committed_snapshot = _clone_precoders(working_F)
        for k in active_users:
            committed_snapshot[int(k)][block] = np.array(block_plans[int(k)]["F_used"], copy=True)

        corrected_plans: dict[int, dict[str, Any]] = {}
        for k in active_users:
            plan = block_plans[int(k)]
            if bool(plan["skipped"]):
                corrected_plans[int(k)] = plan
                continue

            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            F_used = np.array(plan["F_used"], copy=True)
            required_rate = float(B_used) / float(max(n_used, 1))
            actual_rate = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))
            if actual_rate >= required_rate:
                corrected_plans[int(k)] = {
                    "B_used": B_used,
                    "n_used": n_used,
                    "R_used": actual_rate,
                    "F_used": F_used,
                    "skipped": False,
                }
                continue

            B_fix, n_fix, R_fix, F_fix = _allocate_bits_for_user_block_precoder_net(
                system,
                committed_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                B_used,
                sim_params,
                transmit_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            corrected_plans[int(k)] = {
                "B_used": int(B_fix),
                "n_used": int(n_fix),
                "R_used": float(R_fix),
                "F_used": np.array(F_fix, copy=True),
                "skipped": bool(B_fix <= 0),
            }

        for k in active_users:
            final_plan = corrected_plans[int(k)]
            committed_snapshot[int(k)][block] = np.array(final_plan["F_used"], copy=True)

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot))
            signal_power, interference_power, _, sinr_db = _compute_user_link_budget(
                system, allocation_snapshot, int(k), int(block)
            )
            user_rates.append(rate)
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(_power_to_db(interference_power))
            user_signal_db.append(_power_to_db(signal_power))

        epoch_history.append(
            {
                "block": int(block),
                "epoch": 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [1.0 for _ in active_users],
                "max_precoder_delta": 0.0,
                "sum_rate": float(sum(user_rates)),
                "unweighted_sum_rate": float(sum(user_rates)),
                "blended_objective": float(sum(user_rates)),
                "objective_mode": "unweighted_sum_rate",
            }
        )

        block_bits = 0
        for k in active_users:
            final_plan = corrected_plans[int(k)]
            working_F[int(k)][block] = np.array(final_plan["F_used"], copy=True)
            B_used = int(final_plan["B_used"])
            n_used = int(final_plan["n_used"])
            R_used = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))

            B_plan[int(k)].append(int(B_used))
            n_plan[int(k)].append(int(n_used))
            R_plan[int(k)].append(float(R_used))
            remaining[int(k)] -= int(B_used)
            block_bits += int(B_used)

            required_rate = float(B_used) / float(max(n_used, 1))
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": float(R_used) - required_rate,
                    "queue_weight": 1.0,
                    "skipped": bool(B_used <= 0),
                }
            )
            if verbose:
                if B_used <= 0:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                        )
                    )
                else:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            served_bits=int(B_used),
                            n_kl=int(n_used),
                            required_rate=float(required_rate),
                            achieved_rate=float(R_used),
                        )
                    )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "skipped_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] <= 0)),
                "allocated_bits": int(block_bits),
                "remaining_bits": int(np.sum(remaining)),
                "feasible_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "min_max_bits": int(min([corrected_plans[int(k)]["B_used"] for k in active_users], default=0)),
                "queue_weights": {int(k): 1.0 for k in active_users},
                "final_precoder_delta": 0.0,
            }
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    status="complete",
                    allocated_bits=int(block_bits),
                    remaining_bits=int(np.sum(remaining)),
                )
            )
        block += 1

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))

    result = {
        "method_name": method_name,
        "objective_mode": "unweighted_sum_rate",
        "allocation_mode": "greedy",
        "weight_strategy": "none",
        "precoder_parameterization": _downlink_monte_carlo_precoder_parameterization(model_scope),
        "downlink_precoder_net_scope": str(model_scope),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
            model_scope=model_scope,
        ),
        "n_kl": [list(map(int, v)) for v in n_plan],
        "B_kl": [list(map(int, v)) for v in B_plan],
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": [list(map(float, v)) for v in R_plan],
        "initial_latency": list(map(float, initial_latency)),
        "initial_plan": initial_plan,
        "initial_interference_diag": initial_interference_diag,
        "final_latency": system.latency.tolist(),
        "initial_snr_db": initial_snr_db,
        "final_snr_db": final_snr_db,
        "initial_sinr_db": initial_sinr_db,
        "final_sinr_db": final_sinr_db,
        "final_interference_diag": final_interference_diag,
        "outer_history": outer_history,
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in ((precoder_net_training_history or {}).get("per_user_lagrangian", []))
        ],
        "precoder_net_training_history": _serialize_nested_history(precoder_net_training_history or {}),
        "train_seeds": [int(v) for v in (train_seeds or [])],
        "training_dataset_sizes": [int(v) for v in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
        "skipped_blocks_per_user": [0 for _ in range(system.K)],
        "evaluation_cost_counters": evaluation_cost_counters,
        "scenario_mode": PAYLOAD_COMPLETION_MODE,
    }
    return result


def build_precoder_net_artifact(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    train_seeds: Sequence[int],
    user_models: Sequence[torch.nn.Module],
    precoder_net_training_history: dict[str, Any],
    training_dataset_sizes: Sequence[int],
) -> dict[str, Any]:
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    return {
        "system_params": system_params,
        "sim_params": sim_params,
        "train_seeds": [int(v) for v in train_seeds],
        "training_dataset_sizes": [int(v) for v in training_dataset_sizes],
        "training_channel_episode_counts_per_user": [int(v) for v in training_dataset_sizes],
        "training_active_user_case_counts_per_user": [int(v) for v in training_dataset_sizes],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in precoder_net_training_history.get("per_user_lagrangian", [])
        ],
        "precoder_net_training_history": _serialize_nested_history(precoder_net_training_history),
        "user_model_specs": export_user_model_specs(
            system_params["Nr"],
            system_params["Nb"],
            system_params["dk"],
            uses_blocklength_input=True,
            context_k=int(system_params["K"]),
            context_max_nr=int(np.max(system_params["Nr"])),
            context_max_nb=int(np.max(system_params["Nb"])),
            context_max_dk=int(np.max(system_params["dk"])),
            model_scope=model_scope,
        ),
        "user_model_states": export_user_model_states(user_models),
        "precoder_parameterization": _downlink_monte_carlo_precoder_parameterization(model_scope),
        "downlink_precoder_net_scope": str(model_scope),
        "training_objective": precoder_net_training_history.get(
            "training_objective",
            "lagrangian_sum_finite_blocklength_rate_with_online_full_block_anchor_bits",
        ),
    }

train_blocklength_aware_precoder = train_blocklength_aware_precoder_net
train_blocklength_aware_policy = train_blocklength_aware_precoder_net
_precoder_beam_for_n = _precoder_net_beam_for_n
_policy_beam_for_n = _precoder_net_beam_for_n
_allocate_bits_for_user_block_precoder = _allocate_bits_for_user_block_precoder_net
_allocate_bits_for_user_block_policy = _allocate_bits_for_user_block_precoder_net
evaluate_downlink_precoder = evaluate_downlink_precoder_net
evaluate_downlink_policy = evaluate_downlink_precoder_net
build_precoder_artifact = build_precoder_net_artifact
build_policy_artifact = build_precoder_net_artifact
