from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from determinism import configure_determinism
from downlink_system import DownlinkSystem
from optimizer import (
    _clone_precoders,
    _collect_interference_diagnostics,
    _compute_user_link_budget,
    _ensure_user_block,
    _evaluate_block_candidate,
    _expand_precoders_for_plan,
    _power_to_db,
    _zero_block_precoder,
    estimate_initial_latency_from_random_precoders,
)
from precoder_models import (
    DEVICE,
    build_user_precoder_net_with_blocklength,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_blocklength,
    infer_precoder_torch_with_blocklength,
)

LOG2E_SQ = float(np.log2(np.e) ** 2)


def _to_complex_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.complex64, copy=False)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


def _rate_to_max_bits(n_kl: int, rate: float) -> int:
    return int(np.floor(float(n_kl) * float(rate)))


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


def _training_block_ids(sim_params: dict[str, Any]) -> list[int]:
    max_total_blocks = max(1, int(sim_params.get("max_total_blocks", 1)))
    blocks_per_seed = max(1, int(sim_params.get("precoder_net_train_blocks_per_seed", 1)))
    return list(range(min(max_total_blocks, blocks_per_seed)))


def _build_training_n_kl_values(
    T_ref: int,
    n_min: int,
    fine_step: int,
    coarse_step: int,
) -> list[int]:
    T_ref = int(T_ref)
    n_min = min(int(n_min), T_ref)
    fine_step = max(1, int(fine_step))
    coarse_step = max(fine_step, int(coarse_step))
    if T_ref <= n_min:
        return [T_ref]
    if (T_ref - n_min) < coarse_step:
        return list(range(T_ref, n_min - 1, -fine_step))

    n_values = [T_ref]
    current = int(T_ref)
    while current - coarse_step >= n_min:
        current -= coarse_step
        n_values.append(int(current))

    if n_values[-1] != n_min:
        current = int(n_values[-1]) - fine_step
        while current >= n_min:
            n_values.append(int(current))
            current -= fine_step

    seen: set[int] = set()
    ordered_values: list[int] = []
    for n_kl in n_values:
        if int(n_kl) in seen:
            continue
        seen.add(int(n_kl))
        ordered_values.append(int(n_kl))
    return ordered_values


def _training_active_masks(K: int, sim_params: dict[str, Any]) -> list[np.ndarray]:
    enumerate_all_up_to_k = max(1, int(sim_params.get("precoder_net_train_enumerate_all_masks_up_to_k", 3)))
    masks: list[np.ndarray] = []
    if int(K) <= enumerate_all_up_to_k:
        for mask_idx in range(1, 1 << int(K)):
            mask = np.array(
                [1.0 if (mask_idx >> bit) & 1 else 0.0 for bit in range(int(K))],
                dtype=np.float32,
            )
            masks.append(mask)
    else:
        masks.append(np.ones(int(K), dtype=np.float32))
        for k in range(int(K)):
            single = np.zeros(int(K), dtype=np.float32)
            single[k] = 1.0
            masks.append(single)
        for i, j in combinations(range(int(K)), 2):
            pair = np.zeros(int(K), dtype=np.float32)
            pair[int(i)] = 1.0
            pair[int(j)] = 1.0
            masks.append(pair)
        for k in range(int(K)):
            leave_one_out = np.ones(int(K), dtype=np.float32)
            leave_one_out[k] = 0.0
            if float(np.sum(leave_one_out)) > 0.0:
                masks.append(leave_one_out)

    unique_masks: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for mask in masks:
        key = tuple(int(v > 0.5) for v in mask.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_masks.append(mask)
    unique_masks.sort(key=lambda mask: (-int(np.sum(mask)), tuple(-int(v > 0.5) for v in mask.tolist())))
    return unique_masks


def _build_training_n_target_levels(
    system: DownlinkSystem,
    active_mask: Sequence[int | float],
    sim_params: dict[str, Any],
) -> list[list[int]]:
    K = int(system.K)
    n_min = int(sim_params["n_kl_min"])
    fine_step = int(sim_params["n_kl_step"])
    coarse_step = max(fine_step, int(sim_params.get("precoder_net_train_n_kl_coarse_step", 5)))
    active_users = [int(k) for k in range(K) if float(active_mask[int(k)]) > 0.5]
    if len(active_users) == 0:
        return []

    candidate_values_by_user = {
        int(k): _build_training_n_kl_values(
            T_ref=int(system.T[int(k)]),
            n_min=int(n_min),
            fine_step=int(fine_step),
            coarse_step=int(coarse_step),
        )
        for k in active_users
    }
    level_count = min(len(candidate_values_by_user[int(k)]) for k in active_users)
    levels: list[list[int]] = []
    for level_idx in range(level_count):
        n_targets = [0 for _ in range(K)]
        for k in active_users:
            n_targets[int(k)] = int(candidate_values_by_user[int(k)][level_idx])
        levels.append(n_targets)
    return levels


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


def _scenario_forward_pass(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
) -> dict[str, Any]:
    K = int(system_params["K"])
    active_mask = np.asarray(scenario["active_mask"], dtype=np.float32)
    n_targets_list = [int(v) for v in n_targets]
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
            infer_precoder_torch_with_blocklength(
                user_models[k],
                H_block_t,
                int(n_targets_list[k]),
                active_mask_t,
                noise_cov_input_t,
                float(scenario["epsilon"][k]),
                int(system_params["Nb"][k]),
                int(system_params["dk"][k]),
                float(scenario["P"][k]),
            )
        )

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
        required_rate = float(scenario["min_bits_required"][k]) / float(max(int(n_targets_list[k]), 1))
        rates[k] = rate
        powers[k] = power
        required_rates[k] = required_rate
        sum_rate = sum_rate + rate

    return {
        "active_mask": active_mask,
        "n_targets": n_targets_list,
        "predicted_beams": predicted_beams,
        "rates": rates,
        "powers": powers,
        "required_rates": required_rates,
        "sum_rate": sum_rate,
    }


def _scenario_metrics_with_models(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
) -> dict[str, Any]:
    K = int(system_params["K"])
    with torch.no_grad():
        forward = _scenario_forward_pass(system_params, scenario, user_models, n_targets)

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
        if margin < -1e-9:
            feasible = False

    active_margins = [rate_margins[k] for k in active_users]
    return {
        "feasible": bool(feasible),
        "active_users": active_users,
        "rate_values": rate_values,
        "required_rates": required_rates,
        "rate_margins": rate_margins,
        "min_rate_margin": float(min(active_margins)) if active_margins else 0.0,
        "sum_rate": float(forward["sum_rate"].detach().cpu()),
    }


def _best_joint_n_target_reduction(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    current_n_targets: Sequence[int],
    *,
    n_min: int,
    n_step: int,
) -> dict[str, Any] | None:
    candidate_users = [
        int(k)
        for k, active in enumerate(scenario["active_mask"])
        if int(active) > 0 and int(current_n_targets[int(k)]) - int(n_step) >= int(n_min)
    ]
    if len(candidate_users) == 0:
        return None

    for subset_size in range(len(candidate_users), 0, -1):
        best_option: dict[str, Any] | None = None
        best_key: tuple[float, float] | None = None
        for subset in combinations(candidate_users, subset_size):
            candidate_n_targets = [int(v) for v in current_n_targets]
            for k in subset:
                candidate_n_targets[int(k)] -= int(n_step)
            metrics = _scenario_metrics_with_models(
                system_params,
                scenario,
                user_models,
                candidate_n_targets,
            )
            if not bool(metrics["feasible"]):
                continue
            option_key = (float(metrics["min_rate_margin"]), float(metrics["sum_rate"]))
            if best_key is None or option_key > best_key:
                best_key = option_key
                best_option = {
                    "reduced_users": [int(k) for k in subset],
                    "candidate_n_targets": [int(v) for v in candidate_n_targets],
                    "metrics": metrics,
                }
        if best_option is not None:
            return best_option
    return None


def _apply_joint_n_target_curriculum(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    scenario_states: Sequence[dict[str, Any]],
    user_models: Sequence[torch.nn.Module],
    *,
    max_rounds_override: int | None = None,
) -> dict[str, Any]:
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    max_rounds = (
        max(0, int(max_rounds_override))
        if max_rounds_override is not None
        else max(0, int(sim_params.get("precoder_net_train_max_reduction_rounds_per_epoch", 4)))
    )
    feasible_scenarios = 0
    reduction_events = 0

    for scenario in scenario_states:
        current_n_targets = [int(v) for v in scenario["current_n_targets"]]
        metrics = _scenario_metrics_with_models(system_params, scenario, user_models, current_n_targets)
        rounds = 0

        while rounds < max_rounds and bool(metrics["feasible"]):
            reduction = _best_joint_n_target_reduction(
                system_params,
                scenario,
                user_models,
                current_n_targets,
                n_min=n_min,
                n_step=n_step,
            )
            if reduction is None:
                break
            current_n_targets = [int(v) for v in reduction["candidate_n_targets"]]
            scenario["current_n_targets"] = [int(v) for v in current_n_targets]
            metrics = reduction["metrics"]
            reduction_events += int(len(reduction["reduced_users"]))
            rounds += 1

        if bool(metrics["feasible"]):
            feasible_scenarios += 1

    return {
        "feasible_training_case_fraction": float(feasible_scenarios) / float(max(len(scenario_states), 1)),
        "curriculum_reduction_events": int(reduction_events),
    }


def _should_apply_curriculum(epoch_num: int, sim_params: dict[str, Any]) -> bool:
    warmup_epochs = max(0, int(sim_params.get("precoder_net_train_curriculum_warmup_epochs", 0)))
    interval_epochs = max(1, int(sim_params.get("precoder_net_train_curriculum_interval_epochs", 1)))
    epoch_num = int(epoch_num)
    if epoch_num < max(1, warmup_epochs):
        return False
    return ((epoch_num - max(1, warmup_epochs)) % interval_epochs) == 0


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


def _summarize_training_case_structure(training_cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(training_cases) == 0:
        return _empty_case_count_summary(
            "global_active_user_cases_by_initial_n_kl",
            "per_user_active_user_cases_by_initial_n_kl",
        )

    K = len(training_cases[0]["active_mask"])
    training_cases_by_seed: dict[int, int] = {}
    training_cases_by_block: dict[int, int] = {}
    training_cases_by_active_user_count: dict[int, int] = {}
    training_cases_by_active_mask: dict[str, int] = {}
    active_user_cases_per_user = [0 for _ in range(K)]

    for training_case in training_cases:
        seed = int(training_case["seed"])
        block = int(training_case.get("block", 0))
        active_mask = [int(v) for v in training_case["active_mask"]]
        active_users = [int(k) for k, is_active in enumerate(active_mask) if int(is_active) > 0]
        active_count = len(active_users)
        mask_key = "".join(str(int(v)) for v in active_mask)

        training_cases_by_seed[seed] = training_cases_by_seed.get(seed, 0) + 1
        training_cases_by_block[block] = training_cases_by_block.get(block, 0) + 1
        training_cases_by_active_user_count[active_count] = (
            training_cases_by_active_user_count.get(active_count, 0) + 1
        )
        training_cases_by_active_mask[mask_key] = training_cases_by_active_mask.get(mask_key, 0) + 1

        for k in active_users:
            active_user_cases_per_user[int(k)] += 1

    return {
        "total_training_cases": int(len(training_cases)),
        "training_cases_by_seed": {str(int(k)): int(v) for k, v in sorted(training_cases_by_seed.items())},
        "training_cases_by_block": {str(int(k)): int(v) for k, v in sorted(training_cases_by_block.items())},
        "training_cases_by_active_user_count": {
            str(int(k)): int(v) for k, v in sorted(training_cases_by_active_user_count.items())
        },
        "training_cases_by_active_mask": {
            str(k): int(v) for k, v in sorted(training_cases_by_active_mask.items())
        },
        "active_user_cases_per_user": [int(v) for v in active_user_cases_per_user],
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

    summary = _summarize_training_case_structure(training_cases)
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


def summarize_training_dataset(training_scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_training_cases_with_n_kl(
        training_scenarios,
        n_key="n_targets",
        global_n_key="global_active_user_cases_by_initial_n_kl",
        per_user_n_key="per_user_active_user_cases_by_initial_n_kl",
    )


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
    active_masks = _training_active_masks(K, sim_params)
    block_ids = _training_block_ids(sim_params)
    min_bits_floor = max(1, int(sim_params.get("precoder_net_train_min_bits_required", 1)))
    scenarios: list[dict[str, Any]] = []

    for seed in train_seeds:
        if verbose:
            print(f"\n================ DOWNLINK RAW TRAINING DATA seed={int(seed)} ================")
        configure_determinism(int(seed))
        system = DownlinkSystem(system_params, seed=int(seed))
        for block in block_ids:
            for k in range(K):
                system.ensure_block(k, int(block))
            # Refresh the precoder snapshot after block expansion so F_override
            # always has the same block support as the system state.
            working_F = system.clone_precoders()
            H_block = _context_channels_for_block(system, int(block))

            for active_mask in active_masks:
                active_users = [int(k) for k in range(K) if float(active_mask[int(k)]) > 0.5]
                if len(active_users) == 0:
                    continue

                input_snapshot = _masked_precoder_snapshot(system, working_F, int(block), active_mask)
                min_bits_required = [
                    int(min_bits_floor) if float(active_mask[int(k)]) > 0.5 else 0
                    for k in range(K)
                ]
                n_target_levels = _build_training_n_target_levels(system, active_mask, sim_params)
                for level_idx, initial_n_targets in enumerate(n_target_levels):
                    scenarios.append(
                        {
                            "seed": int(seed),
                            "block": int(block),
                            "n_level_index": int(level_idx),
                            "H_block": [np.asarray(H_kl, dtype=np.complex64) for H_kl in H_block],
                            "active_mask": [int(v > 0.5) for v in active_mask.tolist()],
                            "n_targets": [int(v) for v in initial_n_targets],
                            "min_bits_required": [int(v) for v in min_bits_required],
                            "P": [float(v) for v in system.P.tolist()],
                            "sigma2": [float(v) for v in system.sigma2.tolist()],
                            "epsilon": [float(v) for v in system.epsilon.tolist()],
                            "input_noise_covariances": _scenario_input_noise_covariances(
                                system,
                                input_snapshot,
                                int(block),
                                active_mask,
                            ),
                        }
                    )

    return scenarios


def train_blocklength_aware_precoder_net(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_scenarios: Sequence[dict[str, Any]],
    *,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple[list[torch.nn.Module], dict[str, Any], list[int]]:
    K = int(system_params["K"])
    max_nr = int(np.max(system_params["Nr"]))
    max_nb = int(np.max(system_params["Nb"]))
    models: list[torch.nn.Module] = [
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
    optimizer = torch.optim.Adam(
        [param for model in models for param in model.parameters()],
        lr=float(lr),
    )
    training_history = {
        "per_user_lagrangian": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "sum_rate": [],
        "avg_user_rate": [],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_power_violation": [[] for _ in range(K)],
        "avg_lagrangian": [],
        "avg_rate_violation_over_users": [],
        "avg_power_violation_over_users": [],
        "feasible_training_case_fraction": [],
        "curriculum_reduction_events": [],
        "dataset_summary": summarize_training_dataset(training_scenarios),
        "training_objective": "lagrangian_sum_finite_blocklength_rate_with_fixed_min_bits_objective",
    }
    dataset_sizes = [
        int(sum(int(scenario["active_mask"][k]) for scenario in training_scenarios))
        for k in range(K)
    ]
    scenario_states = [
        {
            **scenario,
            "max_n_targets": [int(v) for v in scenario["n_targets"]],
            "current_n_targets": [int(v) for v in scenario["n_targets"]],
        }
        for scenario in training_scenarios
    ]
    lambda_rate = np.full(K, float(sim_params.get("initial_lambda_rate_constraint", 0.1)), dtype=float)
    lambda_power = np.full(K, float(sim_params.get("initial_lambda_power_constraint", 0.01)), dtype=float)
    lr_rate = float(sim_params.get("lr_rate_constraint", 1e-2))
    lr_power = float(sim_params.get("lr_power_constraint", 1e-3))

    if verbose:
        print(
            "\n================ DOWNLINK JOINT PRECODER NET TRAIN ================\n"
            f"Training cases: {len(scenario_states)} | epochs: {int(epochs)} | batch_size: {int(batch_size)}\n"
            f"Active user-cases per user: {dataset_sizes}"
        )

    if len(scenario_states) == 0:
        return [model.eval() for model in models], training_history, dataset_sizes

    rng = np.random.default_rng(1000)
    indices = np.arange(len(scenario_states))
    cumulative_n_kl_use_global_counts: dict[int, int] = {}
    cumulative_n_kl_use_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    epoch_start_n_kl_use_summaries: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        for model in models:
            model.train()
        epoch_start_summary = _summarize_training_cases_with_n_kl(
            scenario_states,
            n_key="current_n_targets",
            global_n_key="global_active_user_case_uses_by_n_kl_this_epoch",
            per_user_n_key="per_user_active_user_case_uses_by_n_kl_this_epoch",
        )
        epoch_start_summary["epoch"] = int(epoch + 1)
        epoch_start_n_kl_use_summaries.append(epoch_start_summary)
        _accumulate_active_user_case_uses_by_n_kl(
            scenario_states,
            n_key="current_n_targets",
            global_counts=cumulative_n_kl_use_global_counts,
            per_user_counts=cumulative_n_kl_use_per_user_counts,
        )
        rng.shuffle(indices)
        epoch_term_sums = np.zeros(K, dtype=float)
        epoch_term_counts = np.zeros(K, dtype=float)
        epoch_rate_sums = np.zeros(K, dtype=float)
        epoch_sum_rate_sums = 0.0
        epoch_sum_rate_counts = 0.0
        epoch_rate_violation_sums = np.zeros(K, dtype=float)
        epoch_power_violation_sums = np.zeros(K, dtype=float)

        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start : start + max(int(batch_size), 1)]
            optimizer.zero_grad()

            loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
            batch_rate_violation = np.zeros(K, dtype=float)
            batch_power_violation = np.zeros(K, dtype=float)
            batch_active_counts = np.zeros(K, dtype=float)
            total_active_terms = 0.0

            for idx in batch_idx:
                scenario = scenario_states[int(idx)]
                forward = _scenario_forward_pass(
                    system_params,
                    scenario,
                    models,
                    scenario["current_n_targets"],
                )
                active_mask = forward["active_mask"]
                for k in range(K):
                    rate = forward["rates"][k]
                    power = forward["powers"][k]
                    if rate is None or power is None:
                        continue

                    required_rate = float(forward["required_rates"][k])
                    rate_violation = torch.tensor(required_rate, dtype=torch.float32, device=DEVICE) - rate
                    power_violation = power - float(scenario["P"][k])
                    rate_violation_pos = F.relu(rate_violation)
                    power_violation_pos = F.relu(power_violation)
                    term = (
                        -rate
                        + float(lambda_rate[k]) * rate_violation_pos
                        + float(lambda_power[k]) * power_violation_pos
                    )
                    loss = loss + term
                    batch_rate_violation[k] += float(rate_violation_pos.detach().cpu())
                    batch_power_violation[k] += float(power_violation_pos.detach().cpu())
                    batch_active_counts[k] += 1.0
                    total_active_terms += 1.0
                    epoch_term_sums[k] += float(term.detach().cpu())
                    epoch_term_counts[k] += 1.0
                    epoch_rate_sums[k] += float(rate.detach().cpu())
                    epoch_rate_violation_sums[k] += float(rate_violation_pos.detach().cpu())
                    epoch_power_violation_sums[k] += float(power_violation_pos.detach().cpu())
                if float(np.sum(active_mask)) > 0.0:
                    epoch_sum_rate_sums += float(forward["sum_rate"].detach().cpu())
                    epoch_sum_rate_counts += 1.0

            if total_active_terms <= 0.0:
                continue

            loss = loss / float(total_active_terms)
            loss.backward()
            for model in models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for k in range(K):
                if batch_active_counts[k] <= 0.0:
                    continue
                lambda_rate[k] = max(0.0, float(lambda_rate[k]) + lr_rate * (batch_rate_violation[k] / batch_active_counts[k]))
                lambda_power[k] = max(
                    0.0,
                    float(lambda_power[k]) + lr_power * (batch_power_violation[k] / batch_active_counts[k]),
                )

        for model in models:
            model.eval()
        curriculum_applied = _should_apply_curriculum(epoch + 1, sim_params)
        curriculum_stats = _apply_joint_n_target_curriculum(
            system_params,
            sim_params,
            scenario_states,
            models,
            max_rounds_override=None if curriculum_applied else 0,
        )
        epoch_lagrangians = []
        epoch_rates = []
        epoch_rate_violations = []
        epoch_power_violations = []
        for k in range(K):
            avg_term = float(epoch_term_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate = float(epoch_rate_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate_violation = float(epoch_rate_violation_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_power_violation = float(epoch_power_violation_sums[k] / max(epoch_term_counts[k], 1.0))
            training_history["per_user_lagrangian"][k].append(avg_term)
            training_history["per_user_rate"][k].append(avg_rate)
            training_history["avg_rate_violation"][k].append(avg_rate_violation)
            training_history["avg_power_violation"][k].append(avg_power_violation)
            epoch_lagrangians.append(avg_term)
            epoch_rates.append(avg_rate)
            epoch_rate_violations.append(avg_rate_violation)
            epoch_power_violations.append(avg_power_violation)
        avg_sum_rate = float(epoch_sum_rate_sums / max(epoch_sum_rate_counts, 1.0))
        training_history["sum_rate"].append(avg_sum_rate)
        training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)
        training_history["avg_lagrangian"].append(float(np.mean(epoch_lagrangians)) if epoch_lagrangians else 0.0)
        training_history["avg_rate_violation_over_users"].append(
            float(np.mean(epoch_rate_violations)) if epoch_rate_violations else 0.0
        )
        training_history["avg_power_violation_over_users"].append(
            float(np.mean(epoch_power_violations)) if epoch_power_violations else 0.0
        )
        training_history["feasible_training_case_fraction"].append(float(curriculum_stats["feasible_training_case_fraction"]))
        training_history["curriculum_reduction_events"].append(int(curriculum_stats["curriculum_reduction_events"]))
        if verbose:
            print(
                f"Joint precoder-net epoch {epoch + 1}/{int(epochs)}: "
                f"sum_rate={avg_sum_rate:.6f} | "
                f"avg_user_rate={training_history['avg_user_rate'][-1]:.6f} | "
                f"avg_lagrangian={training_history['avg_lagrangian'][-1]:.6f} | "
                f"avg_rate_violation={training_history['avg_rate_violation_over_users'][-1]:.6f} | "
                f"avg_power_violation={training_history['avg_power_violation_over_users'][-1]:.6f} | "
                f"per_user_rate={epoch_rates} | "
                f"per_user_lagrangian={epoch_lagrangians} | "
                f"per_user_rate_violation={epoch_rate_violations} | "
                f"per_user_power_violation={epoch_power_violations} | "
                f"curriculum_applied={int(curriculum_applied)} | "
                f"feasible_case_fraction={float(curriculum_stats['feasible_training_case_fraction']):.4f} | "
                f"reduction_events={int(curriculum_stats['curriculum_reduction_events'])}"
            )

    training_history["post_training_summary"] = {
        "epochs_requested": int(epochs),
        "train_min_bits_required": int(max(1, int(sim_params.get("precoder_net_train_min_bits_required", 1)))),
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
        "per_user_final_power_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["avg_power_violation"]
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
        "final_avg_power_violation": (
            float(training_history["avg_power_violation_over_users"][-1])
            if training_history["avg_power_violation_over_users"]
            else 0.0
        ),
        "best_avg_power_violation": (
            float(min(training_history["avg_power_violation_over_users"]))
            if training_history["avg_power_violation_over_users"]
            else 0.0
        ),
        "final_feasible_training_case_fraction": (
            float(training_history["feasible_training_case_fraction"][-1])
            if training_history["feasible_training_case_fraction"]
            else 0.0
        ),
        "total_curriculum_reduction_events": int(sum(training_history["curriculum_reduction_events"])),
        "cumulative_training_uses_by_n_kl": _serialize_n_kl_case_counts(
            cumulative_n_kl_use_global_counts,
            cumulative_n_kl_use_per_user_counts,
            global_key="global_active_user_case_uses_by_n_kl_over_all_epochs",
            per_user_key="per_user_active_user_case_uses_by_n_kl_over_all_epochs",
        ),
        "epoch_start_n_kl_use_summaries": epoch_start_n_kl_use_summaries,
        "final_training_case_n_kl_summary": _summarize_training_cases_with_n_kl(
            scenario_states,
            n_key="current_n_targets",
            global_n_key="global_active_user_cases_by_final_n_kl",
            per_user_n_key="per_user_active_user_cases_by_final_n_kl",
        ),
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
) -> np.ndarray:
    k = int(user)
    l = int(block)
    H_block = _context_channels_for_block(system, l)
    input_noise_cov = system.get_interference_plus_noise_covariance(k, l, F_override=input_precoders)
    return infer_precoder_numpy_with_blocklength(
        model,
        H_block,
        int(n_kl),
        active_mask,
        np.asarray(input_noise_cov, dtype=np.complex128),
        float(system.epsilon[k]),
        nb=int(system.Nb[k]),
        dk=int(system.dk[k]),
        power_limit=float(system.P[k]),
        device=DEVICE,
    )


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
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    min_bits_required = max(1, int(sim_params.get("precoder_net_train_min_bits_required", 1)))

    F_T = _precoder_net_beam_for_n(system, model, k, l, T_k, active_mask, frozen_F)
    snapshot_T = _clone_precoders(frozen_F)
    snapshot_T[k][l] = np.array(F_T, copy=True)
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
            )
            candidate_snapshot = _clone_precoders(frozen_F)
            candidate_snapshot[k][l] = np.array(F_candidate, copy=True)
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=candidate_snapshot))
            if (float(B_used) / float(candidate)) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                chosen_F = np.array(F_candidate, copy=True)
                candidate -= n_step
            else:
                break

    return int(B_used), int(chosen_n), float(chosen_R), chosen_F


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
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = estimate_initial_latency_from_random_precoders(
        system,
        sim_params,
        allocation_mode="greedy",
    )

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: list[list[int]] = [[] for _ in range(system.K)]
    B_plan: list[list[int]] = [[] for _ in range(system.K)]
    R_plan: list[list[float]] = [[] for _ in range(system.K)]
    working_F = system.clone_precoders()
    sweep_history: list[dict[str, Any]] = []
    outer_history: list[dict[str, Any]] = []
    rate_points: list[dict[str, Any]] = []
    max_blocks = int(sim_params.get("max_total_blocks", 256))
    min_bits_required = max(1, int(sim_params.get("precoder_net_train_min_bits_required", 1)))

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
            )

        if verbose:
            print(
                f"\n=== Precoder-net block {block} | active_users={len(active_users)} | "
                f"remaining_bits={int(np.sum(remaining))} ==="
            )

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        while len(transmit_users) > 0:
            current_eval = _evaluate_block_candidate(system, working_F, transmit_users, block)
            infeasible_users = [
                int(user_id)
                for user_id, max_bits in zip(current_eval["user_ids"], current_eval["user_max_bits"])
                if int(max_bits) <= 0
            ]
            if len(infeasible_users) == 0:
                break
            for k in infeasible_users:
                _zero_block_precoder(system, working_F, k, block)
                skipped_users.append(int(k))
            transmit_users = [k for k in transmit_users if int(k) not in infeasible_users]
            if verbose:
                print(f"  precoder-net block={block:02d} skipping users {infeasible_users}")

        allocation_snapshot = _clone_precoders(working_F)
        transmit_mask = [1 if k in transmit_users else 0 for k in range(system.K)]
        block_plans: dict[int, dict[str, Any]] = {}
        for k in active_users:
            if int(k) in skipped_users:
                block_plans[int(k)] = {
                    "B_used": 0,
                    "n_used": int(system.T[int(k)]),
                    "R_used": float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot)),
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

        sweep_history.append(
            {
                "block": int(block),
                "sweep": 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [1.0 for _ in active_users],
                "max_precoder_delta": 0.0,
                "sum_rate": float(sum(user_rates)),
                "weighted_sum_rate": float(sum(user_rates)),
                "blended_objective": float(sum(user_rates)),
                "objective_mode": "precoder_net_forward_pass",
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
                    print(f"  user={int(k):02d} block={int(block):02d} skipped")
                else:
                    print(
                        f"  user={int(k):02d} block={int(block):02d} "
                        f"bits={B_used:4d} n_kl={n_used:4d} "
                        f"required_rate={required_rate:.4f} R_fbl={R_used:.4f}"
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
                f"--- Precoder-net block {int(block)} complete | "
                f"allocated_bits={int(block_bits)} remaining_bits={int(np.sum(remaining))} ---"
            )
        block += 1

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)

    result = {
        "method_name": method_name,
        "objective_mode": "precoder_net_forward_pass",
        "allocation_mode": "greedy",
        "weight_strategy": "remaining_bits",
        "precoder_parameterization": "shared_user_block_context_to_precoder_mlp",
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
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
        "sweep_history": sweep_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in ((precoder_net_training_history or {}).get("per_user_lagrangian", []))
        ],
        "precoder_net_training_history": {
            key: (
                [list(map(float, row)) for row in value]
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)
                else list(map(float, value))
                if isinstance(value, list)
                else value
            )
            for key, value in (precoder_net_training_history or {}).items()
        },
        "train_seeds": [int(v) for v in (train_seeds or [])],
        "training_dataset_sizes": [int(v) for v in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
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
    return {
        "system_params": system_params,
        "sim_params": sim_params,
        "train_seeds": [int(v) for v in train_seeds],
        "training_dataset_sizes": [int(v) for v in training_dataset_sizes],
        "training_active_user_case_counts_per_user": [int(v) for v in training_dataset_sizes],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in precoder_net_training_history.get("per_user_lagrangian", [])
        ],
        "precoder_net_training_history": {
            key: (
                [list(map(float, row)) for row in value]
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)
                else list(map(float, value))
                if isinstance(value, list)
                else value
            )
            for key, value in precoder_net_training_history.items()
        },
        "user_model_specs": export_user_model_specs(
            system_params["Nr"],
            system_params["Nb"],
            system_params["dk"],
            uses_blocklength_input=True,
            context_k=int(system_params["K"]),
            context_max_nr=int(np.max(system_params["Nr"])),
            context_max_nb=int(np.max(system_params["Nb"])),
        ),
        "user_model_states": export_user_model_states(user_models),
        "precoder_parameterization": "shared_user_block_context_to_precoder_mlp",
        "training_objective": "lagrangian_sum_finite_blocklength_rate_with_fixed_min_bits_objective",
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
