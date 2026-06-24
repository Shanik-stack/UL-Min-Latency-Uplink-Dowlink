from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch

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
    build_user_precoder_net_with_context,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_context,
    infer_precoder_torch_with_context,
)


LOG2E_SQ = float(np.log2(np.e) ** 2)


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
        raise RuntimeError("Non-positive logdet sign while evaluating downlink shared-beam rate.")

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
        n_val = int(n_kl)
        if n_val in seen:
            continue
        seen.add(n_val)
        ordered_values.append(n_val)
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


def _serialize_counts(counts: dict[int, int]) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(counts.items())}


def summarize_training_dataset(training_cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(training_cases) == 0:
        return {
            "total_training_cases": 0,
            "total_n_level_evaluations": 0,
            "total_active_user_n_evaluations": 0,
            "training_cases_by_seed": {},
            "training_cases_by_block": {},
            "training_cases_by_active_user_count": {},
            "training_cases_by_active_mask": {},
            "active_user_cases_per_user": [],
            "global_active_user_n_evaluations_by_n_kl": {},
            "per_user_active_user_n_evaluations_by_n_kl": [],
        }

    K = len(training_cases[0]["active_mask"])
    training_cases_by_seed: dict[int, int] = {}
    training_cases_by_block: dict[int, int] = {}
    training_cases_by_active_user_count: dict[int, int] = {}
    training_cases_by_active_mask: dict[str, int] = {}
    active_user_cases_per_user = [0 for _ in range(K)]
    global_active_user_n_evaluations_by_n_kl: dict[int, int] = {}
    per_user_active_user_n_evaluations_by_n_kl = [{} for _ in range(K)]
    total_n_level_evaluations = 0
    total_active_user_n_evaluations = 0

    for training_case in training_cases:
        seed = int(training_case["seed"])
        block = int(training_case["block"])
        active_mask = [int(v) for v in training_case["active_mask"]]
        n_target_levels = training_case["n_target_levels"]
        active_users = [int(k) for k, is_active in enumerate(active_mask) if int(is_active) > 0]
        active_count = len(active_users)
        mask_key = "".join(str(int(v)) for v in active_mask)

        training_cases_by_seed[seed] = training_cases_by_seed.get(seed, 0) + 1
        training_cases_by_block[block] = training_cases_by_block.get(block, 0) + 1
        training_cases_by_active_user_count[active_count] = (
            training_cases_by_active_user_count.get(active_count, 0) + 1
        )
        training_cases_by_active_mask[mask_key] = training_cases_by_active_mask.get(mask_key, 0) + 1
        total_n_level_evaluations += int(len(n_target_levels))

        for k in active_users:
            active_user_cases_per_user[int(k)] += 1

        for n_targets in n_target_levels:
            for k in active_users:
                n_val = int(n_targets[int(k)])
                global_active_user_n_evaluations_by_n_kl[n_val] = (
                    global_active_user_n_evaluations_by_n_kl.get(n_val, 0) + 1
                )
                per_user_active_user_n_evaluations_by_n_kl[int(k)][n_val] = (
                    per_user_active_user_n_evaluations_by_n_kl[int(k)].get(n_val, 0) + 1
                )
                total_active_user_n_evaluations += 1

    return {
        "total_training_cases": int(len(training_cases)),
        "total_n_level_evaluations": int(total_n_level_evaluations),
        "total_active_user_n_evaluations": int(total_active_user_n_evaluations),
        "training_cases_by_seed": _serialize_counts(training_cases_by_seed),
        "training_cases_by_block": _serialize_counts(training_cases_by_block),
        "training_cases_by_active_user_count": _serialize_counts(training_cases_by_active_user_count),
        "training_cases_by_active_mask": {
            str(key): int(value) for key, value in sorted(training_cases_by_active_mask.items())
        },
        "active_user_cases_per_user": [int(v) for v in active_user_cases_per_user],
        "global_active_user_n_evaluations_by_n_kl": _serialize_counts(global_active_user_n_evaluations_by_n_kl),
        "per_user_active_user_n_evaluations_by_n_kl": [
            _serialize_counts(user_counts)
            for user_counts in per_user_active_user_n_evaluations_by_n_kl
        ],
    }


def _shared_beam_forward_pass(
    system_params: dict[str, Any],
    training_case: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> dict[str, Any]:
    K = int(system_params["K"])
    active_mask = np.asarray(training_case["active_mask"], dtype=np.float32)
    active_mask_t = torch.tensor(active_mask, dtype=torch.float32, device=DEVICE)
    H_block_t = [
        torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=DEVICE)
        for H_kl in training_case["H_block"]
    ]
    predicted_beams: list[torch.Tensor] = []
    joint_noise_covariances: list[torch.Tensor | None] = [None for _ in range(K)]

    for k in range(K):
        if float(active_mask[k]) <= 0.5:
            predicted_beams.append(
                torch.zeros(
                    (int(system_params["Nb"][k]), int(system_params["dk"][k])),
                    dtype=torch.complex64,
                    device=DEVICE,
                )
            )
            continue

        noise_cov_input_t = torch.tensor(
            np.asarray(training_case["input_noise_covariances"][k]),
            dtype=torch.complex64,
            device=DEVICE,
        )
        predicted_beams.append(
            infer_precoder_torch_with_context(
                user_models[k],
                H_block_t,
                active_mask_t,
                noise_cov_input_t,
                float(training_case["epsilon"][k]),
                int(system_params["Nb"][k]),
                int(system_params["dk"][k]),
                float(training_case["P"][k]),
            )
        )

    for k in range(K):
        if float(active_mask[k]) <= 0.5:
            continue
        joint_noise_covariances[k] = _joint_noise_covariance_torch(
            H_block_t,
            predicted_beams,
            float(training_case["sigma2"][k]),
            k,
            active_mask,
        )

    return {
        "active_mask": active_mask,
        "H_block_t": H_block_t,
        "predicted_beams": predicted_beams,
        "joint_noise_covariances": joint_noise_covariances,
    }


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
    training_cases: list[dict[str, Any]] = []

    for seed in train_seeds:
        if verbose:
            print(f"\n================ DOWNLINK RAW TRAINING DATA seed={int(seed)} ================")
        configure_determinism(int(seed))
        system = DownlinkSystem(system_params, seed=int(seed))
        for block in block_ids:
            for k in range(K):
                system.ensure_block(k, int(block))
            working_F = system.clone_precoders()
            H_block = _context_channels_for_block(system, int(block))

            for active_mask in active_masks:
                active_users = [int(k) for k in range(K) if float(active_mask[int(k)]) > 0.5]
                if len(active_users) == 0:
                    continue

                input_snapshot = _masked_precoder_snapshot(system, working_F, int(block), active_mask)
                n_target_levels = _build_training_n_target_levels(system, active_mask, sim_params)
                if len(n_target_levels) == 0:
                    continue

                training_cases.append(
                    {
                        "seed": int(seed),
                        "block": int(block),
                        "H_block": [np.asarray(H_kl, dtype=np.complex64) for H_kl in H_block],
                        "active_mask": [int(v > 0.5) for v in active_mask.tolist()],
                        "active_users": [int(k) for k in active_users],
                        "n_target_levels": [[int(v) for v in n_targets] for n_targets in n_target_levels],
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

    return training_cases


def train_shared_beam_precoder_net(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_cases: Sequence[dict[str, Any]],
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
        build_user_precoder_net_with_context(
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
    dataset_summary = summarize_training_dataset(training_cases)
    training_history = {
        "per_user_rate": [[] for _ in range(K)],
        "sum_rate": [],
        "avg_user_rate": [],
        "avg_loss": [],
        "dataset_summary": dataset_summary,
        "training_objective": "average_shared_beam_sum_fbl_rate_over_candidate_n_grid",
    }
    dataset_sizes = [int(v) for v in dataset_summary.get("active_user_cases_per_user", [0 for _ in range(K)])]

    if verbose:
        print(
            "\n================ DOWNLINK SHARED-BEAM PRECODER NET TRAIN ================\n"
            f"Training cases: {len(training_cases)} | epochs: {int(epochs)} | batch_size: {int(batch_size)}\n"
            f"Active user-cases per user: {dataset_sizes}"
        )

    if len(training_cases) == 0:
        training_history["post_training_summary"] = {
            "epochs_requested": int(epochs),
            "final_avg_loss": 0.0,
            "best_avg_loss": 0.0,
            "final_avg_sum_rate": 0.0,
            "best_avg_sum_rate": 0.0,
            "final_avg_user_rate": 0.0,
            "best_avg_user_rate": 0.0,
            "per_user_final_rate": [0.0 for _ in range(K)],
            "per_user_best_rate": [0.0 for _ in range(K)],
        }
        return [model.eval() for model in models], training_history, dataset_sizes

    rng = np.random.default_rng(1000)
    indices = np.arange(len(training_cases))

    for epoch in range(int(epochs)):
        for model in models:
            model.train()
        rng.shuffle(indices)
        epoch_sum_rate_total = 0.0
        epoch_sum_rate_count = 0
        epoch_user_rate_total = np.zeros(K, dtype=float)
        epoch_user_rate_count = np.zeros(K, dtype=float)

        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start : start + max(int(batch_size), 1)]
            optimizer.zero_grad()
            batch_loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
            batch_case_count = 0

            for idx in batch_idx:
                training_case = training_cases[int(idx)]
                forward = _shared_beam_forward_pass(system_params, training_case, models)
                case_level_sum_rates: list[torch.Tensor] = []

                for n_targets in training_case["n_target_levels"]:
                    level_sum_rate = torch.zeros((), dtype=torch.float32, device=DEVICE)
                    for k in training_case["active_users"]:
                        rate = _compute_r_fbl_torch(
                            forward["H_block_t"][int(k)],
                            forward["predicted_beams"][int(k)],
                            epsilon=float(training_case["epsilon"][int(k)]),
                            n_kl=int(n_targets[int(k)]),
                            noise_plus_interference_cov=forward["joint_noise_covariances"][int(k)],
                        )
                        level_sum_rate = level_sum_rate + rate
                        epoch_user_rate_total[int(k)] += float(rate.detach().cpu())
                        epoch_user_rate_count[int(k)] += 1.0

                    case_level_sum_rates.append(level_sum_rate)
                    epoch_sum_rate_total += float(level_sum_rate.detach().cpu())
                    epoch_sum_rate_count += 1

                if len(case_level_sum_rates) == 0:
                    continue
                case_avg_sum_rate = torch.stack(case_level_sum_rates).mean()
                batch_loss = batch_loss - case_avg_sum_rate
                batch_case_count += 1

            if batch_case_count <= 0:
                continue

            batch_loss = batch_loss / float(batch_case_count)
            batch_loss.backward()
            for model in models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for model in models:
            model.eval()

        avg_sum_rate = float(epoch_sum_rate_total / max(epoch_sum_rate_count, 1))
        per_user_rates = [
            float(epoch_user_rate_total[k] / max(epoch_user_rate_count[k], 1.0))
            for k in range(K)
        ]
        avg_user_rate = float(np.mean(per_user_rates)) if len(per_user_rates) > 0 else 0.0
        avg_loss = -avg_sum_rate

        training_history["sum_rate"].append(avg_sum_rate)
        training_history["avg_user_rate"].append(avg_user_rate)
        training_history["avg_loss"].append(avg_loss)
        for k in range(K):
            training_history["per_user_rate"][k].append(float(per_user_rates[k]))

        if verbose:
            print(
                f"Shared-beam joint epoch {epoch + 1}/{int(epochs)}: "
                f"loss={avg_loss:.6e} | avg_sum_rate={avg_sum_rate:.6f} | "
                f"avg_user_rate={avg_user_rate:.6f} | per_user_rate={per_user_rates}"
            )

    training_history["post_training_summary"] = {
        "epochs_requested": int(epochs),
        "final_avg_loss": float(training_history["avg_loss"][-1]) if training_history["avg_loss"] else 0.0,
        "best_avg_loss": float(min(training_history["avg_loss"])) if training_history["avg_loss"] else 0.0,
        "final_avg_sum_rate": float(training_history["sum_rate"][-1]) if training_history["sum_rate"] else 0.0,
        "best_avg_sum_rate": float(max(training_history["sum_rate"])) if training_history["sum_rate"] else 0.0,
        "final_avg_user_rate": (
            float(training_history["avg_user_rate"][-1]) if training_history["avg_user_rate"] else 0.0
        ),
        "best_avg_user_rate": (
            float(max(training_history["avg_user_rate"])) if training_history["avg_user_rate"] else 0.0
        ),
        "per_user_final_rate": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_rate"]
        ],
        "per_user_best_rate": [
            float(max(history)) if len(history) > 0 else 0.0 for history in training_history["per_user_rate"]
        ],
    }

    return [model.eval() for model in models], training_history, dataset_sizes


def _predict_shared_beams_for_active_users(
    system: DownlinkSystem,
    user_models: Sequence[torch.nn.Module],
    working_F: list[list[np.ndarray]],
    block: int,
    transmit_users: Sequence[int],
) -> list[list[np.ndarray]]:
    active_mask = [1 if int(k) in {int(u) for u in transmit_users} else 0 for k in range(system.K)]
    input_snapshot = _masked_precoder_snapshot(system, working_F, int(block), active_mask)
    H_block = _context_channels_for_block(system, int(block))

    for k in range(system.K):
        if int(k) not in {int(u) for u in transmit_users}:
            _zero_block_precoder(system, working_F, int(k), int(block))
            continue
        input_noise_cov = system.get_interference_plus_noise_covariance(int(k), int(block), F_override=input_snapshot)
        working_F[int(k)][int(block)] = infer_precoder_numpy_with_context(
            user_models[int(k)],
            H_block,
            active_mask,
            np.asarray(input_noise_cov, dtype=np.complex128),
            float(system.epsilon[int(k)]),
            nb=int(system.Nb[int(k)]),
            dk=int(system.dk[int(k)]),
            power_limit=float(system.P[int(k)]),
            device=DEVICE,
        )
    return working_F


def _allocate_bits_for_user_block_shared_beam(
    system: DownlinkSystem,
    fixed_snapshot: list[list[np.ndarray]],
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    *,
    allow_infeasible_zero: bool = False,
) -> tuple[int, int, float]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])

    R_T = float(system.compute_block_rate(k, l, T_k, F_override=fixed_snapshot))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    if B_max <= 0:
        if allow_infeasible_zero:
            return 0, T_k, R_T
        raise RuntimeError(
            f"Shared-beam user {k} block {l} infeasible at n=T={T_k}; R_T={R_T:.6f}, B_max={B_max}."
        )

    B_used = int(min(int(remaining_bits), B_max))
    chosen_n = int(T_k)
    chosen_R = float(R_T)
    if int(remaining_bits) <= B_max:
        candidate = T_k - n_step
        while candidate >= n_min:
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=fixed_snapshot))
            if (float(B_used) / float(candidate)) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                candidate -= n_step
            else:
                break

    return int(B_used), int(chosen_n), float(chosen_R)


def evaluate_shared_beam_precoder_net(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    *,
    verbose: bool = True,
    method_name: str = "monte_carlo_shared_beam_train_test",
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

    block = 0
    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Shared-beam evaluation hit max_total_blocks={max_blocks} with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(system.K) if int(remaining[k]) > 0]
        for k in active_users:
            _ensure_user_block(system, working_F, int(k), int(block))

        if verbose:
            print(
                f"\n=== Shared-beam block {block} | active_users={len(active_users)} | "
                f"remaining_bits={int(np.sum(remaining))} ==="
            )

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        forward_pass_idx = 0
        while len(transmit_users) > 0:
            forward_pass_idx += 1
            working_F = _predict_shared_beams_for_active_users(
                system,
                user_models,
                working_F,
                int(block),
                transmit_users,
            )
            current_eval = _evaluate_block_candidate(system, working_F, transmit_users, int(block))
            user_rates = []
            user_sinr_db = []
            user_interference_db = []
            user_signal_db = []
            for k in transmit_users:
                rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=working_F))
                signal_power, interference_power, _, sinr_db = _compute_user_link_budget(
                    system,
                    working_F,
                    int(k),
                    int(block),
                )
                user_rates.append(rate)
                user_sinr_db.append(float(sinr_db))
                user_interference_db.append(_power_to_db(interference_power))
                user_signal_db.append(_power_to_db(signal_power))

            sweep_history.append(
                {
                    "block": int(block),
                    "sweep": int(forward_pass_idx),
                    "active_users": int(len(transmit_users)),
                    "user_ids": [int(k) for k in transmit_users],
                    "user_rates": user_rates,
                    "user_sinr_db": user_sinr_db,
                    "user_interference_db": user_interference_db,
                    "user_signal_db": user_signal_db,
                    "user_weights": [1.0 for _ in transmit_users],
                    "max_precoder_delta": 0.0,
                    "sum_rate": float(sum(user_rates)),
                    "weighted_sum_rate": float(sum(user_rates)),
                    "blended_objective": float(sum(user_rates)),
                    "objective_mode": "shared_beam_forward_pass",
                }
            )

            infeasible_users = [
                int(user_id)
                for user_id, max_bits in zip(current_eval["user_ids"], current_eval["user_max_bits"])
                if int(max_bits) <= 0
            ]
            if len(infeasible_users) == 0:
                break
            for k in infeasible_users:
                _zero_block_precoder(system, working_F, int(k), int(block))
                skipped_users.append(int(k))
            transmit_users = [k for k in transmit_users if int(k) not in infeasible_users]
            if verbose:
                print(f"  shared-beam block={block:02d} skipping users {infeasible_users}")

        allocation_snapshot = _clone_precoders(working_F)
        block_bits = 0
        for k in active_users:
            queue_weight = 1.0
            if int(k) in skipped_users or int(k) not in transmit_users:
                _zero_block_precoder(system, working_F, int(k), int(block))
                skipped_rate = float(
                    system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot)
                )
                B_plan[int(k)].append(0)
                n_plan[int(k)].append(int(system.T[int(k)]))
                R_plan[int(k)].append(float(skipped_rate))
                rate_points.append(
                    {
                        "user": int(k),
                        "block": int(block),
                        "n_kl": int(system.T[int(k)]),
                        "B_kl": 0,
                        "required_rate": 0.0,
                        "achieved_rate": float(skipped_rate),
                        "rate_margin": float(skipped_rate),
                        "queue_weight": float(queue_weight),
                        "skipped": True,
                    }
                )
                if verbose:
                    print(
                        f"  user={int(k):02d} block={int(block):02d} skipped "
                        f"n_kl={int(system.T[int(k)]):4d} R_fbl={skipped_rate:.4f}"
                    )
                continue

            B_used, n_used, R_used = _allocate_bits_for_user_block_shared_beam(
                system,
                allocation_snapshot,
                int(k),
                int(block),
                int(remaining[int(k)]),
                sim_params,
                allow_infeasible_zero=True,
            )
            B_plan[int(k)].append(int(B_used))
            n_plan[int(k)].append(int(n_used))
            R_plan[int(k)].append(float(R_used))
            remaining[int(k)] -= int(B_used)
            block_bits += int(B_used)
            required_rate = float(B_used) / float(max(int(n_used), 1))
            rate_margin = float(R_used) - required_rate
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": rate_margin,
                    "queue_weight": float(queue_weight),
                    "skipped": bool(B_used <= 0),
                }
            )
            if verbose:
                print(
                    f"  user={int(k):02d} block={int(block):02d} "
                    f"bits={int(B_used):4d} n_kl={int(n_used):4d} "
                    f"required_rate={required_rate:.4f} R_fbl={float(R_used):.4f}"
                )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(len(transmit_users)),
                "skipped_users": int(len(skipped_users)),
                "allocated_bits": int(block_bits),
                "remaining_bits": int(np.sum(remaining)),
                "feasible_users": int(len(transmit_users)),
                "min_max_bits": int(min([row["B_kl"] for row in rate_points if int(row["block"]) == int(block)], default=0)),
                "queue_weights": {int(k): 1.0 for k in active_users},
                "final_precoder_delta": 0.0,
            }
        )
        if verbose:
            print(
                f"--- Shared-beam block {int(block)} complete | "
                f"allocated_bits={int(block_bits)} remaining_bits={int(np.sum(remaining))} ---"
            )
        block += 1

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)

    result = {
        "method_name": method_name,
        "objective_mode": "shared_beam_sum_rate",
        "allocation_mode": "greedy",
        "weight_strategy": "remaining_bits",
        "precoder_parameterization": "shared_user_block_context_noise_epsilon_to_shared_beam_mlp",
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=False,
            input_mode="block_context_noise_epsilon",
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
        ),
        "n_kl": [list(map(int, values)) for values in n_plan],
        "B_kl": [list(map(int, values)) for values in B_plan],
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": [list(map(float, values)) for values in R_plan],
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
        "blocks_per_user": [len(values) for values in n_plan],
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
            uses_blocklength_input=False,
            input_mode="block_context_noise_epsilon",
            context_k=int(system_params["K"]),
            context_max_nr=int(np.max(system_params["Nr"])),
            context_max_nb=int(np.max(system_params["Nb"])),
        ),
        "user_model_states": export_user_model_states(user_models),
        "precoder_parameterization": "shared_user_block_context_noise_epsilon_to_shared_beam_mlp",
        "training_objective": "average_shared_beam_sum_fbl_rate_over_candidate_n_grid",
    }


train_shared_beam_precoder = train_shared_beam_precoder_net
evaluate_downlink_shared_beam = evaluate_shared_beam_precoder_net
build_shared_beam_artifact = build_precoder_net_artifact
