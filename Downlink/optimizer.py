from __future__ import annotations

import copy
import random
from typing import Any, List

import numpy as np
import torch

from downlink_system import DownlinkSystem
from precoder_models import (
    DEVICE,
    build_user_precoder_net,
    export_user_model_specs,
    infer_precoder_numpy,
    infer_precoder_torch,
)
LOG2E_SQ = float(np.log2(np.e) ** 2)
SAFE_SWEEP_OBJECTIVE_MODES = {
    "user_rate",
    "weighted_sum_rate",
    "blended_network_rate",
}


def resolve_safe_sweep_objective_mode(sim_params: dict[str, Any]) -> str:
    raw_mode = str(
        sim_params.get(
            "safe_sweep_objective_mode",
            sim_params.get(
                "downlink_safe_sweep_objective_mode",
                sim_params.get("objective_mode", "user_rate"),
            ),
        )
    ).strip().lower()
    if raw_mode not in SAFE_SWEEP_OBJECTIVE_MODES:
        known = ", ".join(sorted(SAFE_SWEEP_OBJECTIVE_MODES))
        raise ValueError(
            f"Unknown safe-sweep objective mode '{raw_mode}'. Expected one of: {known}"
        )
    return raw_mode


def safe_sweep_objective_tag(objective_mode: str) -> str:
    safe_mode = str(objective_mode).strip().lower().replace(" ", "_").replace("-", "_")
    return f"obj_{safe_mode}"


def _clone_precoders(F_nested: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    return [[np.array(F_kl, copy=True) for F_kl in user_F] for user_F in F_nested]


def _build_user_precoder_models(
    system: DownlinkSystem,
    *,
    init_seed: int | None = None,
) -> list[torch.nn.Module]:
    def _construct_models() -> list[torch.nn.Module]:
        models: list[torch.nn.Module] = []
        for k in range(int(system.K)):
            model = build_user_precoder_net(
                nr=int(system.Nr[k]),
                nb=int(system.Nb[k]),
                dk=int(system.dk[k]),
                device=DEVICE,
            )
            models.append(model)
        return models

    if init_seed is None:
        return _construct_models()

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    local_seed = int(init_seed)
    np_seed = int(local_seed % (2**32 - 1))

    try:
        random.seed(local_seed)
        np.random.seed(np_seed)
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)
        return _construct_models()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _build_precoder_snapshot_from_models(
    system: DownlinkSystem,
    user_models: list[torch.nn.Module],
) -> List[List[np.ndarray]]:
    snapshot: List[List[np.ndarray]] = []
    for k in range(int(system.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(len(system.H[k])):
            user_blocks.append(
                infer_precoder_numpy(
                    user_models[k],
                    np.asarray(system.H[k][l], dtype=np.complex64),
                    nb=int(system.Nb[k]),
                    dk=int(system.dk[k]),
                    power_limit=float(system.P[k]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)
    return snapshot


def _copy_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _complex_to_param(F_mat: np.ndarray) -> torch.nn.Parameter:
    arr = np.asarray(F_mat, dtype=np.complex64)
    stacked = np.stack([arr.real, arr.imag], axis=0)
    return torch.nn.Parameter(torch.tensor(stacked, dtype=torch.float32, device=DEVICE))


def _param_to_complex(param: torch.Tensor) -> torch.Tensor:
    return (param[0] + 1j * param[1]).to(torch.complex64)


def _project_power(F_mat: torch.Tensor, power_limit: float, eps: float = 1e-12) -> torch.Tensor:
    power = (torch.linalg.norm(F_mat, ord="fro") ** 2).real
    if float(power.detach().cpu()) <= float(power_limit):
        return F_mat
    scale = torch.sqrt(torch.tensor(float(power_limit), device=F_mat.device, dtype=torch.float32) / (power + eps))
    return F_mat * scale.to(F_mat.dtype)


def _q_inv(epsilon: float) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64, device=DEVICE),
        torch.tensor(1.0, dtype=torch.float64, device=DEVICE),
    )
    p = torch.tensor(1.0 - float(epsilon), dtype=torch.float64, device=DEVICE)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return normal.icdf(p).to(dtype=torch.float32)


def _block_rate_torch_with_override(
    system: DownlinkSystem,
    user: int,
    block: int,
    n_kl: int,
    F_snapshot: List[List[np.ndarray]],
    override_user: int | None = None,
    override_precoder: torch.Tensor | None = None,
) -> torch.Tensor:
    k = int(user)
    l = int(block)
    if l >= len(system.H[k]):
        raise ValueError(f"User {k} has no channel block {l}")

    Hk = torch.tensor(system.H[k][l], dtype=torch.complex64, device=DEVICE)
    Nrk = int(system.Nr[k])
    I = torch.eye(Nrk, dtype=torch.complex64, device=DEVICE)

    if override_user is not None and k == int(override_user):
        Fk = override_precoder
    else:
        Fk = torch.tensor(F_snapshot[k][l], dtype=torch.complex64, device=DEVICE)

    noise_cov = float(system.sigma2[k]) * I
    for j in range(system.K):
        if j == k:
            continue
        if override_user is not None and j == int(override_user):
            Fj = override_precoder
        else:
            if l >= len(F_snapshot[j]):
                continue
            Fj = torch.tensor(F_snapshot[j][l], dtype=torch.complex64, device=DEVICE)
        HFj = Hk @ Fj
        noise_cov = noise_cov + HFj @ HFj.conj().transpose(1, 0)

    noise_cov = 0.5 * (noise_cov + noise_cov.conj().transpose(1, 0))
    noise_cov = noise_cov + (1e-6 * I)

    HF = Hk @ Fk
    chol = torch.linalg.cholesky(noise_cov)
    G = torch.linalg.solve(chol, HF)
    A = G @ G.conj().transpose(1, 0)
    A = 0.5 * (A + A.conj().transpose(1, 0))

    sign, logdet = torch.linalg.slogdet(I + A)
    if torch.any(torch.abs(sign) <= 1e-12):
        raise RuntimeError(f"Non-positive logdet sign for user {k}, block {l}")

    C = (logdet / np.log(2.0)).real
    eigvals = torch.linalg.eigvalsh(A)
    V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2).real * LOG2E_SQ
    R = C - torch.sqrt(V / float(max(int(n_kl), 1))) * _q_inv(float(system.epsilon[k]))
    return R.real


def _evaluate_update_objective_numpy(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    focus_user: int,
    block: int,
    objective_mode: str,
    user_weights: dict[int, float],
    network_weight_beta: float,
) -> float:
    if objective_mode == "user_rate":
        return float(system.compute_block_rate(int(focus_user), int(block), int(system.T[int(focus_user)]), F_override=working_F))

    if objective_mode == "blended_network_rate":
        self_rate = float(system.compute_block_rate(int(focus_user), int(block), int(system.T[int(focus_user)]), F_override=working_F))
        others_total = 0.0
        for k in active_users:
            if int(k) == int(focus_user):
                continue
            others_total += float(user_weights.get(int(k), 1.0)) * float(
                system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=working_F)
            )
        return self_rate + float(network_weight_beta) * others_total

    total = 0.0
    for k in active_users:
        total += float(user_weights.get(int(k), 1.0)) * float(
            system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=working_F)
        )
    return total


def _expand_precoders_for_plan(
    system: DownlinkSystem,
    base_F: List[List[np.ndarray]],
    n_kl_plan: List[List[int]],
) -> List[List[np.ndarray]]:
    expanded = _clone_precoders(base_F)
    for k in range(system.K):
        for l in range(len(n_kl_plan[k])):
            if l >= len(expanded[k]):
                system.ensure_block(k, l)
                expanded[k].append(np.array(system.F[k][l], copy=True))
    return expanded


def _rate_to_max_bits(n_kl: int, rate: float) -> int:
    return int(np.floor(float(n_kl) * float(rate)))


def _ensure_user_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
    use_previous_as_template: bool = True,
) -> None:
    k = int(user)
    l = int(block)
    if l < len(working_F[k]):
        return

    template = None
    if use_previous_as_template and len(working_F[k]) > 0:
        prev = np.asarray(working_F[k][-1], dtype=np.complex128)
        if float(np.linalg.norm(prev, ord="fro")) > 1e-12:
            template = prev
    system.ensure_block(k, l, template_precoder=template)
    while len(working_F[k]) <= l:
        working_F[k].append(np.array(system.F[k][len(working_F[k])], copy=True))


def _zero_block_precoder(system: DownlinkSystem, working_F: List[List[np.ndarray]], user: int, block: int) -> None:
    k = int(user)
    l = int(block)
    working_F[k][l] = np.zeros((int(system.Nb[k]), int(system.dk[k])), dtype=np.complex128)


def _effective_cnr_linear(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
) -> float:
    signal_power, _, noise_power, _ = _compute_user_link_budget(system, working_F, int(user), int(block))
    return float(signal_power / max(noise_power, 1e-30))


def _channel_gain_per_rx(
    system: DownlinkSystem,
    user: int,
    block: int,
) -> float:
    Hk = np.asarray(system.H[int(user)][int(block)], dtype=np.complex128)
    return float(np.linalg.norm(Hk, ord="fro") ** 2 / max(1, int(system.Nr[int(user)])))


def _build_user_weights(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    remaining_bits: np.ndarray,
    active_users: List[int],
    block: int,
    sim_params: dict[str, Any],
    strategy: str,
) -> dict[int, float]:
    if len(active_users) == 0:
        return {}

    power = float(sim_params.get("queue_weight_power", 1.0))
    min_weight = float(sim_params.get("queue_weight_min", 0.25))
    raw_scores: dict[int, float] = {}

    if strategy == "inverse_cnr":
        for k in active_users:
            cnr_lin = _effective_cnr_linear(system, working_F, int(k), int(block))
            raw_scores[int(k)] = 1.0 / max(cnr_lin, 1e-30)
    elif strategy == "inverse_channel_gain":
        for k in active_users:
            channel_gain = _channel_gain_per_rx(system, int(k), int(block))
            raw_scores[int(k)] = 1.0 / max(channel_gain, 1e-30)
    else:
        max_remaining = max(int(remaining_bits[int(k)]) for k in active_users)
        denom_remaining = max(float(max_remaining), 1.0)
        for k in active_users:
            raw_scores[int(k)] = float(remaining_bits[int(k)]) / denom_remaining

    max_score = max(raw_scores.values()) if raw_scores else 1.0
    denom = max(float(max_score), 1e-30)

    weights: dict[int, float] = {}
    for k in active_users:
        normalized = (float(raw_scores[int(k)]) / denom) ** power
        weights[int(k)] = min_weight + (1.0 - min_weight) * normalized
    return weights


def _optimize_user_block_precoder(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    user_weights: dict[int, float],
    user: int,
    block: int,
    sim_params: dict[str, Any],
    objective_mode: str,
    precoder_model: torch.nn.Module,
    model_optimizer: torch.optim.Optimizer,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    n_kl = int(system.T[k])
    steps = max(1, int(sim_params.get("user_update_steps", 1)))
    network_weight_beta = float(sim_params.get("network_weight_beta", 0.15))
    H_kl = torch.tensor(system.H[k][l], dtype=torch.complex64, device=DEVICE)

    best_beam = np.array(working_F[k][l], copy=True)
    best_model_state = _copy_model_state(precoder_model)
    best_snapshot = _clone_precoders(working_F)
    best_snapshot[k][l] = best_beam
    best_objective = _evaluate_update_objective_numpy(
        system,
        best_snapshot,
        active_users,
        k,
        l,
        objective_mode,
        user_weights,
        network_weight_beta,
    )

    for _ in range(steps):
        model_optimizer.zero_grad()
        F_candidate = infer_precoder_torch(
            precoder_model,
            H_kl,
            nb=int(system.Nb[k]),
            dk=int(system.dk[k]),
            power_limit=float(system.P[k]),
        )
        if objective_mode in {"weighted_sum_rate", "blended_network_rate"}:
            objective = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
            for j in active_users:
                rate_j = _block_rate_torch_with_override(
                    system,
                    int(j),
                    l,
                    int(system.T[int(j)]),
                    working_F,
                    override_user=k,
                    override_precoder=F_candidate,
                )
                if objective_mode == "blended_network_rate" and int(j) == k:
                    objective = objective + rate_j
                else:
                    scale = float(user_weights.get(int(j), 1.0))
                    if objective_mode == "blended_network_rate":
                        scale *= network_weight_beta
                    objective = objective + scale * rate_j
        else:
            objective = _block_rate_torch_with_override(
                system,
                k,
                l,
                n_kl,
                working_F,
                override_user=k,
                override_precoder=F_candidate,
            )
        (-objective).backward()
        model_optimizer.step()

        beam_np = infer_precoder_numpy(
            precoder_model,
            np.asarray(system.H[k][l], dtype=np.complex64),
            nb=int(system.Nb[k]),
            dk=int(system.dk[k]),
            power_limit=float(system.P[k]),
            device=DEVICE,
        )
        beam_snapshot = _clone_precoders(working_F)
        beam_snapshot[k][l] = beam_np
        objective_np = _evaluate_update_objective_numpy(
            system,
            beam_snapshot,
            active_users,
            k,
            l,
            objective_mode,
            user_weights,
            network_weight_beta,
        )
        if objective_np > best_objective:
            best_beam = beam_np
            best_objective = objective_np
            best_model_state = _copy_model_state(precoder_model)

    precoder_model.load_state_dict(best_model_state)

    return best_beam


def _block_delta(
    before_beams: dict[int, np.ndarray],
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
) -> float:
    deltas = []
    for k in active_users:
        prev = np.asarray(before_beams[k], dtype=np.complex128)
        curr = np.asarray(working_F[k][block], dtype=np.complex128)
        denom = max(float(np.linalg.norm(prev, ord="fro")), 1e-12)
        deltas.append(float(np.linalg.norm(curr - prev, ord="fro") / denom))
    return max(deltas) if deltas else 0.0


def _compute_user_link_budget(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
) -> tuple[float, float, float, float]:
    k = int(user)
    l = int(block)
    Hk = np.asarray(system.H[k][l], dtype=np.complex128)
    Fk = np.asarray(working_F[k][l], dtype=np.complex128)
    signal_power = float(np.linalg.norm(Hk @ Fk, ord="fro") ** 2 / max(1, int(system.Nr[k])))
    interference_power = 0.0
    for j in range(system.K):
        if j == k or l >= len(working_F[j]):
            continue
        Fj = np.asarray(working_F[j][l], dtype=np.complex128)
        interference_power += float(np.linalg.norm(Hk @ Fj, ord="fro") ** 2 / max(1, int(system.Nr[k])))
    noise_power = float(system.sigma2[k])
    sinr_db = 10.0 * np.log10(max(signal_power / max(interference_power + noise_power, 1e-30), 1e-30))
    return signal_power, interference_power, noise_power, sinr_db


def _power_to_db(power: float) -> float:
    return float(10.0 * np.log10(max(float(power), 1e-30)))


def _collect_interference_diagnostics(system: DownlinkSystem) -> dict[str, Any]:
    K = int(system.K)
    max_blocks = max((len(v) for v in system.n_kl), default=0)
    signal = np.full((K, max_blocks), np.nan, dtype=float)
    total_interference = np.full((K, max_blocks), np.nan, dtype=float)
    noise = np.full((K, max_blocks), np.nan, dtype=float)
    sinr_db = np.full((K, max_blocks), np.nan, dtype=float)
    pairwise_block = np.full((max_blocks, K, K), np.nan, dtype=float)
    pairwise_sum = np.zeros((K, K), dtype=float)
    pairwise_inr_sum = np.zeros((K, K), dtype=float)
    pairwise_count = np.zeros((K, K), dtype=float)

    for k in range(K):
        for l in range(len(system.n_kl[k])):
            Hk = np.asarray(system.H[k][l], dtype=np.complex128)
            Fk = np.asarray(system.F[k][l], dtype=np.complex128)
            signal_power = float(np.linalg.norm(Hk @ Fk, ord="fro") ** 2 / max(1, int(system.Nr[k])))
            noise_power = float(system.sigma2[k])
            interference_power = 0.0

            for j in range(K):
                if j == k or l >= len(system.F[j]):
                    continue
                Fj = np.asarray(system.F[j][l], dtype=np.complex128)
                coupling = float(np.linalg.norm(Hk @ Fj, ord="fro") ** 2 / max(1, int(system.Nr[k])))
                pairwise_block[l, k, j] = coupling
                pairwise_sum[k, j] += coupling
                pairwise_inr_sum[k, j] += coupling / max(noise_power, 1e-30)
                pairwise_count[k, j] += 1.0
                interference_power += coupling

            signal[k, l] = signal_power
            total_interference[k, l] = interference_power
            noise[k, l] = noise_power
            sinr_db[k, l] = float(
                10.0 * np.log10(max(signal_power / max(interference_power + noise_power, 1e-30), 1e-30))
            )

    avg_pairwise_power = np.divide(
        pairwise_sum,
        pairwise_count,
        out=np.full_like(pairwise_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_db = 10.0 * np.log10(
        np.maximum(
            np.divide(
                pairwise_inr_sum,
                pairwise_count,
                out=np.full_like(pairwise_inr_sum, np.nan),
                where=pairwise_count > 0,
            ),
            1e-30,
        )
    )
    row_sum = np.nansum(avg_pairwise_power, axis=1, keepdims=True)
    avg_pairwise_share = np.divide(
        avg_pairwise_power,
        row_sum,
        out=np.full_like(avg_pairwise_power, np.nan),
        where=row_sum > 0,
    )
    block_totals = np.nansum(pairwise_block, axis=(1, 2)) if max_blocks > 0 else np.asarray([], dtype=float)
    worst_block = int(np.nanargmax(block_totals)) if block_totals.size > 0 else -1

    return {
        "blocks_per_user": [len(v) for v in system.n_kl],
        "signal": signal.tolist(),
        "total_interference": total_interference.tolist(),
        "noise": noise.tolist(),
        "sinr_db": sinr_db.tolist(),
        "pairwise_block": pairwise_block.tolist(),
        "avg_pairwise_power": avg_pairwise_power.tolist(),
        "avg_pairwise_inr_db": avg_pairwise_inr_db.tolist(),
        "avg_pairwise_share": avg_pairwise_share.tolist(),
        "worst_block": int(worst_block),
    }


def _evaluate_block_candidate(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
) -> dict[str, Any]:
    rates = []
    max_bits = []
    for k in active_users:
        rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=working_F))
        bits = max(_rate_to_max_bits(int(system.T[int(k)]), rate), 0)
        rates.append(rate)
        max_bits.append(int(bits))

    feasible_count = int(sum(bits > 0 for bits in max_bits))
    return {
        "user_ids": [int(k) for k in active_users],
        "user_rates": rates,
        "user_max_bits": max_bits,
        "feasible_count": feasible_count,
        "min_max_bits": int(min(max_bits)) if len(max_bits) > 0 else 0,
        "min_rate": float(min(rates)) if len(rates) > 0 else 0.0,
        "sum_rate": float(sum(rates)),
    }


def optimize_precoders_for_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    sim_params: dict[str, Any],
    verbose: bool = True,
    objective_mode: str = "user_rate",
    user_weights: dict[int, float] | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if len(active_users) == 0:
        return history

    weights = user_weights or {int(k): 1.0 for k in active_users}
    print_every = max(1, int(sim_params.get("print_every_sweep", 1)))
    tol = float(sim_params["precoder_tol"])
    if objective_mode == "weighted_sum_rate":
        objective_label = "weighted_sum_rate"
    elif objective_mode == "blended_network_rate":
        objective_label = "blended_objective"
    else:
        objective_label = "sum_rate"
    network_weight_beta = float(sim_params.get("network_weight_beta", 0.15))

    for sweep_idx in range(int(sim_params["max_precoder_sweeps"])):
        before_beams = {k: np.array(working_F[k][block], copy=True) for k in active_users}

        for k in active_users:
            beam_k = _optimize_user_block_precoder(
                system,
                working_F,
                active_users,
                weights,
                k,
                block,
                sim_params,
                objective_mode,
                user_models[int(k)],
                model_optimizers[int(k)],
            )
            working_F[k][block] = np.array(beam_k, copy=True)

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=working_F))
            signal_power, interference_power, _, sinr_db = _compute_user_link_budget(system, working_F, int(k), int(block))
            user_rates.append(rate)
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(_power_to_db(interference_power))
            user_signal_db.append(_power_to_db(signal_power))

        total_rate = float(sum(user_rates))
        weighted_total = float(sum(float(weights.get(int(k), 1.0)) * rate for k, rate in zip(active_users, user_rates)))
        blended_total = float(total_rate + network_weight_beta * weighted_total)
        delta = _block_delta(before_beams, working_F, active_users, block)
        history.append(
            {
                "block": int(block),
                "sweep": sweep_idx + 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [float(weights.get(int(k), 1.0)) for k in active_users],
                "max_precoder_delta": float(delta),
                "sum_rate": total_rate,
                "weighted_sum_rate": weighted_total,
                "blended_objective": blended_total,
                "objective_mode": objective_mode,
            }
        )
        if objective_mode == "weighted_sum_rate":
            objective_value = weighted_total
        elif objective_mode == "blended_network_rate":
            objective_value = blended_total
        else:
            objective_value = total_rate
        if verbose and (((sweep_idx + 1) % print_every) == 0 or sweep_idx == 0 or delta <= tol):
            print(
                f"[Block {block:02d} | Sweep {sweep_idx + 1:03d}] "
                f"active_users={len(active_users)} "
                f"{objective_label}={objective_value:.4f} "
                f"sum_rate={total_rate:.4f} "
                f"max_delta={delta:.6e}"
            )
        if delta <= tol:
            break

    return history


def _allocate_bits_for_user_block_greedy(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    allow_infeasible_zero: bool = False,
) -> tuple[int, int, float]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])

    R_T = float(system.compute_block_rate(k, l, T_k, F_override=working_F))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    if B_max <= 0:
        if allow_infeasible_zero:
            return 0, T_k, R_T
        raise RuntimeError(
            f"User {k} block {l} remains infeasible after block-level precoder convergence at n=T={T_k}; "
            f"R_T={R_T:.6f}, B_max={B_max}."
        )

    B_used = int(min(int(remaining_bits), B_max))
    chosen_n = T_k
    chosen_R = R_T
    if int(remaining_bits) <= B_max:
        candidate = T_k - n_step
        while candidate >= n_min:
            R_candidate = float(system.compute_block_rate(k, l, candidate, F_override=working_F))
            if (float(B_used) / float(candidate)) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = R_candidate
                candidate -= n_step
            else:
                break

    return B_used, chosen_n, chosen_R


def _allocate_bits_for_user_block_weighted(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    queue_weight: float,
    allow_infeasible_zero: bool = False,
) -> tuple[int, int, float]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    latency_penalty = float(sim_params.get("utility_latency_penalty", 0.5))

    R_T = float(system.compute_block_rate(k, l, T_k, F_override=working_F))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    if B_max <= 0:
        if allow_infeasible_zero:
            return 0, T_k, R_T
        raise RuntimeError(
            f"User {k} block {l} remains infeasible after block-level precoder convergence at n=T={T_k}; "
            f"R_T={R_T:.6f}, B_max={B_max}."
        )

    best_choice: tuple[float, float, int, int, float] | None = None
    for candidate in range(n_min, T_k + 1, n_step):
        R_candidate = float(system.compute_block_rate(k, l, candidate, F_override=working_F))
        B_candidate = max(_rate_to_max_bits(candidate, R_candidate), 0)
        if B_candidate <= 0:
            continue

        B_used = int(min(int(remaining_bits), B_candidate))
        throughput = float(B_used) / float(max(candidate, 1))
        latency_ratio = float(candidate) / float(max(T_k, 1))
        utility = float(queue_weight) * float(B_used) - latency_penalty * latency_ratio
        choice = (utility, B_used, throughput, -int(candidate), float(R_candidate))
        if best_choice is None or choice > best_choice:
            best_choice = choice

    if best_choice is None:
        if allow_infeasible_zero:
            return 0, T_k, R_T
        raise RuntimeError(
            f"User {k} block {l} remains infeasible after block-level precoder convergence at all n in [{n_min}, {T_k}]."
        )

    _, B_used, _, neg_n, R_used = best_choice
    return int(B_used), int(-neg_n), float(R_used)


def _allocate_bits_for_user_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    allocation_mode: str,
    queue_weight: float = 1.0,
    allow_infeasible_zero: bool = False,
) -> tuple[int, int, float]:
    if allocation_mode == "weighted_utility":
        return _allocate_bits_for_user_block_weighted(
            system,
            working_F,
            user,
            block,
            remaining_bits,
            sim_params,
            queue_weight=queue_weight,
            allow_infeasible_zero=allow_infeasible_zero,
        )
    return _allocate_bits_for_user_block_greedy(
        system,
        working_F,
        user,
        block,
        remaining_bits,
        sim_params,
        allow_infeasible_zero=allow_infeasible_zero,
    )


def estimate_initial_latency_from_random_precoders(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    allocation_mode: str,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline_system = DownlinkSystem(system.sc, seed=system.seed)
    # Keep the initial random-precoder baseline tied only to the experiment seed,
    # not to whichever RNG state training happened to leave behind.
    baseline_models = _build_user_precoder_models(baseline_system, init_seed=int(system.seed))
    remaining = np.asarray(baseline_system.B, dtype=int).copy()
    n_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    B_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    R_plan: List[List[float]] = [[] for _ in range(baseline_system.K)]
    working_F = _build_precoder_snapshot_from_models(baseline_system, baseline_models)
    max_blocks = int(sim_params.get("max_total_blocks", 256))
    block = 0

    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Initial random-precoder latency estimate hit max_total_blocks={max_blocks} "
                f"with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(baseline_system.K) if int(remaining[k]) > 0]
        for k in active_users:
            _ensure_user_block(baseline_system, working_F, k, block, use_previous_as_template=False)
        working_F = _build_precoder_snapshot_from_models(baseline_system, baseline_models)
        queue_weights = _build_user_weights(
            baseline_system,
            working_F,
            remaining,
            active_users,
            block,
            sim_params,
            strategy="remaining_bits",
        )

        for k in active_users:
            B_used, n_used, R_used = _allocate_bits_for_user_block(
                baseline_system,
                working_F,
                k,
                block,
                int(remaining[k]),
                sim_params,
                allocation_mode=allocation_mode,
                queue_weight=float(queue_weights.get(int(k), 1.0)),
                allow_infeasible_zero=True,
            )
            if B_used <= 0:
                _zero_block_precoder(baseline_system, working_F, k, block)
                n_plan[k].append(int(baseline_system.T[k]))
                B_plan[k].append(0)
                R_plan[k].append(float(R_used))
                continue

            n_plan[k].append(int(n_used))
            B_plan[k].append(int(B_used))
            R_plan[k].append(float(R_used))
            remaining[k] -= int(B_used)

        block += 1

    initial_F = _expand_precoders_for_plan(baseline_system, working_F, n_plan)
    baseline_system.apply_solution(initial_F, n_plan)
    latency = baseline_system.latency.tolist()
    return latency, {"n_kl": n_plan, "B_kl": B_plan, "R_alloc": R_plan}, _collect_interference_diagnostics(baseline_system)


def _run_safe_sweep(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool,
    method_name: str,
    objective_mode: str,
    allocation_mode: str,
    weight_strategy: str = "remaining_bits",
) -> dict[str, Any]:
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = estimate_initial_latency_from_random_precoders(
        system,
        sim_params,
        allocation_mode="greedy",
    )
    user_models = _build_user_precoder_models(system)
    model_optimizers = [
        torch.optim.Adam(model.parameters(), lr=float(sim_params.get("user_update_lr", sim_params["step_lr"])))
        for model in user_models
    ]

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: List[List[int]] = [[] for _ in range(system.K)]
    B_plan: List[List[int]] = [[] for _ in range(system.K)]
    R_plan: List[List[float]] = [[] for _ in range(system.K)]
    working_F: List[List[np.ndarray]] = _build_precoder_snapshot_from_models(system, user_models)
    sweep_history: list[dict[str, float]] = []
    outer_history: list[dict[str, float]] = []
    rate_points: list[dict[str, float]] = []
    max_blocks = int(sim_params.get("max_total_blocks", 256))

    block = 0
    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Reached max_total_blocks={max_blocks} with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(system.K) if int(remaining[k]) > 0]
        for k in active_users:
            _ensure_user_block(system, working_F, k, block)
        working_F = _build_precoder_snapshot_from_models(system, user_models)
        queue_weights = _build_user_weights(
            system,
            working_F,
            remaining,
            active_users,
            block,
            sim_params,
            strategy=weight_strategy,
        )
        if verbose:
            print(
                f"\n=== Optimizing block {block} | active_users={len(active_users)} | "
                f"remaining_bits={int(np.sum(remaining))} ==="
            )
            if objective_mode in {"weighted_sum_rate", "blended_network_rate"}:
                weights_text = ", ".join(f"u{k}={queue_weights[k]:.3f}" for k in active_users)
                print(f"    weight_strategy={weight_strategy} | user_weights: {weights_text}")

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        block_history: list[dict[str, float]] = []
        block_eval: dict[str, Any] = {
            "feasible_count": 0,
            "min_max_bits": 0,
            "user_ids": [],
            "user_max_bits": [],
            "user_rates": [],
            "sum_rate": 0.0,
            "min_rate": 0.0,
        }

        while len(transmit_users) > 0:
            transmit_weights = {int(k): float(queue_weights.get(int(k), 1.0)) for k in transmit_users}
            current_history = optimize_precoders_for_block(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                sim_params,
                verbose=verbose,
                objective_mode=objective_mode,
                user_weights=transmit_weights,
            )
            current_eval = _evaluate_block_candidate(system, working_F, transmit_users, block)
            block_history.extend(current_history)
            block_eval = current_eval
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
                print(
                    f"  block={block:02d} skipping users {infeasible_users}; "
                    "re-optimizing remaining transmitters."
                )
        sweep_history.extend(block_history)

        block_bits = 0
        for k in active_users:
            queue_weight = float(queue_weights.get(int(k), 1.0))
            if int(k) in skipped_users:
                _zero_block_precoder(system, working_F, k, block)
                skipped_rate = float(system.compute_block_rate(k, block, int(system.T[k]), F_override=working_F))
                B_plan[k].append(0)
                n_plan[k].append(int(system.T[k]))
                R_plan[k].append(skipped_rate)
                rate_points.append(
                    {
                        "user": int(k),
                        "block": int(block),
                        "n_kl": int(system.T[k]),
                        "B_kl": 0,
                        "required_rate": 0.0,
                        "achieved_rate": skipped_rate,
                        "rate_margin": skipped_rate,
                        "queue_weight": queue_weight,
                        "skipped": True,
                    }
                )
                if verbose:
                    print(
                        f"  user={k:02d} block={block:02d} skipped "
                        f"n_kl={int(system.T[k]):4d} R_fbl={skipped_rate:.4f}"
                    )
                continue

            B_used, n_used, R_used = _allocate_bits_for_user_block(
                system,
                working_F,
                k,
                block,
                int(remaining[k]),
                sim_params,
                allocation_mode=allocation_mode,
                queue_weight=queue_weight,
            )
            B_plan[k].append(int(B_used))
            n_plan[k].append(int(n_used))
            R_plan[k].append(float(R_used))
            remaining[k] -= int(B_used)
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
                    "queue_weight": queue_weight,
                    "skipped": False,
                }
            )
            if verbose:
                print(
                    f"  user={k:02d} block={block:02d} "
                    f"bits={B_used:4d} n_kl={n_used:4d} "
                    f"required_rate={required_rate:.4f} "
                    f"R_fbl={R_used:.4f} margin={rate_margin:.4f}"
                )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(len(transmit_users)),
                "skipped_users": int(len(skipped_users)),
                "allocated_bits": int(block_bits),
                "remaining_bits": int(np.sum(remaining)),
                "feasible_users": int(block_eval["feasible_count"]),
                "min_max_bits": int(block_eval["min_max_bits"]),
                "queue_weights": {int(k): float(v) for k, v in queue_weights.items()},
                "final_precoder_delta": float(block_history[-1]["max_precoder_delta"]) if block_history else 0.0,
            }
        )
        if verbose:
            print(
                f"--- Block {block} allocation complete | "
                f"allocated_bits={block_bits} remaining_bits={int(np.sum(remaining))} ---"
            )
        block += 1

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)
    return {
        "method_name": method_name,
        "objective_mode": objective_mode,
        "allocation_mode": allocation_mode,
        "weight_strategy": weight_strategy,
        "precoder_parameterization": "shared_user_channel_to_precoder_mlp",
        "user_model_specs": export_user_model_specs(system.Nr, system.Nb, system.dk),
        "n_kl": copy.deepcopy(n_plan),
        "B_kl": copy.deepcopy(B_plan),
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": copy.deepcopy(R_plan),
        "initial_latency": initial_latency,
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
    }


def optimize_downlink_safe_sweep(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    objective_mode = resolve_safe_sweep_objective_mode(sim_params)
    return _run_safe_sweep(
        system,
        sim_params,
        verbose=verbose,
        method_name="downlink_greedy_safe_sweep",
        objective_mode=objective_mode,
        allocation_mode="greedy",
        weight_strategy="remaining_bits",
    )
