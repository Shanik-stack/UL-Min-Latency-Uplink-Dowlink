from __future__ import annotations

import copy
import random
from typing import Any, List

import numpy as np
import torch

from downlink_system import DownlinkSystem
from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE, build_experiment_scenario
from precoder_models import (
    DEVICE,
    build_user_precoder_net,
    build_shared_bs_precoder_net,
    export_user_model_specs,
    infer_raw_bs_precoders_numpy,
    infer_raw_bs_precoders_torch,
    infer_raw_precoder_numpy,
    infer_raw_precoder_torch,
    model_outputs_full_bs_precoder,
    resolve_downlink_precoder_net_scope,
)
from terminal_logging import format_log_line, format_latency_log_line
LOG2E_SQ = float(np.log2(np.e) ** 2)
USER_RATE_MODE = "user_rate"
UNWEIGHTED_SUM_RATE_MODE = "unweighted_sum_rate"
REMAINING_BITS_WEIGHTED_SUM_RATE_MODE = "remaining_bits_weighted_sum_rate"
BLENDED_NETWORK_RATE_MODE = "blended_network_rate"
CONVERGENCE_OBJECTIVE_MODE_ALIASES = {
    USER_RATE_MODE: UNWEIGHTED_SUM_RATE_MODE,
    "sum_rate": UNWEIGHTED_SUM_RATE_MODE,
    UNWEIGHTED_SUM_RATE_MODE: UNWEIGHTED_SUM_RATE_MODE,
    "weighted_sum_rate": REMAINING_BITS_WEIGHTED_SUM_RATE_MODE,
    REMAINING_BITS_WEIGHTED_SUM_RATE_MODE: REMAINING_BITS_WEIGHTED_SUM_RATE_MODE,
    BLENDED_NETWORK_RATE_MODE: BLENDED_NETWORK_RATE_MODE,
}
CONVERGENCE_OBJECTIVE_MODES = set(CONVERGENCE_OBJECTIVE_MODE_ALIASES.values())
CONSTRAINT_LOSS_FORMS = {"plain_lagrangian", "augmented_lagrangian"}
POWER_PROJECTION_SAFETY_MARGIN = 1e-6


def resolve_convergence_objective_mode(sim_params: dict[str, Any]) -> str:
    raw_mode = str(
        sim_params.get(
            "convergence_block_objective_mode",
            sim_params.get(
                "safe_sweep_objective_mode",
                sim_params.get(
                    "downlink_safe_sweep_objective_mode",
                    sim_params.get("objective_mode", UNWEIGHTED_SUM_RATE_MODE),
                ),
            ),
        )
    ).strip().lower()
    if raw_mode not in CONVERGENCE_OBJECTIVE_MODE_ALIASES:
        known = ", ".join(sorted(CONVERGENCE_OBJECTIVE_MODES))
        raise ValueError(
            f"Unknown convergence objective mode '{raw_mode}'. Expected one of: {known}"
        )
    return str(CONVERGENCE_OBJECTIVE_MODE_ALIASES[raw_mode])


def resolve_safe_sweep_objective_mode(sim_params: dict[str, Any]) -> str:
    return resolve_convergence_objective_mode(sim_params)


def convergence_objective_tag(objective_mode: str) -> str:
    safe_mode = str(resolve_objective_mode_alias(objective_mode)).strip().lower().replace(" ", "_").replace("-", "_")
    return f"obj_{safe_mode}"


def safe_sweep_objective_tag(objective_mode: str) -> str:
    return convergence_objective_tag(objective_mode)


def resolve_objective_mode_alias(objective_mode: str) -> str:
    raw_mode = str(objective_mode).strip().lower()
    if raw_mode not in CONVERGENCE_OBJECTIVE_MODE_ALIASES:
        known = ", ".join(sorted(CONVERGENCE_OBJECTIVE_MODES))
        raise ValueError(
            f"Unknown convergence objective mode '{raw_mode}'. Expected one of: {known}"
        )
    return str(CONVERGENCE_OBJECTIVE_MODE_ALIASES[raw_mode])


def objective_uses_user_weights(objective_mode: str) -> bool:
    canonical_mode = resolve_objective_mode_alias(objective_mode)
    return canonical_mode in {
        REMAINING_BITS_WEIGHTED_SUM_RATE_MODE,
        BLENDED_NETWORK_RATE_MODE,
    }


def objective_display_name(objective_mode: str) -> str:
    return str(resolve_objective_mode_alias(objective_mode))


def objective_weight_strategy_name(
    objective_mode: str,
    configured_weight_strategy: str,
) -> str:
    if not objective_uses_user_weights(objective_mode):
        return "none"
    return str(configured_weight_strategy)


def resolve_constraint_loss_form(sim_params: dict[str, Any]) -> str:
    raw_mode = str(sim_params.get("constraint_loss_form", "plain_lagrangian")).strip().lower()
    if raw_mode not in CONSTRAINT_LOSS_FORMS:
        known = ", ".join(sorted(CONSTRAINT_LOSS_FORMS))
        raise ValueError(
            f"Unknown constraint loss form '{raw_mode}'. Expected one of: {known}"
        )
    return raw_mode


def _constraint_violation_activation(value: torch.Tensor, loss_form: str) -> torch.Tensor:
    if loss_form == "plain_lagrangian":
        return torch.nn.functional.leaky_relu(value)
    return torch.relu(value)


def _clone_precoders(F_nested: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    return [[np.array(F_kl, copy=True) for F_kl in user_F] for user_F in F_nested]


def _downlink_precoder_parameterization(model_scope: str) -> str:
    scope = resolve_downlink_precoder_net_scope(model_scope)
    if scope == "bs_shared_net":
        return "bs_shared_block_context_to_full_precoder_mlp"
    return "per_user_channel_to_precoder_mlp"


def _initial_baseline_model_scope() -> str:
    # Keep the initial random baseline fixed across downlink architecture variants
    # so comparisons at the same seed start from the same point.
    return "per_user_nets"


def _build_user_precoder_models(
    system: DownlinkSystem,
    *,
    init_seed: int | None = None,
    model_scope: str = "per_user_nets",
) -> list[torch.nn.Module]:
    resolved_scope = resolve_downlink_precoder_net_scope(model_scope)

    def _construct_models() -> list[torch.nn.Module]:
        if resolved_scope == "bs_shared_net":
            shared_model = build_shared_bs_precoder_net(
                k_count=int(system.K),
                max_nr=int(np.max(system.Nr)),
                max_nb=int(np.max(system.Nb)),
                max_dk=int(np.max(system.dk)),
                device=DEVICE,
            )
            return [shared_model for _ in range(int(system.K))]

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


def _build_user_model_optimizers(
    user_models: list[torch.nn.Module],
    *,
    lr: float,
) -> list[torch.optim.Optimizer]:
    optimizers_by_model_id: dict[int, torch.optim.Optimizer] = {}
    model_optimizers: list[torch.optim.Optimizer] = []
    for model in user_models:
        model_id = id(model)
        if model_id not in optimizers_by_model_id:
            optimizers_by_model_id[model_id] = torch.optim.Adam(model.parameters(), lr=float(lr))
        model_optimizers.append(optimizers_by_model_id[model_id])
    return model_optimizers


def _user_models_output_full_bs_precoder(user_models: list[torch.nn.Module]) -> bool:
    return len(user_models) > 0 and model_outputs_full_bs_precoder(user_models[0])


def _active_mask_for_users(system: DownlinkSystem, active_users: List[int]) -> list[int]:
    active_set = {int(k) for k in active_users}
    return [1 if int(k) in active_set else 0 for k in range(int(system.K))]


def _context_channels_for_block(system: DownlinkSystem, block: int) -> list[np.ndarray]:
    channels: list[np.ndarray] = []
    l = int(block)
    for k in range(int(system.K)):
        if l < len(system.H[k]):
            channels.append(np.asarray(system.H[k][l], dtype=np.complex64))
        else:
            channels.append(np.zeros((int(system.Nr[k]), int(system.Nb[k])), dtype=np.complex64))
    return channels


def _infer_shared_block_precoders_numpy(
    system: DownlinkSystem,
    shared_model: torch.nn.Module,
    block: int,
    active_users: List[int],
) -> dict[int, np.ndarray]:
    beams = infer_raw_bs_precoders_numpy(
        shared_model,
        _context_channels_for_block(system, int(block)),
        _active_mask_for_users(system, active_users),
        system.Nb,
        system.dk,
        device=DEVICE,
    )
    return {int(k): np.asarray(beams[int(k)], dtype=np.complex128) for k in active_users}


def _infer_shared_block_precoders_torch(
    system: DownlinkSystem,
    shared_model: torch.nn.Module,
    block: int,
    active_users: List[int],
) -> dict[int, torch.Tensor]:
    H_block = [
        torch.tensor(np.asarray(H_kl), dtype=torch.complex64, device=DEVICE)
        for H_kl in _context_channels_for_block(system, int(block))
    ]
    active_mask = torch.tensor(_active_mask_for_users(system, active_users), dtype=torch.float32, device=DEVICE)
    beams = infer_raw_bs_precoders_torch(
        shared_model,
        H_block,
        active_mask,
        system.Nb,
        system.dk,
    )
    return {int(k): beams[int(k)] for k in active_users}


def _build_precoder_snapshot_from_models(
    system: DownlinkSystem,
    user_models: list[torch.nn.Module],
) -> List[List[np.ndarray]]:
    if _user_models_output_full_bs_precoder(user_models):
        snapshot: List[List[np.ndarray]] = [[] for _ in range(int(system.K))]
        max_blocks = max((len(user_blocks) for user_blocks in system.H), default=0)
        shared_model = user_models[0]
        for l in range(max_blocks):
            block_precoders = _infer_shared_block_precoders_numpy(
                system,
                shared_model,
                int(l),
                list(range(int(system.K))),
            )
            for k in range(int(system.K)):
                snapshot[k].append(np.asarray(block_precoders[int(k)], dtype=np.complex128))
            system.project_block_precoders_to_power(snapshot, l)
        return snapshot

    snapshot: List[List[np.ndarray]] = []
    for k in range(int(system.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(len(system.H[k])):
            user_blocks.append(
                infer_raw_precoder_numpy(
                    user_models[k],
                    np.asarray(system.H[k][l], dtype=np.complex64),
                    nb=int(system.Nb[k]),
                    dk=int(system.dk[k]),
                    device=DEVICE,
                    user_index=int(k),
                )
            )
        snapshot.append(user_blocks)
    max_blocks = max((len(user_blocks) for user_blocks in snapshot), default=0)
    for l in range(max_blocks):
        system.project_block_precoders_to_power(snapshot, l)
    return snapshot


def _refresh_block_precoders_from_models(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    active_users: List[int],
    block: int,
) -> None:
    l = int(block)
    if _user_models_output_full_bs_precoder(user_models):
        block_precoders = _infer_shared_block_precoders_numpy(
            system,
            user_models[0],
            l,
            active_users,
        )
        for k in active_users:
            k_int = int(k)
            if l >= len(working_F[k_int]):
                raise ValueError(f"User {k_int} has no precoder slot for block {l}.")
            working_F[k_int][l] = np.asarray(block_precoders[k_int], dtype=np.complex128)
        system.project_block_precoders_to_power(working_F, l, active_users=[int(k) for k in active_users])
        return

    for k in active_users:
        k_int = int(k)
        if l >= len(working_F[k_int]):
            raise ValueError(f"User {k_int} has no precoder slot for block {l}.")
        working_F[k_int][l] = infer_raw_precoder_numpy(
            user_models[k_int],
            np.asarray(system.H[k_int][l], dtype=np.complex64),
            nb=int(system.Nb[k_int]),
            dk=int(system.dk[k_int]),
            device=DEVICE,
            user_index=int(k_int),
        )
    system.project_block_precoders_to_power(working_F, l, active_users=[int(k) for k in active_users])


def _copy_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _complex_to_param(F_mat: np.ndarray) -> torch.nn.Parameter:
    arr = np.asarray(F_mat, dtype=np.complex64)
    stacked = np.stack([arr.real, arr.imag], axis=0)
    return torch.nn.Parameter(torch.tensor(stacked, dtype=torch.float32, device=DEVICE))


def _param_to_complex(param: torch.Tensor) -> torch.Tensor:
    return (param[0] + 1j * param[1]).to(torch.complex64)


def _project_active_precoders_to_block_power(
    system: DownlinkSystem,
    precoders: dict[int, torch.Tensor],
    active_users: List[int],
    eps: float = 1e-12,
) -> dict[int, torch.Tensor]:
    if len(active_users) == 0:
        return precoders
    total_power = torch.zeros((), dtype=torch.float32, device=DEVICE)
    for k in active_users:
        total_power = total_power + (torch.linalg.norm(precoders[int(k)], ord="fro") ** 2).real
    if float(total_power.detach().cpu()) <= float(eps):
        return precoders
    scale = (
        torch.sqrt(
            torch.tensor(float(system.block_power_budget), device=DEVICE, dtype=torch.float32) / (total_power + eps)
        )
        * (1.0 - float(POWER_PROJECTION_SAFETY_MARGIN))
    )
    return {int(k): (precoders[int(k)] * scale.to(precoders[int(k)].dtype)) for k in active_users}


def _q_inv(epsilon: float) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64, device=DEVICE),
        torch.tensor(1.0, dtype=torch.float64, device=DEVICE),
    )
    p = torch.tensor(1.0 - float(epsilon), dtype=torch.float64, device=DEVICE)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return normal.icdf(p).to(dtype=torch.float32)


def _resolve_user_n_kl(
    system: DownlinkSystem,
    user: int,
    n_kl_overrides: dict[int, int] | None = None,
) -> int:
    k = int(user)
    if n_kl_overrides is None:
        return int(system.T[k])
    return int(n_kl_overrides.get(k, int(system.T[k])))


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
    n_kl_overrides: dict[int, int] | None = None,
) -> float:
    canonical_mode = resolve_objective_mode_alias(objective_mode)
    snapshot = _clone_precoders(working_F)
    system.project_block_precoders_to_power(snapshot, int(block), active_users=[int(k) for k in active_users])
    if canonical_mode == USER_RATE_MODE:
        n_focus = _resolve_user_n_kl(system, int(focus_user), n_kl_overrides)
        return float(system.compute_block_rate(int(focus_user), int(block), n_focus, F_override=snapshot))

    if canonical_mode == BLENDED_NETWORK_RATE_MODE:
        n_focus = _resolve_user_n_kl(system, int(focus_user), n_kl_overrides)
        self_rate = float(system.compute_block_rate(int(focus_user), int(block), n_focus, F_override=snapshot))
        others_total = 0.0
        for k in active_users:
            if int(k) == int(focus_user):
                continue
            n_k = _resolve_user_n_kl(system, int(k), n_kl_overrides)
            others_total += float(user_weights.get(int(k), 1.0)) * float(
                system.compute_block_rate(int(k), int(block), n_k, F_override=snapshot)
            )
        return self_rate + float(network_weight_beta) * others_total

    total = 0.0
    for k in active_users:
        n_k = _resolve_user_n_kl(system, int(k), n_kl_overrides)
        rate_k = float(system.compute_block_rate(int(k), int(block), n_k, F_override=snapshot))
        if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
            total += float(user_weights.get(int(k), 1.0)) * rate_k
        else:
            total += rate_k
    return total


def _evaluate_network_objective_numpy(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
    objective_mode: str,
    user_weights: dict[int, float],
    network_weight_beta: float,
    n_kl_overrides: dict[int, int] | None = None,
) -> float:
    canonical_mode = resolve_objective_mode_alias(objective_mode)
    snapshot = _clone_precoders(working_F)
    system.project_block_precoders_to_power(snapshot, int(block), active_users=[int(k) for k in active_users])

    total_rate = 0.0
    weighted_total = 0.0
    for k in active_users:
        n_k = _resolve_user_n_kl(system, int(k), n_kl_overrides)
        rate_k = float(system.compute_block_rate(int(k), int(block), n_k, F_override=snapshot))
        total_rate += rate_k
        weighted_total += float(user_weights.get(int(k), 1.0)) * rate_k

    if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
        return weighted_total
    if canonical_mode == BLENDED_NETWORK_RATE_MODE:
        return total_rate + float(network_weight_beta) * weighted_total
    return total_rate


def _block_rate_torch_from_precoders(
    system: DownlinkSystem,
    active_users: List[int],
    block: int,
    user: int,
    n_kl: int,
    precoders: dict[int, torch.Tensor],
) -> torch.Tensor:
    k = int(user)
    l = int(block)
    Hk = torch.tensor(system.H[k][l], dtype=torch.complex64, device=DEVICE)
    Nrk = int(system.Nr[k])
    I = torch.eye(Nrk, dtype=torch.complex64, device=DEVICE)

    Fk = precoders[int(k)]
    noise_cov = float(system.sigma2[k]) * I
    for j in active_users:
        j_int = int(j)
        if j_int == k:
            continue
        Fj = precoders[j_int]
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


def _evaluate_constrained_block_state(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    active_users: List[int],
    update_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    objective_mode: str,
    user_weights: dict[int, float],
    lambda_rate: dict[int, float],
    lambda_power_block: float,
    *,
    n_kl_overrides: dict[int, int] | None = None,
) -> dict[str, Any]:
    network_weight_beta = float(sim_params.get("network_rate_weight", 0.15))
    constraint_loss_form = resolve_constraint_loss_form(sim_params)
    canonical_mode = resolve_objective_mode_alias(objective_mode)
    rho_rate = float(sim_params.get("augmented_lagrangian_rho_rate", 0.0))
    rho_power = float(sim_params.get("augmented_lagrangian_rho_power", 0.0))
    update_user_set = {int(k) for k in update_users}

    precoders: dict[int, torch.Tensor] = {}
    if _user_models_output_full_bs_precoder(user_models) and len(update_user_set) > 0:
        precoders = _infer_shared_block_precoders_torch(
            system,
            user_models[0],
            int(block),
            active_users,
        )
    else:
        for k in active_users:
            k_int = int(k)
            if k_int in update_user_set:
                H_kl = torch.tensor(system.H[k_int][int(block)], dtype=torch.complex64, device=DEVICE)
                precoders[k_int] = infer_raw_precoder_torch(
                    user_models[k_int],
                    H_kl,
                    nb=int(system.Nb[k_int]),
                    dk=int(system.dk[k_int]),
                    user_index=int(k_int),
                )
            else:
                precoders[k_int] = torch.tensor(
                    working_F[k_int][int(block)],
                    dtype=torch.complex64,
                    device=DEVICE,
                )
    precoders = _project_active_precoders_to_block_power(system, precoders, active_users)

    rates: dict[int, torch.Tensor] = {}
    powers: dict[int, torch.Tensor] = {}
    required_rates: dict[int, float] = {}
    rate_gaps: dict[int, torch.Tensor] = {}
    rate_violation_pos: dict[int, torch.Tensor] = {}
    rate_constraint_terms: dict[int, torch.Tensor] = {}

    for k in active_users:
        k_int = int(k)
        n_k = _resolve_user_n_kl(system, k_int, n_kl_overrides)
        rate_k = _block_rate_torch_from_precoders(
            system,
            active_users,
            int(block),
            k_int,
            n_k,
            precoders,
        )
        power_k = (torch.linalg.norm(precoders[k_int], ord="fro") ** 2).real
        required_rate_k = float(requested_bits.get(k_int, 0)) / float(max(int(n_k), 1))
        rate_gap_k = torch.tensor(required_rate_k, dtype=torch.float32, device=DEVICE) - rate_k
        rates[k_int] = rate_k
        powers[k_int] = power_k
        required_rates[k_int] = float(required_rate_k)
        rate_gaps[k_int] = rate_gap_k
        rate_violation_pos[k_int] = torch.relu(rate_gap_k)
        rate_constraint_terms[k_int] = _constraint_violation_activation(rate_gap_k, constraint_loss_form)

    block_power = (
        torch.stack([powers[int(k)] for k in active_users]).sum()
        if active_users
        else torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    )
    block_power_gap = block_power - float(system.block_power_budget)
    block_power_violation_pos = torch.relu(block_power_gap)
    block_power_constraint_term = _constraint_violation_activation(block_power_gap, constraint_loss_form)

    total_rate = torch.stack([rates[int(k)] for k in active_users]).sum() if active_users else torch.tensor(0.0, device=DEVICE)
    weighted_total = torch.stack(
        [
            torch.tensor(float(user_weights.get(int(k), 1.0)), dtype=torch.float32, device=DEVICE) * rates[int(k)]
            for k in active_users
        ]
    ).sum() if active_users else torch.tensor(0.0, device=DEVICE)
    blended_total = total_rate + float(network_weight_beta) * weighted_total
    if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
        objective = weighted_total
    elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
        objective = blended_total
    else:
        objective = total_rate

    loss = -objective
    for k in active_users:
        k_int = int(k)
        loss = loss + float(lambda_rate.get(k_int, 0.0)) * rate_constraint_terms[k_int]
        if constraint_loss_form == "augmented_lagrangian":
            loss = loss + 0.5 * rho_rate * rate_violation_pos[k_int].pow(2)
    loss = loss + float(lambda_power_block) * block_power_constraint_term
    if constraint_loss_form == "augmented_lagrangian":
        loss = loss + 0.5 * rho_power * block_power_violation_pos.pow(2)

    return {
        "loss": loss,
        "rates": rates,
        "powers": powers,
        "required_rates": required_rates,
        "rate_gap": rate_gaps,
        "rate_violation_pos": rate_violation_pos,
        "block_power": block_power,
        "block_power_gap": block_power_gap,
        "block_power_violation_pos": block_power_violation_pos,
        "sum_rate": total_rate,
        REMAINING_BITS_WEIGHTED_SUM_RATE_MODE: weighted_total,
        "weighted_sum_rate": weighted_total,
        "blended_objective": blended_total,
        "constraint_loss_form": constraint_loss_form,
    }


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

    power = float(sim_params.get("remaining_bits_weight_power", 1.0))
    min_weight = float(sim_params.get("minimum_user_weight", 0.25))
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


def _optimize_shared_block_precoders(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    user_weights: dict[int, float],
    block: int,
    sim_params: dict[str, Any],
    objective_mode: str,
    shared_model: torch.nn.Module,
    shared_optimizer: torch.optim.Optimizer,
    n_kl_overrides: dict[int, int] | None = None,
) -> dict[int, np.ndarray]:
    steps = max(1, int(sim_params.get("user_update_steps", 1)))
    network_weight_beta = float(sim_params.get("network_rate_weight", 0.15))
    canonical_mode = resolve_objective_mode_alias(objective_mode)

    best_model_state = _copy_model_state(shared_model)
    best_snapshot = _clone_precoders(working_F)
    best_objective = _evaluate_network_objective_numpy(
        system,
        best_snapshot,
        active_users,
        int(block),
        objective_mode,
        user_weights,
        network_weight_beta,
        n_kl_overrides,
    )

    for _ in range(steps):
        shared_optimizer.zero_grad()
        precoders = _infer_shared_block_precoders_torch(
            system,
            shared_model,
            int(block),
            active_users,
        )
        precoders = _project_active_precoders_to_block_power(system, precoders, active_users)

        total_rate = torch.zeros((), dtype=torch.float32, device=DEVICE)
        weighted_total = torch.zeros((), dtype=torch.float32, device=DEVICE)
        for j in active_users:
            n_j = _resolve_user_n_kl(system, int(j), n_kl_overrides)
            rate_j = _block_rate_torch_from_precoders(
                system,
                active_users,
                int(block),
                int(j),
                n_j,
                precoders,
            )
            total_rate = total_rate + rate_j
            weighted_total = weighted_total + (
                torch.tensor(float(user_weights.get(int(j), 1.0)), dtype=torch.float32, device=DEVICE) * rate_j
            )

        if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
            objective = weighted_total
        elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
            objective = total_rate + float(network_weight_beta) * weighted_total
        else:
            objective = total_rate
        (-objective).backward()
        shared_optimizer.step()

        block_precoders = _infer_shared_block_precoders_numpy(
            system,
            shared_model,
            int(block),
            active_users,
        )
        snapshot = _clone_precoders(working_F)
        for j in active_users:
            snapshot[int(j)][int(block)] = np.asarray(block_precoders[int(j)], dtype=np.complex128)
        system.project_block_precoders_to_power(snapshot, int(block), active_users=[int(j) for j in active_users])
        objective_np = _evaluate_network_objective_numpy(
            system,
            snapshot,
            active_users,
            int(block),
            objective_mode,
            user_weights,
            network_weight_beta,
            n_kl_overrides,
        )
        if objective_np > best_objective:
            best_objective = float(objective_np)
            best_model_state = _copy_model_state(shared_model)
            best_snapshot = snapshot

    shared_model.load_state_dict(best_model_state)
    return {
        int(k): np.asarray(best_snapshot[int(k)][int(block)], dtype=np.complex128)
        for k in active_users
    }


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
    n_kl_overrides: dict[int, int] | None = None,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    n_kl = _resolve_user_n_kl(system, k, n_kl_overrides)
    steps = max(1, int(sim_params.get("user_update_steps", 1)))
    network_weight_beta = float(sim_params.get("network_rate_weight", 0.15))
    canonical_mode = resolve_objective_mode_alias(objective_mode)
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
        n_kl_overrides,
    )

    for _ in range(steps):
        model_optimizer.zero_grad()
        F_candidate = infer_raw_precoder_torch(
            precoder_model,
            H_kl,
            nb=int(system.Nb[k]),
            dk=int(system.dk[k]),
            user_index=int(k),
        )
        precoders = {
            int(j): (
                F_candidate
                if int(j) == k
                else torch.tensor(working_F[int(j)][l], dtype=torch.complex64, device=DEVICE)
            )
            for j in active_users
        }
        precoders = _project_active_precoders_to_block_power(system, precoders, active_users)
        if canonical_mode in {REMAINING_BITS_WEIGHTED_SUM_RATE_MODE, BLENDED_NETWORK_RATE_MODE}:
            objective = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
            for j in active_users:
                n_j = _resolve_user_n_kl(system, int(j), n_kl_overrides)
                rate_j = _block_rate_torch_from_precoders(
                    system,
                    active_users,
                    l,
                    int(j),
                    n_j,
                    precoders,
                )
                if canonical_mode == BLENDED_NETWORK_RATE_MODE and int(j) == k:
                    objective = objective + rate_j
                else:
                    scale = float(user_weights.get(int(j), 1.0))
                    if canonical_mode == BLENDED_NETWORK_RATE_MODE:
                        scale *= network_weight_beta
                    objective = objective + scale * rate_j
        else:
            objective = _block_rate_torch_from_precoders(
                system,
                active_users,
                l,
                k,
                n_kl,
                precoders,
            )
        (-objective).backward()
        model_optimizer.step()

        beam_np = infer_raw_precoder_numpy(
            precoder_model,
            np.asarray(system.H[k][l], dtype=np.complex64),
            nb=int(system.Nb[k]),
            dk=int(system.dk[k]),
            device=DEVICE,
            user_index=int(k),
        )
        beam_snapshot = _clone_precoders(working_F)
        beam_snapshot[k][l] = beam_np
        system.project_block_precoders_to_power(beam_snapshot, l, active_users=[int(j) for j in active_users])
        objective_np = _evaluate_update_objective_numpy(
            system,
            beam_snapshot,
            active_users,
            k,
            l,
            objective_mode,
            user_weights,
            network_weight_beta,
            n_kl_overrides,
        )
        if objective_np > best_objective:
            best_beam = np.array(beam_snapshot[k][l], copy=True)
            best_objective = objective_np
            best_model_state = _copy_model_state(precoder_model)

    precoder_model.load_state_dict(best_model_state)

    return best_beam


def _optimize_user_block_precoder_constrained(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    active_users: List[int],
    requested_bits: dict[int, int],
    user_weights: dict[int, float],
    user: int,
    block: int,
    sim_params: dict[str, Any],
    objective_mode: str,
    precoder_model: torch.nn.Module,
    model_optimizer: torch.optim.Optimizer,
    lambda_rate: dict[int, float],
    lambda_power_block: float,
    n_kl_overrides: dict[int, int] | None = None,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    steps = max(1, int(sim_params.get("user_update_steps", 1)))

    for _ in range(steps):
        model_optimizer.zero_grad()
        state = _evaluate_constrained_block_state(
            system,
            working_F,
            user_models=user_models,
            active_users=active_users,
            update_users=[k],
            block=l,
            requested_bits=requested_bits,
            sim_params=sim_params,
            objective_mode=objective_mode,
            user_weights=user_weights,
            lambda_rate=lambda_rate,
            lambda_power_block=lambda_power_block,
            n_kl_overrides=n_kl_overrides,
        )
        state["loss"].backward()
        model_optimizer.step()

    beam_np = infer_raw_precoder_numpy(
        precoder_model,
        np.asarray(system.H[k][l], dtype=np.complex64),
        nb=int(system.Nb[k]),
        dk=int(system.dk[k]),
        device=DEVICE,
        user_index=int(k),
    )
    beam_snapshot = _clone_precoders(working_F)
    beam_snapshot[k][l] = np.asarray(beam_np, dtype=np.complex128)
    system.project_block_precoders_to_power(beam_snapshot, l, active_users=[int(j) for j in active_users])
    return np.asarray(beam_snapshot[k][l], dtype=np.complex128)


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
    n_kl_overrides: dict[int, int] | None = None,
) -> dict[str, Any]:
    rates = []
    max_bits = []
    n_values = []
    for k in active_users:
        n_k = _resolve_user_n_kl(system, int(k), n_kl_overrides)
        rate = float(system.compute_block_rate(int(k), int(block), n_k, F_override=working_F))
        bits = max(_rate_to_max_bits(n_k, rate), 0)
        rates.append(rate)
        max_bits.append(int(bits))
        n_values.append(int(n_k))

    feasible_count = int(sum(bits > 0 for bits in max_bits))
    return {
        "user_ids": [int(k) for k in active_users],
        "user_n_kl": n_values,
        "user_rates": rates,
        "user_max_bits": max_bits,
        "feasible_count": feasible_count,
        "min_max_bits": int(min(max_bits)) if len(max_bits) > 0 else 0,
        "min_rate": float(min(rates)) if len(rates) > 0 else 0.0,
        "sum_rate": float(sum(rates)),
    }


def _copy_active_model_optimizer_states(
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    model_states = {
        int(k): _copy_model_state(user_models[int(k)])
        for k in active_users
    }
    optimizer_states = {
        int(k): copy.deepcopy(model_optimizers[int(k)].state_dict())
        for k in active_users
    }
    return model_states, optimizer_states


def _restore_active_model_optimizer_states(
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    model_states: dict[int, dict[str, torch.Tensor]],
    optimizer_states: dict[int, dict[str, Any]],
) -> None:
    for k in active_users:
        user_models[int(k)].load_state_dict(model_states[int(k)])
        model_optimizers[int(k)].load_state_dict(optimizer_states[int(k)])


def _capture_active_block_solver_state(
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    lambda_rate: dict[int, float],
    lambda_power_block: float,
) -> dict[str, Any]:
    model_states, optimizer_states = _copy_active_model_optimizer_states(
        user_models,
        model_optimizers,
        active_users,
    )
    return {
        "working_F": _clone_precoders(working_F),
        "model_states": model_states,
        "optimizer_states": optimizer_states,
        "lambda_rate": {int(k): float(v) for k, v in lambda_rate.items()},
        "lambda_power_block": float(lambda_power_block),
    }


def _restore_active_block_solver_state(
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    state: dict[str, Any],
) -> None:
    working_F[:] = _clone_precoders(state["working_F"])
    _restore_active_model_optimizer_states(
        user_models,
        model_optimizers,
        active_users,
        state["model_states"],
        state["optimizer_states"],
    )


def _all_committed_bits_feasible(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
    committed_bits: dict[int, int],
    n_kl_targets: dict[int, int],
) -> tuple[bool, dict[int, float], list[int]]:
    user_rates: dict[int, float] = {}
    infeasible_users: list[int] = []
    for k in active_users:
        k_int = int(k)
        B_used = int(committed_bits.get(k_int, 0))
        n_k = _resolve_user_n_kl(system, k_int, n_kl_targets)
        rate_k = float(system.compute_block_rate(k_int, int(block), n_k, F_override=working_F))
        user_rates[k_int] = rate_k
        if B_used <= 0:
            continue
        required_rate = float(B_used) / float(max(n_k, 1))
        if float(required_rate - rate_k) > 0.0:
            infeasible_users.append(int(k_int))
    return len(infeasible_users) == 0, user_rates, infeasible_users


def _resolve_reduced_n_reoptimization_users(
    active_users: List[int],
    candidate_user: int,
    infeasible_users: list[int],
    scope: str,
) -> list[int]:
    ordered_active = [int(k) for k in active_users]
    infeasible_set = {int(k) for k in infeasible_users}
    scope_key = str(scope).strip().lower()

    if scope_key == "all_active_users":
        return ordered_active

    if scope_key == "infeasible_users_only":
        return [k for k in ordered_active if k in infeasible_set]

    if scope_key == "candidate_and_infeasible_users":
        update_users = [int(candidate_user)]
        update_users.extend(k for k in ordered_active if k in infeasible_set and int(k) != int(candidate_user))
        return update_users

    raise ValueError(
        "simulation.n_kl_reduction_update_scope must be one of "
        "{'all_active_users', 'infeasible_users_only', 'candidate_and_infeasible_users'}."
    )


def _reduce_blocklengths_with_reoptimization(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    *,
    objective_mode: str,
    user_weights: dict[int, float],
    verbose: bool,
    dual_state: dict[str, dict[int, float]] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, float]], dict[str, dict[int, float]]]:
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    reoptimization_scope = str(
        sim_params.get("n_kl_reduction_update_scope", "all_active_users")
    ).strip().lower()

    current_n_targets = {
        int(k): int(system.T[int(k)])
        for k in active_users
    }
    committed_bits: dict[int, int] = {}
    current_rates: dict[int, float] = {}
    plans: dict[int, dict[str, Any]] = {}
    refinement_history: list[dict[str, float]] = []
    current_dual_state = copy.deepcopy(dual_state) if dual_state is not None else None
    stopped_reduction_users: set[int] = set()

    for k in active_users:
        k_int = int(k)
        T_k = int(system.T[k_int])
        R_T = float(system.compute_block_rate(k_int, int(block), T_k, F_override=working_F))
        B_max = max(_rate_to_max_bits(T_k, R_T), 0)
        B_used = int(min(int(requested_bits.get(k_int, 0)), B_max))
        committed_bits[k_int] = int(B_used)
        current_rates[k_int] = float(R_T)
        plans[k_int] = {
            "B_used": int(B_used),
            "n_used": int(T_k),
            "R_used": float(R_T if B_used > 0 else 0.0),
        }

    full_service_users = []
    for k in active_users:
        k_int = int(k)
        requested_k = int(requested_bits.get(k_int, 0))
        B_used = int(committed_bits.get(k_int, 0))
        if requested_k <= 0 or B_used <= 0:
            continue
        if B_used < requested_k:
            if verbose:
                print(
                    f"  user={k_int:02d} block={block:02d} serves partial bits at n=T; "
                    "not reducing n_kl."
                )
            continue
        full_service_users.append(int(k_int))

    epoch_idx = 0
    while len(full_service_users) > 0:
        active_reduction_users = [
            int(k_int)
            for k_int in full_service_users
            if int(k_int) not in stopped_reduction_users
        ]
        if len(active_reduction_users) == 0:
            break
        progress_this_epoch = False
        start_offset = int(epoch_idx % max(len(active_reduction_users), 1))
        ordered_users = (
            active_reduction_users[start_offset:] + active_reduction_users[:start_offset]
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Reduction]",
                    block=int(block),
                    epoch=int(epoch_idx + 1),
                    users=[int(k) for k in ordered_users],
                )
            )

        for k_int in ordered_users:
            current_n = int(current_n_targets.get(k_int, int(system.T[k_int])))
            candidate = int(current_n - int(n_step))
            if candidate < int(n_min):
                stopped_reduction_users.add(int(k_int))
                continue

            candidate_targets = dict(current_n_targets)
            candidate_targets[k_int] = int(candidate)
            feasible_without_reopt, candidate_rates, infeasible_users = _all_committed_bits_feasible(
                system,
                working_F,
                active_users,
                block,
                committed_bits,
                candidate_targets,
            )
            if feasible_without_reopt:
                current_n_targets = candidate_targets
                current_rates = candidate_rates
                plans[k_int]["n_used"] = int(candidate)
                plans[k_int]["R_used"] = float(candidate_rates.get(k_int, current_rates.get(k_int, 0.0)))
                progress_this_epoch = True
                if verbose:
                    print(
                        f"  user={k_int:02d} block={block:02d} accepted smaller n_kl={int(candidate):4d} "
                        "without fresh re-optimization."
                    )
                continue
            if verbose:
                infeasible_text = ", ".join(str(int(user_id)) for user_id in infeasible_users)
                print(
                    f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate):4d} "
                    f"breaks committed-user feasibility [{infeasible_text}]; trying fresh re-optimization."
                )

            update_users = _resolve_reduced_n_reoptimization_users(
                active_users,
                k_int,
                infeasible_users,
                reoptimization_scope,
            )
            solver_checkpoint = _capture_active_block_solver_state(
                working_F,
                user_models,
                model_optimizers,
                active_users,
                (
                    current_dual_state["lambda_rate"]
                    if current_dual_state is not None
                    else {
                        int(user_id): float(sim_params.get("initial_lambda_rate_constraint", 0.1))
                        for user_id in active_users
                    }
                ),
                (
                    float(current_dual_state["lambda_power_block"])
                    if current_dual_state is not None
                    else float(sim_params.get("initial_lambda_power_constraint", 0.01))
                ),
            )
            solve_result = optimize_precoders_for_block_constrained(
                system,
                working_F,
                user_models,
                model_optimizers,
                active_users,
                block,
                committed_bits,
                sim_params,
                verbose=verbose,
                objective_mode=objective_mode,
                user_weights=user_weights,
                n_kl_overrides=candidate_targets,
                users_to_update=update_users,
                dual_state=current_dual_state,
                max_epochs=int(sim_params["max_epochs"]),
            )
            candidate_history = solve_result["history"]
            feasible, candidate_rates, remaining_infeasible_users = _all_committed_bits_feasible(
                system,
                working_F,
                active_users,
                block,
                committed_bits,
                candidate_targets,
            )
            if not feasible:
                _restore_active_block_solver_state(
                    working_F,
                    user_models,
                    model_optimizers,
                    active_users,
                    solver_checkpoint,
                )
                if verbose:
                    infeasible_text = ", ".join(str(int(user_id)) for user_id in remaining_infeasible_users)
                    updated_text = ", ".join(str(int(user_id)) for user_id in update_users)
                    print(
                        f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate):4d} "
                        f"after updating users [{updated_text}] "
                        f"still leaves committed users infeasible [{infeasible_text}] after re-optimization; "
                        "stopping further n_kl reduction for this user in this block."
                    )
                stopped_reduction_users.add(int(k_int))
                continue

            current_n_targets = candidate_targets
            current_rates = candidate_rates
            refinement_history.extend(candidate_history)
            current_dual_state = copy.deepcopy(solve_result["dual_state"])
            plans[k_int]["n_used"] = int(candidate)
            plans[k_int]["R_used"] = float(candidate_rates.get(k_int, current_rates.get(k_int, 0.0)))
            progress_this_epoch = True
            if verbose:
                updated_text = ", ".join(str(int(user_id)) for user_id in update_users)
                print(
                    f"  user={k_int:02d} block={block:02d} accepted smaller n_kl={int(candidate):4d} "
                    f"after fresh re-optimization of users [{updated_text}]."
                )

        if not progress_this_epoch:
            break
        epoch_idx += 1

    for k in active_users:
        k_int = int(k)
        final_n = int(current_n_targets[k_int])
        final_rate = float(system.compute_block_rate(k_int, int(block), final_n, F_override=working_F))
        plans[k_int]["n_used"] = int(final_n)
        plans[k_int]["R_used"] = float(final_rate if committed_bits.get(k_int, 0) > 0 else 0.0)

    final_dual_state = (
        current_dual_state
        if current_dual_state is not None
        else {
            "lambda_rate": {
                int(k): float(sim_params.get("initial_lambda_rate_constraint", 0.1))
                for k in active_users
            },
            "lambda_power_block": float(sim_params.get("initial_lambda_power_constraint", 0.01)),
        }
    )
    return plans, refinement_history, final_dual_state


def optimize_precoders_for_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    sim_params: dict[str, Any],
    verbose: bool = True,
    objective_mode: str = UNWEIGHTED_SUM_RATE_MODE,
    user_weights: dict[int, float] | None = None,
    n_kl_overrides: dict[int, int] | None = None,
    users_to_update: List[int] | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    if len(active_users) == 0:
        return history

    weights = user_weights or {int(k): 1.0 for k in active_users}
    shared_bs_scope = _user_models_output_full_bs_precoder(user_models)
    update_users = [int(k) for k in (active_users if shared_bs_scope else (users_to_update or active_users))]
    print_every = max(1, int(sim_params.get("print_every_epoch", 1)))
    tol = float(sim_params.get("precoder_tol", 1e-4))
    canonical_mode = resolve_objective_mode_alias(objective_mode)
    if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
        objective_label = REMAINING_BITS_WEIGHTED_SUM_RATE_MODE
    elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
        objective_label = "blended_objective"
    else:
        objective_label = UNWEIGHTED_SUM_RATE_MODE
    network_weight_beta = float(sim_params.get("network_rate_weight", 0.15))

    for epoch_idx in range(int(sim_params["max_precoder_epochs"])):
        before_beams = {k: np.array(working_F[k][block], copy=True) for k in update_users}

        if shared_bs_scope:
            block_precoders = _optimize_shared_block_precoders(
                system,
                working_F,
                active_users,
                weights,
                int(block),
                sim_params,
                objective_mode,
                user_models[0],
                model_optimizers[0],
                n_kl_overrides,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(block_precoders[int(k)], dtype=np.complex128)
            system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])
        else:
            for k in update_users:
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
                    n_kl_overrides,
                )
                working_F[k][block] = np.array(beam_k, copy=True)
                system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            n_k = _resolve_user_n_kl(system, int(k), n_kl_overrides)
            rate = float(system.compute_block_rate(int(k), int(block), n_k, F_override=working_F))
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
                "epoch": epoch_idx + 1,
                "active_users": int(len(active_users)),
                "updated_users": int(len(update_users)),
                "user_ids": [int(k) for k in active_users],
                "updated_user_ids": [int(k) for k in update_users],
                "user_n_kl": [_resolve_user_n_kl(system, int(k), n_kl_overrides) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [float(weights.get(int(k), 1.0)) for k in active_users],
                "max_precoder_delta": float(delta),
                "sum_rate": total_rate,
                REMAINING_BITS_WEIGHTED_SUM_RATE_MODE: weighted_total,
                "weighted_sum_rate": weighted_total,
                "blended_objective": blended_total,
                "objective_mode": canonical_mode,
            }
        )
        if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
            objective_value = weighted_total
        elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
            objective_value = blended_total
        else:
            objective_value = total_rate
        if verbose and (((epoch_idx + 1) % print_every) == 0 or epoch_idx == 0 or delta <= tol):
            print(
                f"[Block {block:02d} | Epoch {epoch_idx + 1:03d}] "
                f"active_users={len(active_users)} "
                f"updated_users={len(update_users)} "
                f"{objective_label}={objective_value:.4f} "
                f"sum_rate={total_rate:.4f} "
                f"max_delta={delta:.6e}"
            )
        if delta <= tol:
            break

    return history


def optimize_precoders_for_block_constrained(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    *,
    verbose: bool = True,
    objective_mode: str = UNWEIGHTED_SUM_RATE_MODE,
    user_weights: dict[int, float] | None = None,
    n_kl_overrides: dict[int, int] | None = None,
    users_to_update: List[int] | None = None,
    dual_state: dict[str, dict[int, float]] | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    history: list[dict[str, float]] = []
    if len(active_users) == 0:
        return {
            "history": history,
            "dual_state": {"lambda_rate": {}, "lambda_power_block": 0.0},
            "solve_status": "empty",
        }

    weights = user_weights or {int(k): 1.0 for k in active_users}
    shared_bs_scope = _user_models_output_full_bs_precoder(user_models)
    update_users = [int(k) for k in (active_users if shared_bs_scope else (users_to_update or active_users))]
    print_every = max(1, int(sim_params.get("print_every_epoch", 1)))
    max_epochs = max(1, int(max_epochs if max_epochs is not None else sim_params["max_epochs"]))
    lr_rate = float(sim_params.get("lr_rate_constraint", 1e-2))
    lr_power = float(sim_params.get("lr_power_constraint", 1e-3))
    kkt_primal_tol = float(sim_params.get("kkt_primal_tol", 1e-5))
    kkt_complementarity_tol = float(sim_params.get("kkt_complementarity_tol", 1e-5))
    kkt_stationarity_tol = float(sim_params.get("kkt_stationarity_tol", 1e-4))
    canonical_mode = resolve_objective_mode_alias(objective_mode)

    if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
        objective_label = REMAINING_BITS_WEIGHTED_SUM_RATE_MODE
    elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
        objective_label = "blended_objective"
    else:
        objective_label = UNWEIGHTED_SUM_RATE_MODE

    lambda_rate = {
        int(k): float((dual_state or {}).get("lambda_rate", {}).get(int(k), sim_params.get("initial_lambda_rate_constraint", 0.1)))
        for k in active_users
    }
    lambda_power_block = float(
        (dual_state or {}).get("lambda_power_block", sim_params.get("initial_lambda_power_constraint", 0.01))
    )

    best_primal_residual = float("inf")
    best_feasible_objective = -float("inf")
    solve_status = "max_epochs_reached"
    best_primal_state = _capture_active_block_solver_state(
        working_F,
        user_models,
        model_optimizers,
        active_users,
        lambda_rate,
        lambda_power_block,
    )
    best_feasible_state: dict[str, Any] | None = None

    for epoch_idx in range(max_epochs):
        before_beams = {k: np.array(working_F[k][block], copy=True) for k in update_users}

        if shared_bs_scope:
            shared_optimizer = model_optimizers[0]
            for _ in range(max(1, int(sim_params.get("user_update_steps", 1)))):
                shared_optimizer.zero_grad()
                shared_step_state = _evaluate_constrained_block_state(
                    system,
                    working_F,
                    user_models,
                    active_users,
                    active_users,
                    block,
                    requested_bits,
                    sim_params,
                    objective_mode,
                    weights,
                    lambda_rate,
                    lambda_power_block,
                    n_kl_overrides=n_kl_overrides,
                )
                shared_step_state["loss"].backward()
                shared_optimizer.step()
            block_precoders = _infer_shared_block_precoders_numpy(
                system,
                user_models[0],
                int(block),
                active_users,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(block_precoders[int(k)], dtype=np.complex128)
            system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])
            shared_optimizer.zero_grad()
        else:
            for k in update_users:
                beam_k = _optimize_user_block_precoder_constrained(
                    system,
                    working_F,
                    user_models,
                    active_users,
                    requested_bits,
                    weights,
                    k,
                    block,
                    sim_params,
                    objective_mode,
                    user_models[int(k)],
                    model_optimizers[int(k)],
                    lambda_rate,
                    lambda_power_block,
                    n_kl_overrides,
                )
                working_F[int(k)][block] = np.array(beam_k, copy=True)
                system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])

            for k in update_users:
                model_optimizers[int(k)].zero_grad()
        state = _evaluate_constrained_block_state(
            system,
            working_F,
            user_models,
            active_users,
            active_users if shared_bs_scope else update_users,
            block,
            requested_bits,
            sim_params,
            objective_mode,
            weights,
            lambda_rate,
            lambda_power_block,
            n_kl_overrides=n_kl_overrides,
        )
        state["loss"].backward()

        rate_gaps = {int(k): float(state["rate_gap"][int(k)].detach().cpu()) for k in active_users}
        rate_violations = {int(k): float(state["rate_violation_pos"][int(k)].detach().cpu()) for k in active_users}
        block_power_gap = float(state["block_power_gap"].detach().cpu())
        block_power_violation = float(state["block_power_violation_pos"].detach().cpu())
        exact_feasible = (
            all(float(rate_gaps[int(k)]) <= 0.0 for k in active_users)
            and float(block_power_gap) <= 0.0
        )
        r_p = max(
            max(rate_violations.values(), default=0.0),
            block_power_violation,
        )
        r_c = 0.0
        for k in active_users:
            k_int = int(k)
            r_c = max(
                r_c,
                abs(float(lambda_rate.get(k_int, 0.0)) * rate_violations[k_int]),
            )
        r_c = max(r_c, abs(float(lambda_power_block) * block_power_violation))
        user_rates = [float(state["rates"][int(k)].detach().cpu()) for k in active_users]
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            signal_power, interference_power, _, sinr_db = _compute_user_link_budget(system, working_F, int(k), int(block))
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(_power_to_db(interference_power))
            user_signal_db.append(_power_to_db(signal_power))

        total_rate = float(state["sum_rate"].detach().cpu())
        weighted_total = float(state["weighted_sum_rate"].detach().cpu())
        blended_total = float(state["blended_objective"].detach().cpu())
        block_power = float(state["block_power"].detach().cpu())
        if canonical_mode == REMAINING_BITS_WEIGHTED_SUM_RATE_MODE:
            objective_value = weighted_total
        elif canonical_mode == BLENDED_NETWORK_RATE_MODE:
            objective_value = blended_total
        else:
            objective_value = total_rate
        delta = _block_delta(before_beams, working_F, update_users, block)
        r_s = float(delta)
        history.append(
            {
                "block": int(block),
                "epoch": epoch_idx + 1,
                "active_users": int(len(active_users)),
                "updated_users": int(len(update_users)),
                "user_ids": [int(k) for k in active_users],
                "updated_user_ids": [int(k) for k in update_users],
                "user_n_kl": [_resolve_user_n_kl(system, int(k), n_kl_overrides) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [float(weights.get(int(k), 1.0)) for k in active_users],
                "user_rate_gaps": [float(rate_gaps[int(k)]) for k in active_users],
                "max_precoder_delta": float(delta),
                "sum_rate": total_rate,
                REMAINING_BITS_WEIGHTED_SUM_RATE_MODE: weighted_total,
                "weighted_sum_rate": weighted_total,
                "blended_objective": blended_total,
                "objective_mode": canonical_mode,
                "block_power": block_power,
                "block_power_budget": float(system.block_power_budget),
                "block_power_gap": block_power_gap,
                "block_power_violation": block_power_violation,
                "kkt_primal_residual": float(r_p),
                "kkt_complementarity_residual": float(r_c),
                "kkt_stationarity_residual": float(r_s),
            }
        )

        for k in active_users:
            k_int = int(k)
            lambda_rate[k_int] = max(0.0, float(lambda_rate[k_int]) + lr_rate * rate_violations[k_int])
        lambda_power_block = max(0.0, float(lambda_power_block) + lr_power * block_power_violation)

        if r_p < best_primal_residual:
            best_primal_residual = float(r_p)
            best_primal_state = _capture_active_block_solver_state(
                working_F,
                user_models,
                model_optimizers,
                active_users,
                lambda_rate,
                lambda_power_block,
            )

        if exact_feasible and objective_value >= best_feasible_objective:
            best_feasible_objective = float(objective_value)
            best_feasible_state = _capture_active_block_solver_state(
                working_F,
                user_models,
                model_optimizers,
                active_users,
                lambda_rate,
                lambda_power_block,
            )

        if (
            r_p <= kkt_primal_tol
            and r_c <= kkt_complementarity_tol
            and r_s <= kkt_stationarity_tol
        ):
            solve_status = "kkt_converged"

        if epoch_idx > 0 and r_s <= kkt_stationarity_tol and r_p > kkt_primal_tol:
            solve_status = "stationary_infeasible"

        if verbose and (((epoch_idx + 1) % print_every) == 0 or epoch_idx == 0 or solve_status in {"kkt_converged", "stationary_infeasible"}):
            print(
                format_log_line(
                    "[DL Convergence]",
                    block=int(block),
                    epoch=f"{epoch_idx + 1}/{max_epochs}",
                    active_users=int(len(active_users)),
                    updated_users=int(len(update_users)),
                    objective=float(objective_value),
                    sum_rate=float(total_rate),
                    r_p=float(r_p),
                    r_c=float(r_c),
                    r_s=float(r_s),
                    status=str(solve_status if solve_status != "max_epochs_reached" else "running"),
                )
            )

        if solve_status in {"kkt_converged", "stationary_infeasible"}:
            break

    restored_state = best_feasible_state if best_feasible_state is not None else best_primal_state
    _restore_active_block_solver_state(
        working_F,
        user_models,
        model_optimizers,
        active_users,
        restored_state,
    )
    lambda_rate = dict(restored_state["lambda_rate"])
    lambda_power_block = float(restored_state["lambda_power_block"])
    if best_feasible_state is not None and solve_status == "max_epochs_reached":
        solve_status = "max_epochs_feasible_best"
    elif best_feasible_state is None and solve_status == "max_epochs_reached":
        solve_status = "max_epochs_best_primal"
    if len(history) > 0:
        history[-1]["solve_status"] = str(solve_status)
        history[-1]["solve_segment_end"] = True

    return {
        "history": history,
        "dual_state": {
            "lambda_rate": {int(k): float(v) for k, v in lambda_rate.items()},
            "lambda_power_block": float(lambda_power_block),
        },
        "solve_status": solve_status,
        "best_feasible_found": bool(best_feasible_state is not None),
    }


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
    latency_penalty = float(sim_params.get("latency_penalty_weight", 0.5))

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


def _allocate_fixed_target_for_user_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user: int,
    block: int,
    target_bits: int,
    sim_params: dict[str, Any],
) -> tuple[int, int, float]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])

    R_T = float(system.compute_block_rate(k, l, T_k, F_override=working_F))
    B_max = max(_rate_to_max_bits(T_k, R_T), 0)
    B_used = int(min(max(int(target_bits), 0), B_max))
    if int(B_used) <= 0:
        return 0, int(T_k), 0.0
    chosen_n = int(T_k)
    chosen_R = float(R_T)

    if int(B_used) >= int(target_bits) and int(target_bits) > 0:
        candidate = T_k - n_step
        while candidate >= n_min:
            R_candidate = float(system.compute_block_rate(k, l, candidate, F_override=working_F))
            if (float(target_bits) / float(max(candidate, 1))) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                candidate -= int(n_step)
            else:
                break

    return int(B_used), int(chosen_n), float(chosen_R)


def estimate_initial_latency_from_random_precoders(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    allocation_mode: str,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline_system = DownlinkSystem(system.sc, seed=system.seed)
    # Keep the initial random-precoder baseline tied only to the experiment seed,
    # not to whichever RNG state training happened to leave behind, and not to the
    # selected downlink precoder-net scope.
    baseline_models = _build_user_precoder_models(
        baseline_system,
        init_seed=int(system.seed),
        model_scope=_initial_baseline_model_scope(),
    )
    remaining = np.asarray(baseline_system.B, dtype=int).copy()
    n_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    B_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    R_plan: List[List[float]] = [[] for _ in range(baseline_system.K)]
    working_F = baseline_system.clone_precoders()
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
        _refresh_block_precoders_from_models(
            baseline_system,
            working_F,
            baseline_models,
            active_users,
            block,
        )
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


def _estimate_initial_latency_from_random_precoders_fixed_block_targets(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline_system = DownlinkSystem(system.sc, seed=system.seed)
    baseline_models = _build_user_precoder_models(
        baseline_system,
        init_seed=int(system.seed),
        model_scope=_initial_baseline_model_scope(),
    )
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])
    n_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    B_plan: List[List[int]] = [[] for _ in range(baseline_system.K)]
    R_plan: List[List[float]] = [[] for _ in range(baseline_system.K)]
    working_F = baseline_system.clone_precoders()
    skipped_blocks_per_user = [0 for _ in range(baseline_system.K)]

    for block in range(num_blocks):
        active_users = [k for k in range(baseline_system.K) if int(block_targets[k, block]) > 0]
        for k in active_users:
            _ensure_user_block(baseline_system, working_F, k, block, use_previous_as_template=False)
        _refresh_block_precoders_from_models(
            baseline_system,
            working_F,
            baseline_models,
            active_users,
            block,
        )

        for k in active_users:
            target_bits = int(block_targets[k, block])
            B_used, n_used, R_used = _allocate_fixed_target_for_user_block(
                baseline_system,
                working_F,
                int(k),
                int(block),
                int(target_bits),
                sim_params,
            )
            if int(B_used) <= 0:
                _zero_block_precoder(baseline_system, working_F, int(k), int(block))
            n_plan[int(k)].append(int(n_used))
            B_plan[int(k)].append(int(B_used))
            R_plan[int(k)].append(float(R_used))
            if int(B_used) <= 0:
                skipped_blocks_per_user[int(k)] += 1

    initial_F = _expand_precoders_for_plan(baseline_system, working_F, n_plan)
    baseline_system.apply_solution(initial_F, n_plan)
    latency = baseline_system.latency.tolist()
    return (
        latency,
        {
            "n_kl": n_plan,
            "B_kl": B_plan,
            "R_alloc": R_plan,
            "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
            "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
            "block_bit_targets": block_targets.tolist(),
        },
        _collect_interference_diagnostics(baseline_system),
    )


def estimate_initial_latency_from_random_precoders_for_scenario(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    if str(scenario["mode"]) == FIXED_BLOCK_TARGETS_MODE:
        return _estimate_initial_latency_from_random_precoders_fixed_block_targets(
            system,
            sim_params,
            scenario,
        )
    return estimate_initial_latency_from_random_precoders(
        system,
        sim_params,
        allocation_mode="greedy",
    )


def _run_safe_sweep(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool,
    method_name: str,
    objective_mode: str,
    allocation_mode: str,
    weight_strategy: str = "remaining_bits",
) -> dict[str, Any]:
    objective_mode = resolve_objective_mode_alias(objective_mode)
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = estimate_initial_latency_from_random_precoders(
        system,
        sim_params,
        allocation_mode="greedy",
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="payload_completion",
                method="convergence",
            )
        )
    user_models = _build_user_precoder_models(system, model_scope=model_scope)
    model_optimizers = _build_user_model_optimizers(
        user_models,
        lr=float(sim_params.get("user_update_lr", 5e-3)),
    )

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: List[List[int]] = [[] for _ in range(system.K)]
    B_plan: List[List[int]] = [[] for _ in range(system.K)]
    R_plan: List[List[float]] = [[] for _ in range(system.K)]
    working_F: List[List[np.ndarray]] = system.clone_precoders()
    epoch_history: list[dict[str, float]] = []
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
        _refresh_block_precoders_from_models(
            system,
            working_F,
            user_models,
            active_users,
            block,
        )
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
                format_log_line(
                    "[DL Convergence Block]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    remaining_bits=int(np.sum(remaining)),
                    objective=str(objective_mode),
                )
            )
            if objective_uses_user_weights(objective_mode):
                weights_text = ", ".join(f"u{k}={queue_weights[k]:.3f}" for k in active_users)
                print(f"    weight_strategy={weight_strategy} | user_weights: {weights_text}")

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        skipped_user_rates: dict[int, float] = {}
        block_history: list[dict[str, float]] = []
        block_dual_state: dict[str, dict[int, float]] | None = None
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
            requested_bits_block = {int(k): int(remaining[int(k)]) for k in transmit_users}
            solve_result = optimize_precoders_for_block_constrained(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                verbose=verbose,
                objective_mode=objective_mode,
                user_weights=transmit_weights,
                dual_state=block_dual_state,
                max_epochs=int(sim_params["max_epochs"]),
            )
            current_history = solve_result["history"]
            block_dual_state = copy.deepcopy(solve_result["dual_state"])
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
                        "[DL Convergence Block]",
                        block=int(block),
                        skipped_users=[int(k) for k in infeasible_users],
                        action="reoptimize_remaining",
                    )
                )
        final_plans: dict[int, dict[str, Any]] = {}
        if len(transmit_users) > 0:
            transmit_weights = {int(k): float(queue_weights.get(int(k), 1.0)) for k in transmit_users}
            requested_bits_block = {int(k): int(remaining[int(k)]) for k in transmit_users}
            final_plans, refinement_history, block_dual_state = _reduce_blocklengths_with_reoptimization(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                objective_mode=objective_mode,
                user_weights=transmit_weights,
                verbose=verbose,
                dual_state=block_dual_state,
            )
            block_history.extend(refinement_history)
        epoch_history.extend(block_history)

        block_bits = 0
        for k in active_users:
            queue_weight = float(queue_weights.get(int(k), 1.0))
            if int(k) in skipped_users:
                _zero_block_precoder(system, working_F, k, block)
                skipped_rate = float(skipped_user_rates.get(int(k), 0.0))
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
                        format_log_line(
                            "[DL Convergence Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                            n_kl=int(system.T[k]),
                            achieved_rate=float(skipped_rate),
                        )
                    )
                continue

            plan = final_plans.get(int(k), {"B_used": 0, "n_used": int(system.T[k]), "R_used": 0.0})
            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            R_used = float(plan["R_used"])
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
                    format_log_line(
                        "[DL Convergence Allocation]",
                        user=int(k),
                        block=int(block),
                        served_bits=int(B_used),
                        n_kl=int(n_used),
                        required_rate=float(required_rate),
                        achieved_rate=float(R_used),
                        rate_margin=float(rate_margin),
                    )
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
                format_log_line(
                    "[DL Convergence Block]",
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
    return {
        "method_name": method_name,
        "objective_mode": objective_display_name(objective_mode),
        "allocation_mode": allocation_mode,
        "weight_strategy": objective_weight_strategy_name(objective_mode, weight_strategy),
        "precoder_parameterization": _downlink_precoder_parameterization(model_scope),
        "downlink_precoder_net_scope": str(model_scope),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            model_scope=model_scope,
            context_k=int(system.K),
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
        ),
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
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
    }


def _run_safe_sweep_fixed_block_targets(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool,
    method_name: str,
    objective_mode: str,
) -> dict[str, Any]:
    objective_mode = resolve_objective_mode_alias(objective_mode)
    scenario = build_experiment_scenario(system.sc, sim_params, seed=int(system.seed))
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])

    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = _estimate_initial_latency_from_random_precoders_fixed_block_targets(
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
                method="convergence",
            )
        )
    model_scope = resolve_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    user_models = _build_user_precoder_models(system, model_scope=model_scope)
    model_optimizers = _build_user_model_optimizers(
        user_models,
        lr=float(sim_params.get("user_update_lr", 5e-3)),
    )

    n_plan: List[List[int]] = [[] for _ in range(system.K)]
    B_plan: List[List[int]] = [[] for _ in range(system.K)]
    R_plan: List[List[float]] = [[] for _ in range(system.K)]
    working_F: List[List[np.ndarray]] = system.clone_precoders()
    epoch_history: list[dict[str, float]] = []
    outer_history: list[dict[str, float]] = []
    rate_points: list[dict[str, float]] = []
    skipped_blocks_per_user = [0 for _ in range(system.K)]

    for block in range(num_blocks):
        active_users = [k for k in range(system.K) if int(block_targets[k, block]) > 0]
        for k in active_users:
            _ensure_user_block(system, working_F, k, block)
        _refresh_block_precoders_from_models(
            system,
            working_F,
            user_models,
            active_users,
            block,
        )
        queue_weights = {int(k): 1.0 for k in active_users}
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Block]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    target_bits=int(np.sum(block_targets[:, block])),
                    objective=str(objective_mode),
                )
            )

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        skipped_user_rates: dict[int, float] = {}
        block_history: list[dict[str, float]] = []
        block_dual_state: dict[str, dict[int, float]] | None = None
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
            requested_bits_block = {
                int(k): int(block_targets[int(k), block])
                for k in transmit_users
            }
            solve_result = optimize_precoders_for_block_constrained(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                verbose=verbose,
                objective_mode=objective_mode,
                user_weights=queue_weights,
                dual_state=block_dual_state,
                max_epochs=int(sim_params["max_epochs"]),
            )
            current_history = solve_result["history"]
            block_dual_state = copy.deepcopy(solve_result["dual_state"])
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
                    f"  fixed-target block={block:02d} zero-service users {infeasible_users}; "
                    "re-optimizing remaining transmitters."
                )
        final_plans: dict[int, dict[str, Any]] = {}
        if len(transmit_users) > 0:
            requested_bits_block = {
                int(k): int(block_targets[int(k), block])
                for k in transmit_users
            }
            uniform_weights = {int(k): 1.0 for k in transmit_users}
            final_plans, refinement_history, block_dual_state = _reduce_blocklengths_with_reoptimization(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                objective_mode=objective_mode,
                user_weights=uniform_weights,
                verbose=verbose,
                dual_state=block_dual_state,
            )
            block_history.extend(refinement_history)
        epoch_history.extend(block_history)

        block_bits = 0
        block_unserved_bits = 0
        for k in active_users:
            target_bits = int(block_targets[int(k), block])
            if int(k) in skipped_users:
                _zero_block_precoder(system, working_F, k, block)
                R_zero = float(skipped_user_rates.get(int(k), 0.0))
                B_plan[int(k)].append(0)
                n_plan[int(k)].append(int(system.T[k]))
                R_plan[int(k)].append(float(R_zero))
                block_unserved_bits += int(target_bits)
                skipped_blocks_per_user[int(k)] += 1
                rate_points.append(
                    {
                        "user": int(k),
                        "block": int(block),
                        "n_kl": int(system.T[k]),
                        "B_kl": 0,
                        "target_bits": int(target_bits),
                        "unserved_bits": int(target_bits),
                        "required_rate": 0.0,
                        "achieved_rate": float(R_zero),
                        "rate_margin": float(R_zero),
                        "queue_weight": 1.0,
                        "skipped": True,
                        "partially_served": False,
                    }
                )
                if verbose:
                    print(
                        format_log_line(
                            "[DL Convergence Allocation]",
                            user=int(k),
                            block=int(block),
                            status="zero_service",
                            target_bits=int(target_bits),
                            n_kl=int(system.T[k]),
                        )
                    )
                continue

            plan = final_plans.get(int(k), {"B_used": 0, "n_used": int(system.T[k]), "R_used": 0.0})
            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            R_used = float(plan["R_used"])
            B_plan[int(k)].append(int(B_used))
            n_plan[int(k)].append(int(n_used))
            R_plan[int(k)].append(float(R_used))
            block_bits += int(B_used)
            unserved_bits = max(int(target_bits) - int(B_used), 0)
            block_unserved_bits += int(unserved_bits)
            if int(B_used) <= 0:
                skipped_blocks_per_user[int(k)] += 1

            required_rate = float(B_used) / float(max(int(n_used), 1)) if int(B_used) > 0 else 0.0
            rate_margin = float(R_used) - required_rate
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "target_bits": int(target_bits),
                    "unserved_bits": int(unserved_bits),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": rate_margin,
                    "queue_weight": 1.0,
                    "skipped": bool(int(B_used) <= 0),
                    "partially_served": bool(0 < int(B_used) < int(target_bits)),
                }
            )
            if verbose:
                status = "partial" if 0 < int(B_used) < int(target_bits) else "full"
                print(
                    format_log_line(
                        "[DL Convergence Allocation]",
                        user=int(k),
                        block=int(block),
                        target_bits=int(target_bits),
                        served_bits=int(B_used),
                        unserved_bits=int(unserved_bits),
                        n_kl=int(n_used),
                        achieved_rate=float(R_used),
                        status=str(status),
                    )
                )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(sum(1 for k in active_users if int(B_plan[int(k)][-1]) > 0)),
                "skipped_users": int(sum(1 for k in active_users if int(B_plan[int(k)][-1]) <= 0)),
                "allocated_bits": int(block_bits),
                "target_bits": int(np.sum(block_targets[:, block])),
                "unserved_bits": int(block_unserved_bits),
                "future_target_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "remaining_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "feasible_users": int(block_eval["feasible_count"]),
                "min_max_bits": int(block_eval["min_max_bits"]),
                "queue_weights": {int(k): 1.0 for k in active_users},
                "final_precoder_delta": float(block_history[-1]["max_precoder_delta"]) if block_history else 0.0,
            }
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Block]",
                    block=int(block),
                    status="complete",
                    served_bits=int(block_bits),
                    unserved_bits=int(block_unserved_bits),
                )
            )

    final_F = _expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = _collect_interference_diagnostics(system)
    return {
        "method_name": method_name,
        "objective_mode": objective_display_name(objective_mode),
        "allocation_mode": "fixed_block_targets",
        "weight_strategy": objective_weight_strategy_name(objective_mode, "uniform_active_user_weight"),
        "precoder_parameterization": _downlink_precoder_parameterization(model_scope),
        "downlink_precoder_net_scope": str(model_scope),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            model_scope=model_scope,
            context_k=int(system.K),
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
        ),
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
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def optimize_downlink_safe_sweep(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    objective_mode = resolve_convergence_objective_mode(sim_params)
    scenario = build_experiment_scenario(system.sc, sim_params, seed=int(system.seed))
    if str(scenario["mode"]) == FIXED_BLOCK_TARGETS_MODE:
        return _run_safe_sweep_fixed_block_targets(
            system,
            sim_params,
            verbose=verbose,
            method_name="convergence_per_epoch_baseline",
            objective_mode=objective_mode,
        )
    return _run_safe_sweep(
        system,
        sim_params,
        verbose=verbose,
        method_name="convergence_per_epoch_baseline",
        objective_mode=objective_mode,
        allocation_mode="greedy",
        weight_strategy="remaining_bits",
    )


def optimize_downlink_convergence_epoch(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    return optimize_downlink_safe_sweep(system, sim_params, verbose=verbose)
