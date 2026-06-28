import copy
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from UplinkSystem import UplinkSystem
from experiment_scenarios import FIXED_BLOCK_TARGETS_MODE, PAYLOAD_COMPLETION_MODE, build_experiment_scenario
from uplink_rate_model import build_uplink_rate_covariance


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOG2E_SQ = float(np.log2(np.e) ** 2)


def clone_nested_arrays(nested: Sequence[Sequence[np.ndarray]]) -> List[List[np.ndarray]]:
    return [[np.array(block, copy=True) for block in user_blocks] for user_blocks in nested]


def restore_channels(uplinksystem: UplinkSystem, original_H: Sequence[Sequence[np.ndarray]]) -> None:
    uplinksystem.H = clone_nested_arrays(original_H)


def normalize_system_channels(uplinksystem: UplinkSystem) -> List[Tuple[np.complex64, np.float64]]:
    norm_stats: List[Tuple[np.complex64, np.float64]] = []
    for k in range(int(uplinksystem.K)):
        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        mean = np.mean(H_user)
        var = np.mean(np.abs(H_user - mean) ** 2) + 1e-12
        uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var))
        norm_stats.append((mean, var))
    return norm_stats


def apply_norm_stats_to_system(
    uplinksystem: UplinkSystem,
    norm_stats: Sequence[Tuple[np.complex64, np.float64]],
) -> None:
    for k in range(int(uplinksystem.K)):
        mean, var = norm_stats[k]
        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var + 1e-12))


def _to_complex_numpy(F) -> np.ndarray:
    if isinstance(F, np.ndarray):
        return F.astype(np.complex128, copy=False)
    if hasattr(F, "detach"):
        return F.detach().cpu().numpy().astype(np.complex128, copy=False)
    return np.asarray(F, dtype=np.complex128)


def collect_uplink_interference_diagnostics(uplinksystem: UplinkSystem) -> dict:
    K = int(uplinksystem.K)
    max_blocks = max((len(v) for v in uplinksystem.n_kl), default=0)
    signal = np.full((K, max_blocks), np.nan, dtype=float)
    total_interference = np.full((K, max_blocks), np.nan, dtype=float)
    noise = np.full((K, max_blocks), np.nan, dtype=float)
    sinr_db = np.full((K, max_blocks), np.nan, dtype=float)
    pairwise_block = np.full((max_blocks, K, K), np.nan, dtype=float)
    pairwise_sum = np.zeros((K, K), dtype=float)
    pairwise_inr_sum = np.zeros((K, K), dtype=float)
    pairwise_count = np.zeros((K, K), dtype=float)

    for k in range(K):
        for l in range(len(uplinksystem.n_kl[k])):
            H_k = np.asarray(uplinksystem.H[k][l], dtype=np.complex128)
            F_k = np.asarray(uplinksystem.F[k][l], dtype=np.complex128)
            X_k = np.asarray(uplinksystem.X[k][l], dtype=np.complex128)
            desired = H_k @ (F_k @ X_k)

            p_sig = float(np.mean(np.abs(desired) ** 2))
            if l < len(uplinksystem.N[k]):
                p_noise = float(np.mean(np.abs(uplinksystem.N[k][l]) ** 2))
            else:
                p_noise = float(uplinksystem.sigma2[k])
            p_interf = 0.0

            for j in range(K):
                if j == k:
                    continue
                if len(uplinksystem.H[j]) == 0 or len(uplinksystem.F[j]) == 0 or len(uplinksystem.X[j]) == 0:
                    continue

                lj = uplinksystem._resolve_metric_block_index(j, l, require_x=True)
                H_j = np.asarray(uplinksystem.H[j][lj], dtype=np.complex128)
                if H_j.shape[0] != H_k.shape[0]:
                    raise ValueError(
                        "Uplink interference diagnostics require a common BS receive dimension NR across users. "
                        f"Victim user {k} block {l} has NR={H_k.shape[0]}, "
                        f"interferer user {j} block {lj} has NR={H_j.shape[0]}."
                    )
                F_j = np.asarray(uplinksystem.F[j][lj], dtype=np.complex128)
                X_j = np.asarray(uplinksystem.X[j][lj], dtype=np.complex128)
                interference = H_j @ (F_j @ X_j)
                coupling = float(np.mean(np.abs(interference) ** 2))
                pairwise_block[l, k, j] = coupling
                pairwise_sum[k, j] += coupling
                pairwise_inr_sum[k, j] += coupling / max(p_noise, 1e-30)
                pairwise_count[k, j] += 1.0
                p_interf += coupling

            signal[k, l] = p_sig
            total_interference[k, l] = p_interf
            noise[k, l] = p_noise
            sinr_db[k, l] = float(
                10.0 * np.log10(max(p_sig / max(p_interf + p_noise, 1e-30), 1e-30))
            )

    avg_pairwise_power = np.divide(
        pairwise_sum,
        pairwise_count,
        out=np.full_like(pairwise_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_lin = np.divide(
        pairwise_inr_sum,
        pairwise_count,
        out=np.full_like(pairwise_inr_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_db = 10.0 * np.log10(np.maximum(avg_pairwise_inr_lin, 1e-30))
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
        "blocks_per_user": [len(v) for v in uplinksystem.n_kl],
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


def apply_training_solution(
    uplink_system: UplinkSystem,
    n_star: Sequence[Sequence[int]],
    F_star: Sequence[Sequence],
) -> None:
    K = int(uplink_system.K)
    n_kl_new: List[List[int]] = []
    F_new: List[List[np.ndarray]] = []

    for k in range(K):
        nk = list(map(int, n_star[k]))
        if any(int(n_kl) <= 0 for n_kl in nk):
            raise ValueError(f"Uplink blocklengths must be strictly positive for user {k}, got {nk}.")
        n_kl_new.append(nk)

        Lk = len(nk)
        if len(F_star[k]) == 0:
            Fk = list(uplink_system.F[k])[:Lk]
            if len(Fk) < Lk and len(Fk) > 0:
                Fk = Fk + [np.array(Fk[-1], copy=True)] * (Lk - len(Fk))
            F_new.append(Fk)
            continue

        Fk = [_to_complex_numpy(F) for F in F_star[k]]
        if len(Fk) < Lk:
            Fk = Fk + [np.array(Fk[-1], copy=True)] * (Lk - len(Fk))
        else:
            Fk = Fk[:Lk]
        F_new.append(Fk)

    uplink_system.update_system(F=F_new, n_kl=n_kl_new, regenerate_noise_on_nl_change=True)


def ensure_blocks_up_to(uplinksystem: UplinkSystem, block_idx: int) -> None:
    for k in range(int(uplinksystem.K)):
        while len(uplinksystem.H[k]) <= int(block_idx):
            uplinksystem.add_block(k)


def estimate_initial_random_precoder_schedule(
    system_params: dict,
    sim_cfg: dict,
    *,
    seed: int,
) -> dict:
    from Optimizer_per_block import _compute_R_fbl_np

    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    max_total_blocks = int(sim_cfg.get("max_total_blocks", 256))

    remaining_bits = [int(v) for v in baseline_system.B]
    initial_n_kl: List[List[int]] = [[] for _ in range(K)]
    initial_B_kl: List[List[int]] = [[] for _ in range(K)]
    initial_R_fbl: List[List[float]] = [[] for _ in range(K)]
    initial_F: List[List[np.ndarray]] = [[] for _ in range(K)]

    block = 0
    while any(bits > 0 for bits in remaining_bits):
        if block >= max_total_blocks:
            raise RuntimeError(
                f"Initial random-precoder uplink schedule hit max_total_blocks={max_total_blocks} "
                f"with remaining bits {remaining_bits}."
            )

        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            if remaining_bits[k] <= 0:
                continue

            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )

            B_try = int(remaining_bits[k])
            B_used = 0
            R_T = _compute_R_fbl_np(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
            )
            for _ in range(12):
                if (B_try / float(max(T_ref, 1))) <= R_T:
                    B_used = int(B_try)
                    break
                B_new = int(np.floor(float(T_ref) * float(R_T)))
                B_new = max(0, min(B_new, B_try))
                if B_new == B_try or B_new <= 0:
                    B_used = 0
                    break
                B_try = B_new

            best_n = int(T_ref)
            best_R = float(R_T)
            if B_used > 0:
                candidate_n = int(T_ref) - int(n_kl_step)
                while candidate_n >= int(n_kl_min):
                    R_candidate = _compute_R_fbl_np(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        candidate_n,
                        noise_plus_interference_cov,
                    )
                    if (float(B_used) / float(max(candidate_n, 1))) <= R_candidate:
                        best_n = int(candidate_n)
                        best_R = float(R_candidate)
                        candidate_n -= int(n_kl_step)
                    else:
                        break

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R))
            initial_F[k].append(np.array(F_kl, copy=True))
            remaining_bits[k] = max(0, int(remaining_bits[k]) - int(B_used))

        block += 1

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    initial_n = [int(sum(user_n)) for user_n in initial_n_kl]
    initial_latency = [float(v) for v in baseline_system.latency]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(n_kl, 1))
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": initial_latency,
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
    }


def _estimate_initial_random_precoder_schedule_fixed_block_targets(
    system_params: dict,
    sim_cfg: dict,
    *,
    seed: int,
    scenario: dict,
) -> dict:
    from Optimizer_per_block import _compute_R_fbl_np

    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    block_targets = np.asarray(scenario["block_bit_targets"], dtype=int)
    num_blocks = int(scenario["num_blocks"])

    initial_n_kl: List[List[int]] = [[] for _ in range(K)]
    initial_B_kl: List[List[int]] = [[] for _ in range(K)]
    initial_R_fbl: List[List[float]] = [[] for _ in range(K)]
    initial_F: List[List[np.ndarray]] = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )
            R_T = _compute_R_fbl_np(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
            )
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            best_n = int(T_ref)
            best_R = float(R_T)

            if int(B_used) >= int(target_bits) and int(target_bits) > 0:
                candidate_n = int(T_ref) - int(n_kl_step)
                while candidate_n >= int(n_kl_min):
                    R_candidate = _compute_R_fbl_np(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        candidate_n,
                        noise_plus_interference_cov,
                    )
                    if (float(target_bits) / float(max(candidate_n, 1))) <= R_candidate:
                        best_n = int(candidate_n)
                        best_R = float(R_candidate)
                        candidate_n -= int(n_kl_step)
                    else:
                        break

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R) if int(B_used) > 0 else 0.0)
            initial_F[k].append(
                np.array(F_kl, copy=True) if int(B_used) > 0 else np.zeros_like(F_kl, dtype=np.complex64)
            )
            if int(B_used) <= 0:
                skipped_blocks_per_user[k] += 1

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    initial_n = [int(sum(int(max(v, 0)) for v in user_n)) for user_n in initial_n_kl]
    initial_latency = [float(v) for v in baseline_system.latency]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(int(n_kl), 1))
            if int(n_kl) > 0 and int(bits) > 0
            else 0.0
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n if initial_n[k] > 0 else 0.0)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": initial_latency,
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": FIXED_BLOCK_TARGETS_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def estimate_initial_random_precoder_schedule_for_scenario(
    system_params: dict,
    sim_cfg: dict,
    *,
    seed: int,
) -> dict:
    scenario = build_experiment_scenario(system_params, sim_cfg, seed=int(seed))
    if str(scenario["mode"]) == FIXED_BLOCK_TARGETS_MODE:
        return _estimate_initial_random_precoder_schedule_fixed_block_targets(
            system_params,
            sim_cfg,
            seed=int(seed),
            scenario=scenario,
        )

    baseline = estimate_initial_random_precoder_schedule(system_params, sim_cfg, seed=int(seed))
    baseline["skipped_blocks_per_user"] = [0 for _ in range(int(system_params["K"]))]
    baseline["scenario_mode"] = PAYLOAD_COMPLETION_MODE
    return baseline


def max_precoder_delta(F_old: Sequence[Sequence], F_new: Sequence[Sequence]) -> float:
    max_delta = 0.0
    K = max(len(F_old), len(F_new))
    for k in range(K):
        old_blocks = F_old[k] if k < len(F_old) else []
        new_blocks = F_new[k] if k < len(F_new) else []
        Lk = max(len(old_blocks), len(new_blocks))
        for l in range(Lk):
            if l >= len(old_blocks) or l >= len(new_blocks):
                max_delta = max(max_delta, 1.0)
                continue
            A = _to_complex_numpy(old_blocks[l])
            B = _to_complex_numpy(new_blocks[l])
            denom = max(float(np.linalg.norm(A, ord="fro")), 1e-12)
            delta = float(np.linalg.norm(A - B, ord="fro") / denom)
            max_delta = max(max_delta, delta)
    return max_delta


def complex_to_ri_parameter(F: np.ndarray, device=DEVICE) -> torch.nn.Parameter:
    arr = np.asarray(F, dtype=np.complex64)
    stacked = np.stack([arr.real, arr.imag], axis=0)
    return torch.nn.Parameter(torch.tensor(stacked, dtype=torch.float32, device=device))


def ri_to_complex_tensor(param: torch.Tensor) -> torch.Tensor:
    return (param[0] + 1j * param[1]).to(torch.complex64)


def project_power_complex_torch(F: torch.Tensor, P: float, eps: float = 1e-12) -> torch.Tensor:
    power = (torch.linalg.norm(F, ord="fro") ** 2).real
    if float(power.detach().cpu()) <= float(P):
        return F
    scale = torch.sqrt(torch.tensor(float(P), device=F.device, dtype=torch.float32) / (power + eps))
    return F * scale.to(F.dtype)


def q_inv_torch(epsilon: float, device=DEVICE) -> torch.Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    p = torch.tensor(1.0 - float(epsilon), device=device, dtype=torch.float64)
    p = torch.clamp(p, 1e-12, 1.0 - 1e-12)
    return normal.icdf(p).to(dtype=torch.float32)


def compute_joint_rates_torch(
    H_list: Sequence[torch.Tensor],
    F_list: Sequence[torch.Tensor],
    sigma2_list: Sequence[float],
    epsilon_list: Sequence[float],
    n_values: Sequence[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    rates: List[torch.Tensor] = []
    capacities: List[torch.Tensor] = []
    dispersions: List[torch.Tensor] = []

    K = len(H_list)
    for k in range(K):
        Hk = H_list[k]
        Fk = F_list[k]
        Nr = Hk.shape[0]
        I = torch.eye(Nr, dtype=torch.complex64, device=Hk.device)
        noise_cov = float(sigma2_list[k]) * I

        for j in range(K):
            if j == k:
                continue
            Hj = H_list[j]
            if Hj.shape[0] != Nr:
                raise ValueError(
                    "Joint SINR optimization assumes a common BS receive dimension NR across users. "
                    f"User {k} has NR={Nr}, user {j} has NR={Hj.shape[0]}."
                )
            HFj = Hj @ F_list[j]
            noise_cov = noise_cov + HFj @ HFj.conj().transpose(1, 0)

        noise_cov = 0.5 * (noise_cov + noise_cov.conj().transpose(1, 0))
        noise_cov = noise_cov + (1e-6 * I)
        HFk = Hk @ Fk
        chol = torch.linalg.cholesky(noise_cov)
        G = torch.linalg.solve(chol, HFk)
        A = G @ G.conj().transpose(1, 0)
        A = 0.5 * (A + A.conj().transpose(1, 0))

        sign, logdet = torch.linalg.slogdet(I + A)
        if torch.any(torch.abs(sign) <= 1e-12):
            raise RuntimeError(f"slogdet sign<=0 while evaluating joint rate for user {k}")

        C = (logdet / np.log(2.0)).real
        eigvals = torch.linalg.eigvalsh(A)
        V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2).real * LOG2E_SQ
        R = C - torch.sqrt(V / float(max(int(n_values[k]), 1))) * q_inv_torch(float(epsilon_list[k]), device=Hk.device)

        capacities.append(C)
        dispersions.append(V)
        rates.append(R.real)

    return rates, capacities, dispersions


def build_single_block_post_training_dict(
    uplinksystem: UplinkSystem,
    norm_stats,
    n_values: Sequence[int],
    F_tensors: Sequence[torch.Tensor],
    rates: Sequence[torch.Tensor | float],
    *,
    loss_history: Sequence[float],
    method_name: str,
    metadata: dict | None = None,
):
    K = int(uplinksystem.K)
    n_star = [[int(n_values[k])] for k in range(K)]
    F_star = [[F_tensors[k].detach().cpu()] for k in range(K)]
    R_star = [[float(rates[k].detach().cpu() if torch.is_tensor(rates[k]) else rates[k])] for k in range(K)]
    L_out = [1] * K

    all_user_block_results = []
    B_used_star = []
    B_kl_star = []

    for k in range(K):
        B_used = max(0, min(int(uplinksystem.B[k]), int(np.floor(int(n_values[k]) * R_star[k][0]))))
        B_used_star.append([B_used])
        B_kl_star.append([B_used])
        Fk = F_tensors[k].detach().cpu()
        all_user_block_results.append([[
            {
                "n_kl": int(n_values[k]),
                "n": int(n_values[k]),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(max(int(n_values[k]), 1)),
                "F": Fk,
                "R_fbl": float(R_star[k][0]),
                "F_power": float((torch.linalg.norm(Fk, ord="fro") ** 2).real.cpu()),
                "lambda_rate": 0.0,
                "lambda_power": 0.0,
                "loss_curve": list(loss_history),
                "method": method_name,
            }
        ]])

    post = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "method_name": method_name,
    }
    if metadata:
        post.update(metadata)
    return post


def nested_int_lists_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return list(map(list, a)) == list(map(list, b))
