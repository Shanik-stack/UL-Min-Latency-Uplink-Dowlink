import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

from precoder_models import (
    DEVICE,
    build_user_precoder_net,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy,
    infer_precoder_torch,
    load_user_precoder_models,
    net_output_to_precoder,
    project_precoder_power,
)
LOG2E_SQ = (np.log2(np.e)) ** 2

# ============================================================
# Utils
# ============================================================
def Q_inv(eps: float, device=DEVICE, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(norm.ppf(1.0 - float(eps)), device=device, dtype=dtype)

def update_lambdas(lambda_rate: float, lambda_power: float,
                   rate_violation_pos: torch.Tensor, power_violation_pos: torch.Tensor,
                   lr_rate: float, lr_power: float):
    r = float(rate_violation_pos.detach().item())
    p = float(power_violation_pos.detach().item())
    lambda_rate = max(0.0, float(lambda_rate) + float(lr_rate) * r)
    lambda_power = max(0.0, float(lambda_power) + float(lr_power) * p)
    return lambda_rate, lambda_power

def project_power(Fmat: torch.Tensor, P: float, eps: float = 1e-12, delta: float = 1e-9) -> torch.Tensor:
    """
    Enforce ||F||_F^2 < P by scaling with a small safety margin.
    delta: fractional margin so final power ≈ (1 - delta)^2 * P
    """
    power = (torch.linalg.norm(Fmat, ord="fro") ** 2).real

    if float(power.detach().cpu()) <= float(P):
        return Fmat

    # Slightly under-scale to guarantee strict inequality
    scale = torch.sqrt(
        torch.tensor(float(P), device=Fmat.device, dtype=torch.float32)
        / (power + eps)
    )

    # scale = scale * (1.0 - delta)

    return Fmat * scale.to(Fmat.dtype)

def print_result(result_list, key: str):
    print(f"\nResult {key}: ", end="")
    for d in result_list[::-1]:
        v = d.get(key, None)
        if torch.is_tensor(v):
            v = float(v.detach().cpu())
        print(v, end=", ")
    print()

# ============================================================
# Precoder MLP
# ============================================================
class UplinkMLP_PrecoderNet(nn.Module):
    def __init__(self, Nr: int, Nt: int, dk: int):
        super().__init__()
        in_dim = 2 * Nr * Nt
        out_dim = 2 * Nt * dk

        h1 = max(512, 8 * out_dim)
        h2 = max(256, 4 * out_dim)
        h3 = max(128, 2 * out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, out_dim),
        )

    def forward(self, H_kl: torch.Tensor) -> torch.Tensor:
        # H_kl: (Nr, Nt) complex64
        H_flat = H_kl.reshape(1, -1)
        x = torch.cat([H_flat.real, H_flat.imag], dim=1)  # (1, 2*Nr*Nt)
        return self.net(x)  # (1, 2*Nt*dk)

# ============================================================
# Lagrangian Loss (finite blocklength rate)
# ============================================================
class LagrangianLoss(nn.Module):
    def __init__(
        self,
        H_kl: torch.Tensor,
        sigma2: float,
        epsilon: float,
        B: float,
        P: float,
        n_kl: int,
        noise_plus_interference_cov: torch.Tensor | None = None,
    ):
        super().__init__()
        self.H_kl = H_kl
        self.sigma2 = float(sigma2)
        self.epsilon = float(epsilon)
        self.B = float(B)
        self.P = float(P)
        self.n_kl = int(n_kl)
        self.noise_plus_interference_cov = noise_plus_interference_cov

    def set_blocklength(self, n_kl: int):
        self.n_kl = int(n_kl)

    def set_payload(self, B: float):
        self.B = float(B)

    def R_fbl(self, Fmat: torch.Tensor) -> torch.Tensor:
        """
        Fmat: (Nt, dk) complex64
        """
        
        H = self.H_kl
        Nr = H.shape[0] 
        I = torch.eye(Nr, dtype=torch.complex64, device=H.device)

        HF = H @ Fmat  # (Nr, dk)

        if self.noise_plus_interference_cov is None:
            noise_cov = self.sigma2 * I
        else:
            noise_cov = self.noise_plus_interference_cov.to(device=H.device, dtype=torch.complex64)

        chol = torch.linalg.cholesky(noise_cov)
        G = torch.linalg.solve(chol, HF)
        A = G @ G.conj().transpose(1, 0)  # whitened effective signal covariance

        sign, logdet = torch.linalg.slogdet(I + A)
        C = logdet / np.log(2.0)  # log2(det(.))
        if not torch.isfinite(A).all():
            bad = (~torch.isfinite(A)).nonzero(as_tuple=False)[:10]
            raise RuntimeError(f"A has NaN/Inf. First bad indices: {bad}")

        # Force exact Hermitian (complex) / symmetric (real)
        A = 0.5 * (A + A.conj().transpose(1, 0))

        eigvals = torch.linalg.eigvalsh(A)  # (Nr,) real
        V = torch.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2) * LOG2E_SQ

        R = C - torch.sqrt(V / float(self.n_kl)) * Q_inv(self.epsilon, device=Fmat.device)
        return R.real

    def forward(self, Fmat: torch.Tensor, lambda_rate: float, lambda_power: float):
        R = self.R_fbl(Fmat)
        F_power = (torch.linalg.norm(Fmat, ord="fro") ** 2).real
        
        

        # constraints (positive => violation)
        rate_violation = (self.B / float(self.n_kl)) - R
        power_violation = F_power - self.P

        # positive-part penalties (keep your original "leaky_relu" style)
        rate_violation_pos = F.leaky_relu(rate_violation)
        power_violation_pos = F.leaky_relu(power_violation)

        loss = (-R
                + float(lambda_rate) * rate_violation_pos
                + float(lambda_power) * power_violation_pos)

        return loss, R, F_power, rate_violation_pos, power_violation_pos

# ============================================================
# Optimize precoder for a fixed n_kl
# ============================================================
def optimize_precoder_for_nl(
    precoder_net: nn.Module,
    loss_fn: LagrangianLoss,
    Nt: int,
    dk: int,
    epochs: int,
    lambda_rate: float,
    lambda_power: float,
    lr_rate: float,
    lr_power: float,
    optimizer: torch.optim.Optimizer,
):
    losses = []

    for epoch in range(int(epochs)):
        Fmat = infer_precoder_torch(precoder_net, loss_fn.H_kl, Nt, dk, loss_fn.P)

        optimizer.zero_grad()
        loss, R, F_power, rv_pos, pv_pos = loss_fn(Fmat, lambda_rate, lambda_power)
        # print("Post F_power: ", F_power)
        feasible = (rv_pos.item() <= 0.0) and (pv_pos.item() <= 0.0)
        if feasible and epoch >= 1000:
            break

        lambda_rate, lambda_power = update_lambdas(
            lambda_rate, lambda_power, rv_pos, pv_pos, lr_rate, lr_power
        )

        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        F_final = infer_precoder_torch(precoder_net, loss_fn.H_kl, Nt, dk, loss_fn.P)
        loss, R, F_power, rv_pos, pv_pos = loss_fn(F_final, lambda_rate, lambda_power)

    return {
        "F": F_final.detach(),
        "lambda_rate": float(lambda_rate),
        "lambda_power": float(lambda_power),
        "R_fbl": float(R.detach().item()),
        "F_power": float(F_power.detach().item()),
        "rate_violation": float(rv_pos.detach().item()),
        "power_violation": float(pv_pos.detach().item()),
        "loss_curve": losses,
    }

# ============================================================
# OptimizeSubBlocklength-Precoder (returns + logs like your older pipeline)
# ============================================================
def optimize_subblocklength_precoder(
    uplinksystem,
    user: int,
    block: int,
    B_rem: int,
    lambda_rate_0: float,
    lambda_power_0: float,
    sim_cfg: dict,
    precoder_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    interference_F_snapshot=None,
):
    print(f"\n========== OptimizeSubBlocklength-Precoder ==========")
    print(f"User: {user}, Block: {block}")
    print(f"Initial B_rem: {B_rem}")
    print("-----------------------------------------------------")

    P = float(uplinksystem.P[user])
    Nr = int(uplinksystem.NR[user])
    Nt = int(uplinksystem.NT[user])
    dk = int(uplinksystem.dk[user])
    sigma2 = float(uplinksystem.sigma2[user])
    epsilon = float(uplinksystem.epsilon[user])

    H_kl = torch.tensor(uplinksystem.H[user][block], dtype=torch.complex64, device=DEVICE)
    noise_plus_interference_cov = torch.tensor(
        uplinksystem.get_interference_plus_noise_covariance(
            user,
            block,
            F_override=interference_F_snapshot,
        ),
        dtype=torch.complex64,
        device=DEVICE,
    )

    T = int(uplinksystem.T[user])
    n_kl_max = int(uplinksystem.T[user])
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    epochs_per_n_kl = int(sim_cfg["epochs_per_n_kl"])
    
    # psuedo_n_kl_max = int(1000*np.sin(1/real_n_kl_max))
    # psuedo_n_kl_max = int(B_rem*1.2)
    # n_kl_max = psuedo_n_kl_max

    lr_net = float(sim_cfg["lr_net"])
    lr_rate = float(sim_cfg["lr_rate_constraint"])
    lr_power = float(sim_cfg["lr_power_constraint"])

    lambda_rate = float(lambda_rate_0)
    lambda_power = float(lambda_power_0)

    loss_fn = LagrangianLoss(
        H_kl=H_kl, sigma2=sigma2, epsilon=epsilon,
        B=float(B_rem), P=float(P), n_kl=n_kl_max,
        noise_plus_interference_cov=noise_plus_interference_cov,
    ).to(DEVICE)

    results = []

    # ------------------------------------------------------------
    # STEP A: Fix n = T feasible by reducing B (like your 2nd code)
    # ------------------------------------------------------------
    print(f"\n--- Step A: Fix n = T = {n_kl_max} and adjust B if needed ---")
    loss_fn.set_blocklength(n_kl_max)
    loss_fn.set_payload(float(B_rem))
    max_payload_reductions = 1
    B_used = None
    B_initial = int(B_rem)
    attempt = 0
    while(1):
        print(f"\nAttempt {attempt+1}: n = {n_kl_max}, B = {int(loss_fn.B)}")
        epochs = epochs_per_n_kl

        out = optimize_precoder_for_nl(
            precoder_net=precoder_net,
            loss_fn=loss_fn,
            Nt=Nt, dk=dk,
            epochs=epochs,
            lambda_rate=lambda_rate, lambda_power=lambda_power,
            lr_rate=lr_rate, lr_power=lr_power,
            optimizer=optimizer
        )

        lambda_rate = out["lambda_rate"]
        lambda_power = out["lambda_power"]

        print(f"R_fbl: {out['R_fbl']}")
        print(f"F_power: {out['F_power']}")
        print(f"Rate violation: {out['rate_violation']}")
        print(f"Power violation: {out['power_violation']}")

        feasible = (out["rate_violation"] <= 0.0) and (out["power_violation"] <= 0.0)
        if feasible:
            print(">>> Feasible at n = T.")
            B_used = int(loss_fn.B)

            results.append({
                "n_kl": int(n_kl_max),
                "n": int(n_kl_max),                 # keep consistent; total n per block = n_kl here
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl_max),
                "F": out["F"],
                "R_fbl": float(out["R_fbl"]),
                "F_power": float(out["F_power"]),
                "lambda_rate": float(lambda_rate),
                "lambda_power": float(lambda_power),
                "loss_curve": out["loss_curve"],
            })
            break

        # if out["power_violation"] > 0.0:
        #     print(">>> Power constraint violated. Cannot fix by reducing B. STOP.")
        #     return [], 0
        elif not feasible and attempt<max_payload_reductions:
            # Reduce B (same policy as your first code: floor(n * R))
            B_update = int(np.floor(n_kl_max * float(out["R_fbl"])))
            B_update_max = max(0, min(B_update, int(loss_fn.B)))
            B_new = B_update_max
            print(f">>> Reducing B from {int(loss_fn.B)} to {B_new}")
            print(f" Reduction B_new = {B_new} = max({0}, min(B_update {B_update} = {n_kl_max} * {float(out['R_fbl'])}, B_try {int(loss_fn.B)})")

            if B_new == int(loss_fn.B) or B_new == 0:
                print(">>> B cannot be reduced further. STOP.")
                return [], 0

            loss_fn.set_payload(float(B_new))
        else:
            break
        attempt+=1

    if B_used is None or B_used <= 0:
        print(">>> No feasible solution at n=T after payload adjustment.")
        return [], 0

    B_rem_after = int(B_initial - B_used)
    print(f"\n>>> Payload accepted at n=T: B_used={B_used}, remaining after this block would be B_rem={B_rem_after}")

    # ------------------------------------------------------------
    # STEP B: Reduce n while keeping B fixed (store all feasible)
    # ------------------------------------------------------------
    print(f"\n--- Step B: Reduce n while keeping B = {B_used} fixed ---")
    n_kl = n_kl_max - n_kl_step

    while n_kl >= n_kl_min:
        if(B_rem_after != 0):
            print(f"\n Not reducing n_kl since not last sub-block (n_kl = {n_kl})")
            break
        print(f"\nTesting n = {n_kl}")
        loss_fn.set_blocklength(n_kl)
        loss_fn.set_payload(float(B_used))

        out = optimize_precoder_for_nl(
            precoder_net=precoder_net,
            loss_fn=loss_fn,
            Nt=Nt, dk=dk,
            epochs=epochs_per_n_kl,
            lambda_rate=lambda_rate, lambda_power=lambda_power,
            lr_rate=lr_rate, lr_power=lr_power,
            optimizer=optimizer
        )

        lambda_rate = out["lambda_rate"]
        lambda_power = out["lambda_power"]

        print(f"R_fbl: {out['R_fbl']}")
        print(f"Rate violation: {out['rate_violation']}")
        print(f"Power violation: {out['power_violation']}")

        feasible = (out["rate_violation"] <= 0.0) and (out["power_violation"] <= 0.0)
        if not feasible:
            print(">>> Not feasible. Stop decreasing n.")
            break

        print(">>> Feasible. Storing result.")
        results.append({
            "n_kl": int(n_kl),
            "n": int(n_kl),
            "B_l": int(B_used),
            "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl),
            "F": out["F"],
            "R_fbl": float(out["R_fbl"]),
            "F_power": float(out["F_power"]),
            "lambda_rate": float(lambda_rate),
            "lambda_power": float(lambda_power),
            "loss_curve": out["loss_curve"],
        })

        n_kl -= n_kl_step

    print("\n>>> Finished OptimizeSubBlocklength-Precoder")
    print_result(results, "n_kl")
    print_result(results, "R_fbl")
    print_result(results, "F_power")
    print_result(results, "lambda_rate")
    print_result(results, "lambda_power")

    return results, int(B_used)

# ============================================================
# Training loop (per user, per block) - same outputs as before
# ============================================================

import numpy as np

def dynamic_subblocklength_precoder_training(
    uplinksystem,
    sim_cfg: dict,
    channel_norm: bool = True,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """
    Training version that is consistent with the TEST logic style:
      - For each user, iteratively allocate blocks until B_rem == 0
      - For each block:
          Step A: ensure feasibility at n=T by reducing B (and training precoder)
          Step B: decrease n_kl with B fixed (and training precoder at each n_kl)
      - Saves per-block trajectories S for plotting.

    Returns:
      post_training_data_dict with:
        - L_out, n_star, F_star, R_star, norm_stats
        - all_user_block_results: list[user][block] = S (trajectory over n_kl)
        - B_used_star: list[user][block] = B_used at n=T (after reduction)
        - B_kl_star:   list[user][block] = bits actually transmitted in that block
    """
    K = int(uplinksystem.K)

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]

    # extra bookkeeping (useful for plots/repro)
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    lambda_rate_0 = float(sim_cfg["initial_lambda_rate_constraint"])
    lambda_power_0 = float(sim_cfg["initial_lambda_power_constraint"])
    user_precoder_models: list[nn.Module] = []

    for k in range(K):
        print(f"\n================ TRAIN USER {k} ================")

        # ---- normalize user channel across all currently available blocks ----
        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        mean = np.mean(H_user)
        var = np.mean(np.abs(H_user - mean) ** 2) + 1e-12
        norm_stats.append((mean, var))

        if channel_norm:
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var))

        # ---- remaining payload ----
        B_rem = int(uplinksystem.B[k])
        ell = 0

        # ensure at least one block exists
        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        user_model = build_user_precoder_net(
            Nr=int(uplinksystem.NR[k]),
            Nt=int(uplinksystem.NT[k]),
            dk=int(uplinksystem.dk[k]),
            device=DEVICE,
        )
        user_optimizer = torch.optim.Adam(user_model.parameters(), lr=float(sim_cfg["lr_net"]))
        user_precoder_models.append(user_model)

        # reset lambdas per user (or keep global if you prefer)
        lambda_rate_user = float(lambda_rate_0)
        lambda_power_user = float(lambda_power_0)

        while B_rem > 0:
            # ensure block exists
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            print(f"\n--- TRAIN User {k}, Block {ell}, B_rem={B_rem} ---")

            # This function already performs:
            #  Step A: reduce B at n=T until feasible (and trains precoder)
            #  Step B: decrease n_kl with fixed B_used (and trains precoder)
            S, B_used = optimize_subblocklength_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=B_rem,
                lambda_rate_0=lambda_rate_user,
                lambda_power_0=lambda_power_user,
                sim_cfg=sim_cfg,
                precoder_net=user_model,
                optimizer=user_optimizer,
                interference_F_snapshot=interference_F_snapshot,
            )

            if len(S) == 0 or B_used <= 0:
                print(f">>> STOP training user {k} at block {ell} (no feasible).")
                break

            # Save trajectory for plotting (your plot_optimization_result_train expects this)
            all_user_block_results[k].append(S)

            # Pick smallest feasible n_kl in this block
            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])

            # Save chosen decisions
            n_star[k].append(n_opt)
            F_star[k].append(F_opt)
            R_star[k].append(R_opt)
            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()


            # Save payload bookkeeping
            B_used_star[k].append(int(B_used))
            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))

            B_rem -= B_kl
            print(f">>> Chosen: n_kl={n_opt}, B_used={B_used}, B_kl={B_kl}, remaining B_rem={B_rem}")

            # advance block if bits remain
            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1

        # write back to system bookkeeping (optional)
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])
        print("Checkingtr F_star")
        print(F_star)

    post_training_data_dict = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "user_model_specs": export_user_model_specs(uplinksystem.NR, uplinksystem.NT, uplinksystem.dk),
        "user_model_states": export_user_model_states(user_precoder_models),
        "precoder_parameterization": "shared_user_channel_to_precoder_mlp",
    }
    return post_training_data_dict

import numpy as np
from scipy.stats import norm

def _Q_inv_np(eps: float) -> float:
    return float(norm.ppf(1.0 - float(eps)))

def _project_power_np(F: np.ndarray, P: float, eps: float = 1e-12) -> np.ndarray:
    p = np.linalg.norm(F, "fro") ** 2
    # if p <= P:
    #     return F
    return F * np.sqrt(P / (p + eps))


def _build_precoder_snapshot_from_models(
    uplinksystem,
    user_models: list[nn.Module],
) -> list[list[np.ndarray]]:
    snapshot: list[list[np.ndarray]] = []
    for k in range(int(uplinksystem.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(len(uplinksystem.H[k])):
            user_blocks.append(
                infer_precoder_numpy(
                    user_models[k],
                    np.asarray(uplinksystem.H[k][l], dtype=np.complex64),
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=float(uplinksystem.P[k]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)
    return snapshot

def _compute_R_fbl_np(
    H: np.ndarray,
    F: np.ndarray,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: np.ndarray | None = None,
) -> float:
    Nr = H.shape[0]
    HF = H @ F

    if noise_plus_interference_cov is None:
        noise_cov = float(sigma2) * np.eye(Nr, dtype=np.complex128)
    else:
        noise_cov = np.asarray(noise_plus_interference_cov, dtype=np.complex128)

    chol = np.linalg.cholesky(noise_cov)
    G = np.linalg.solve(chol, HF)
    A = G @ G.conj().T
    A = 0.5 * (A + A.conj().T)

    I = np.eye(Nr, dtype=np.complex64)
    sign, logdet = np.linalg.slogdet(I + A)
    C = (logdet / np.log(2.0)).real

    eigvals = np.linalg.eigvalsh(A).real
    V = np.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2) * (np.log2(np.e) ** 2)

    R = C - np.sqrt(V / float(n_kl)) * _Q_inv_np(epsilon)
    return float(np.real(R))

def dynamic_subblocklength_precoder_testing(
    uplinksystem,
    post_training_data_dict: dict,
    sim_cfg: dict,
    channel_norm: bool = True,
):
    """
    Testing version consistent with training_v2 block allocation, but WITHOUT precoder optimization:
      - Uses fixed trained precoders F_star (per user per block; fallback to last if needed)
      - Step A: at n=T, reduce B until feasible (for fixed F)
      - Step B: reduce n_kl while keeping B_used fixed (for fixed F)
      - Create additional blocks while B_rem > 0
      - Saves per-block trajectory S_test for plotting (same structure as training)

    Returns:
      test_data_dict with:
        - L_out_test, n_star_test, F_star_test, R_star_test
        - all_user_block_results_test: list[user][block] = S_test (trajectory over n_kl)
        - B_used_star_test, B_kl_star_test
        - norm_stats_used (the normalization stats applied)
    """
    K = int(uplinksystem.K)

    # from training
    F_star_train = post_training_data_dict["F_star"]
    norm_stats_train = post_training_data_dict["norm_stats"]
    user_model_specs = post_training_data_dict.get("user_model_specs")
    user_model_states = post_training_data_dict.get("user_model_states")
    user_models = None
    if user_model_specs is not None and user_model_states is not None:
        user_models = load_user_precoder_models(user_model_specs, user_model_states, device=DEVICE)

    # outputs
    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    # ---- apply same per-user normalization as training ----
    if channel_norm:
        for k in range(K):
            mean, var = norm_stats_train[k]
            H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var + 1e-12))

    for k in range(K):
        print(f"\n================ TEST USER {k} ================")

        P = float(uplinksystem.P[k])
        sigma2 = float(uplinksystem.sigma2[k])
        epsilon = float(uplinksystem.epsilon[k])
        T = int(uplinksystem.T[k])

        n_kl_min = int(sim_cfg["n_kl_min"])
        n_kl_step = int(sim_cfg["n_kl_step"])

        B_rem = int(uplinksystem.B[k])
        ell = 0

        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        while B_rem > 0:
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            H_kl = np.array(uplinksystem.H[k][ell], dtype=np.complex64)
            if user_models is not None:
                shared_snapshot = _build_precoder_snapshot_from_models(uplinksystem, user_models)
                F_fix = np.asarray(shared_snapshot[k][ell], dtype=np.complex64)
                F_fix_t = torch.tensor(F_fix, dtype=torch.complex64)
                noise_plus_interference_cov = uplinksystem.get_interference_plus_noise_covariance(
                    k, ell, F_override=shared_snapshot
                )
            else:
                # ---- fallback for older saved training artifacts ----
                if k < len(F_star_train) and ell < len(F_star_train[k]):
                    F_fix_t = F_star_train[k][ell]
                elif k < len(F_star_train) and len(F_star_train[k]) > 0:
                    print("Number of sub-blocks has exceeded expected value, no fixed precoder from training found, resorting to precoder from last block")
                    F_fix_t = F_star_train[k][-1]
                elif len(F_star_train[k]) == 0:
                    print(f">>> STOP test user {k}: no trained precoder available.")
                    break

                F_fix = F_fix_t.detach().cpu().numpy().astype(np.complex64)
                F_fix = _project_power_np(F_fix, P)
                noise_plus_interference_cov = uplinksystem.get_interference_plus_noise_covariance(
                    k, ell, F_override=F_star_train
                )

            print(f"\n--- TEST User {k}, Block {ell}, B_rem={B_rem} ---")

            # ==========================================================
            # STEP A: ensure feasibility at n=T by reducing B (fixed F)
            # ==========================================================
            B_try = int(B_rem)
            max_payload_reductions = 12
            B_used = None

            for attempt in range(max_payload_reductions):
                R_T = _compute_R_fbl_np(
                    H_kl, F_fix, sigma2, epsilon, n_kl=T,
                    noise_plus_interference_cov=noise_plus_interference_cov
                )
                rate_violation = (B_try / float(T)) - R_T

                print(f"Attempt {attempt+1}: n=T={T}, B={B_try}, R_fbl={R_T}, "
                      f"rate_violation={max(0.0, rate_violation)}")

                if rate_violation <= 0.0:
                    B_used = int(B_try)
                    break

                B_new = int(np.floor(T * R_T))
                B_new = max(0, min(B_new, B_try))
                if B_new == B_try:
                    B_used = 0
                    break
                print(f">>> Reducing B from {B_try} to {B_new}")
                print(f" Reduction B_new {B_try} = max({0}, min(B_new {B_new} = {T} * {R_T}, B_try {B_try})")
                B_try = B_new

            if B_used is None or B_used <= 0:
                print(f">>> STOP test user {k} at block {ell}: cannot make n=T feasible.")
                break

            # Build S trajectory like training: include n=T point + feasible decreasing n_kl points
            S_block = []

            # n=T point
            R_T = _compute_R_fbl_np(
                H_kl, F_fix, sigma2, epsilon, n_kl=T,
                noise_plus_interference_cov=noise_plus_interference_cov
            )
            S_block.append({
                "n_kl": int(T),
                "n": int(T),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(T),
                "F": F_fix_t,  # keep torch tensor for consistency with training plots/SNR plots
                "R_fbl": float(R_T),
                "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                "lambda_rate": 0.0,
                "lambda_power": 0.0,
                "loss_curve": [],
            })

            # ==========================================================
            # STEP B: decrease n_kl with B_used fixed (fixed F)
            # ==========================================================
            best_n = int(T)
            best_R = float(R_T)

            n_kl = T - n_kl_step
            while n_kl >= n_kl_min:
                R = _compute_R_fbl_np(
                    H_kl, F_fix, sigma2, epsilon, n_kl=n_kl,
                    noise_plus_interference_cov=noise_plus_interference_cov
                )
                rate_violation = (B_used / float(n_kl)) - R

                print(f"Test n_kl={n_kl}: R_fbl={R}, rate_violation={rate_violation}")

                if rate_violation <= 0.0:
                    best_n = int(n_kl)
                    best_R = float(R)

                    S_block.append({
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl),
                        "F": F_fix_t,
                        "R_fbl": float(R),
                        "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                    })

                    n_kl -= n_kl_step
                else:
                    break

            print(f">>> Chosen block {ell}: n_kl={best_n}, B_used={B_used}, R_fbl={best_R}")

            # save trajectory for plotting
            all_user_block_results[k].append(S_block)

            # save chosen decisions
            n_star[k].append(best_n)
            F_star[k].append(F_fix_t)
            R_star[k].append(best_R)
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl
            print(f">>> Transmitted B_kl={B_kl}, remaining B_rem={B_rem}")

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1
            else:
                ell += 1
                L_out[k] = ell
                
        
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])

    # push F into uplinksystem (numpy)
    for k in range(K):
        if len(F_star[k]) > 0:
            uplinksystem.F[k] = np.array([F.detach().cpu().numpy() for F in F_star[k]])

    try:
        uplinksystem.update_system()
    except Exception:
        pass

    test_data_dict = {
        "L_out_test": L_out,
        "n_star_test": n_star,
        "F_star_test": F_star,
        "R_star_test": R_star,
        "all_user_block_results_test": all_user_block_results,
        "B_used_star_test": B_used_star,
        "B_kl_star_test": B_kl_star,
        "norm_stats_used": norm_stats_train,
        "user_model_specs": user_model_specs,
        "user_model_states": user_model_states,
        "precoder_parameterization": post_training_data_dict.get(
            "precoder_parameterization",
            "per_block_precoders" if user_models is None else "shared_user_channel_to_precoder_mlp",
        ),
    }
    return test_data_dict

# ============================================================
# Usage
# ============================================================
if __name__ == "__main__":
    from config_loader import get_config
    from UplinkSystem import UplinkSystem

    SYSTEM_TEST_PARAMS, SIMULATION_TEST_PARAMS = get_config()
    print(SYSTEM_TEST_PARAMS)
    print(SIMULATION_TEST_PARAMS)
    uplinksystem = UplinkSystem(SYSTEM_TEST_PARAMS, seed=0)

    # expected keys in SIMULATION_TEST_PARAMS:
    #   initial_lambda_rate_constraint, initial_lambda_power_constraint,
    #   epochs_per_n_kl, lr_net, lr_rate_constraint, lr_power_constraint,
    #   n_kl_min, n_kl_step
    sim_cfg = dict(SIMULATION_TEST_PARAMS)

    L_out, n_star, F_star, R_star, norm_stats = dynamic_subblocklength_precoder_training(
        uplinksystem=uplinksystem,
        sim_cfg=sim_cfg,
        channel_norm=True
    )

    print("\n================ FINAL ================")
    print("L:", L_out)
    print("n*:", n_star)
    print("R*:", R_star)
