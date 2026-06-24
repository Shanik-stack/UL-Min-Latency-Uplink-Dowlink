import numpy as np
import torch

from Optimizer_per_block import DEVICE, UplinkMLP_PrecoderNet, optimize_subblocklength_precoder


def dynamic_subblocklength_precoder_training_baseline(
    uplinksystem,
    sim_cfg: dict,
    channel_norm: bool = True,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """
    Original slow baseline:
      - each user/block gets its own fresh precoder network
      - that block network is fully optimized over the local n-search trajectory
      - no parameters are shared across different blocks
    """

    K = int(uplinksystem.K)

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]

    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    lambda_rate_0 = float(sim_cfg["initial_lambda_rate_constraint"])
    lambda_power_0 = float(sim_cfg["initial_lambda_power_constraint"])

    for k in range(K):
        print(f"\n================ BASELINE TRAIN USER {k} ================")

        H_user = np.array(uplinksystem.H[k], dtype=np.complex64)
        mean = np.mean(H_user)
        var = np.mean(np.abs(H_user - mean) ** 2) + 1e-12
        norm_stats.append((mean, var))

        if channel_norm:
            uplinksystem.H[k] = list((H_user - mean) / np.sqrt(var))

        B_rem = int(uplinksystem.B[k])
        ell = 0

        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        lambda_rate_user = float(lambda_rate_0)
        lambda_power_user = float(lambda_power_0)

        while B_rem > 0:
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            print(f"\n--- BASELINE User {k}, Block {ell}, B_rem={B_rem} ---")

            block_model = UplinkMLP_PrecoderNet(
                Nr=int(uplinksystem.NR[k]),
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
            ).to(DEVICE)
            block_optimizer = torch.optim.Adam(block_model.parameters(), lr=float(sim_cfg["lr_net"]))

            S, B_used = optimize_subblocklength_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=B_rem,
                lambda_rate_0=lambda_rate_user,
                lambda_power_0=lambda_power_user,
                sim_cfg=sim_cfg,
                precoder_net=block_model,
                optimizer=block_optimizer,
                interference_F_snapshot=interference_F_snapshot,
            )

            if len(S) == 0 or B_used <= 0:
                print(f">>> STOP baseline user {k} at block {ell} (no feasible solution).")
                break

            all_user_block_results[k].append(S)

            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])

            n_star[k].append(n_opt)
            F_star[k].append(F_opt)
            R_star[k].append(R_opt)

            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()

            B_used_star[k].append(int(B_used))
            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))

            B_rem -= B_kl
            print(f">>> Baseline chose n_kl={n_opt}, B_used={B_used}, B_kl={B_kl}, remaining B_rem={B_rem}")

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1

        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])

    return {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "precoder_parameterization": "independent_block_precoders_online_convergence",
        "method_name": "converge_in_each_sweep_baseline",
    }

