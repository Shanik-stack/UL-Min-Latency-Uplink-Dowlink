import numpy as np
import matplotlib.pyplot as plt
# from system_constants import *
from config_loader import *
from scipy.stats import norm


def Q_inv(x):
    return norm.ppf(1-x)
import numpy as np
from typing import Any, Dict, List, Optional, Tuple



class UplinkSystem:
    """
    Single-class uplink simulator with fixed noise variance sigma2 (per user).
    sigma2 is calibrated ONCE from target snr_db using initially generated (H,F,X).
    After calibration, sigma2 is kept fixed even as new blocks are added.
    Finite-blocklength rates are evaluated with interference-aware whitening, so
    `snr_db` remains the simulation hyperparameter while the realized link metric
    is SINR once other users are present.

    Shapes:
      H[k][l] : (NR[k], NT[k])
      F[k][l] : (NT[k], dk[k]) with ||F||_F^2 = P[k]
      X[k][l] : (dk[k], n_kl[k][l])
      N[k][l] : (NR[k], n_kl[k][l])
      Y[k][l] : (NR[k], n_kl[k][l])
    """

    STREAM_H = 0
    STREAM_X = 1
    STREAM_F = 2
    STREAM_N = 3

    def __init__(self, system_constants: Dict[str, Any], seed: int):
        self.sc = system_constants
        self.seed = int(seed)

        # ---- required constants ----
        self.K = int(self.sc["K"])
        self.P = list(self.sc["P"])
        self.NR = list(self.sc["NR"])
        self.NT = list(self.sc["NT"])
        self.dk = list(self.sc["dk"])
        self.B = list(self.sc["B"])   # bits per user (mutable in optimizer)
        self.initial_bits_per_symbol = list(self.sc["initial_bits_per_symbol"])

        self.L = list(self.sc["L"])  # list[int], length K
        self.n_kl = [list(v) for v in self.sc["n_kl"]]  # list[list[int]]
        self.snr_db = list(self.sc["snr_db"])  # target SNR in dB per user
        
        self.initial_latency = list(self.sc["initial_latency"])

        self.epsilon = list(self.sc["epsilon"])
        self.fs = list(self.sc["fs"])
        self.T = list(self.sc["T"])  # default block length for add_block()

        # ---- deterministic base seed for per-(k,l,stream) RNG ----
        base_ss = np.random.SeedSequence([self.seed])
        self.base_seed = int(base_ss.generate_state(1, dtype=np.uint32)[0])

        # ---- storage ----
        self.H: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.F: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.X: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.N: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.Y: List[List[np.ndarray]] = [[] for _ in range(self.K)]

        # noise variance (fixed after calibration)
        self.sigma2: List[float] = [np.nan] * self.K

        # metrics
        self.SNR_linear: List[float] = [np.nan] * self.K
        self.SNR_db_measured: List[float] = [np.nan] * self.K
        self.SINR_linear: List[float] = [np.nan] * self.K
        self.SINR_db_measured: List[float] = [np.nan] * self.K
        self.CNR_linear: List[float] = [np.nan] * self.K
        self.CNR_db: List[float] = [np.nan] * self.K

        # derived
        self.n: List[int] = [0] * self.K
        self.latency: List[float] = [0.0] * self.K

        # rate outputs
        self.C: List[np.ndarray] = []
        self.V: List[np.ndarray] = []
        self.R_fbl: List[np.ndarray] = []
        self.usr_avg_C: List[float] = []

        # ---- generate initial blocks for H,F,X ----
        for k in range(self.K):
            for l in range(self.L[k]):
                self._ensure_block(k, l)

        # ---- calibrate sigma2 ONCE from target snr_db using initial blocks ----
        self._calibrate_sigma2_from_target_snr(reference="rx_signal_power")

        # ---- generate noise and received signals using fixed sigma2 ----
        self._generate_noise_all()
        self._generate_received_all()

        # ---- compute capacity/dispersion/fbl ----
        self.update_system()
        self.SNR_linear, self.SNR_db_measured = self.get_SNR()
        self.SINR_linear, self.SINR_db_measured = self.get_SINR()
        print("Initial R_fbl after update system: ", self.R_fbl)


    # ============================================================
    # RNG + complex normal
    # ============================================================
    def _rng_for(self, user: int, block: int, stream: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.base_seed, int(user), int(block), int(stream)])
        return np.random.default_rng(ss)

    @staticmethod
    def _cn(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
        """Circularly symmetric complex Gaussian CN(0,1): E|z|^2 = 1."""
        real = rng.normal(0.0, 1.0 / np.sqrt(2.0), shape)
        imag = rng.normal(0.0, 1.0 / np.sqrt(2.0), shape)
        return real + 1j * imag

    @staticmethod
    def _as_complex_array(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype(np.complex128, copy=False)
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().astype(np.complex128, copy=False)
        return np.asarray(x, dtype=np.complex128)

    @staticmethod
    def _safe_db(value: float) -> float:
        return 10.0 * np.log10(max(float(value), 1e-30))

    def _resolve_metric_block_index(self, user: int, block: int, F_override=None, require_x: bool = False) -> int:
        lengths = [len(self.H[user])]

        if F_override is not None and user < len(F_override) and len(F_override[user]) > 0:
            lengths.append(len(F_override[user]))
        else:
            lengths.append(len(self.F[user]))

        if require_x:
            lengths.append(len(self.X[user]))

        valid_lengths = [n for n in lengths if n > 0]
        if not valid_lengths:
            raise ValueError(f"No blocks available for user {user} when resolving metric block index.")
        return min(int(block), min(valid_lengths) - 1)

    def _build_effective_metric_matrix(self, user: int, block: int, F_override=None) -> np.ndarray:
        k = int(user)
        l = self._resolve_metric_block_index(k, block, F_override=F_override)
        H_k = self._as_complex_array(self.H[k][l])
        F_k = self._as_complex_array(
            (F_override[k][l] if F_override is not None and k < len(F_override) and len(F_override[k]) > 0
             else self.F[k][l])
        )

        HF = H_k @ F_k
        noise_plus_interference = self.get_interference_plus_noise_covariance(k, l, F_override=F_override)
        chol = np.linalg.cholesky(noise_plus_interference)
        G = np.linalg.solve(chol, HF)
        A = G @ G.conj().T
        return 0.5 * (A + A.conj().T)

    def get_interference_plus_noise_covariance(self, user: int, block: int, F_override=None) -> np.ndarray:
        k = int(user)
        Nr = int(self.NR[k])
        R = float(self.sigma2[k]) * np.eye(Nr, dtype=np.complex128)

        for j in range(self.K):
            if j == k:
                continue

            lj = self._resolve_metric_block_index(j, block, F_override=F_override)
            H_j = self._as_complex_array(self.H[j][lj])
            if H_j.shape[0] != Nr:
                raise ValueError(
                    "SINR interference model requires a common BS receive dimension NR across users. "
                    f"User {k} has NR={Nr}, user {j} block {lj} has NR={H_j.shape[0]}."
                )

            if F_override is not None and j < len(F_override) and len(F_override[j]) > 0:
                F_j = self._as_complex_array(F_override[j][lj])
            else:
                F_j = self._as_complex_array(self.F[j][lj])

            HF_j = H_j @ F_j
            R += HF_j @ HF_j.conj().T

        return 0.5 * (R + R.conj().T)

    # ============================================================
    # Block creation
    # ============================================================
    def _ensure_block(self, k: int, l: int) -> None:
        """Ensure H/F/X exist for (k,l) and satisfy power constraint for F.
        Option B: if n_kl changes for an existing block, regenerate X (not N).
        """
        # H
        if l >= len(self.H[k]):
            rng_h = self._rng_for(k, l, self.STREAM_H)
            H_kl = self._cn(rng_h, (self.NR[k], self.NT[k]))
            self.H[k].append(H_kl)

        # X (create OR regenerate if n_kl changed)
        n_kl = int(self.n_kl[k][l])

        if l >= len(self.X[k]):
            rng_x = self._rng_for(k, l, self.STREAM_X)
            X_kl = self._cn(rng_x, (self.dk[k], n_kl))
            self.X[k].append(X_kl)
        else:
            # Existing block: if blocklength changed, rebuild X deterministically
            if self.X[k][l].shape[1] != n_kl:
                rng_x = self._rng_for(k, l, self.STREAM_X)
                self.X[k][l] = self._cn(rng_x, (self.dk[k], n_kl))

                # Invalidate cached derived quantities that depend on X/n_kl (NOT noise N)
                for attr in ("Y", "C", "V", "R_fbl"):
                    if hasattr(self, attr):
                        arr = getattr(self, attr)
                        if k < len(arr) and l < len(arr[k]):
                            arr[k][l] = None

        # F (with ||F||_F^2 = P[k])
        if l >= len(self.F[k]):
            rng_f = self._rng_for(k, l, self.STREAM_F)
            F_kl = self._cn(rng_f, (self.NT[k], self.dk[k]))
            fro = np.linalg.norm(F_kl, ord="fro")
            if fro == 0:
                # pathological; extremely unlikely
                F_kl = np.eye(self.NT[k], self.dk[k], dtype=np.complex128)
                fro = np.linalg.norm(F_kl, ord="fro")
            F_kl = F_kl * np.sqrt(self.P[k]) / fro

            # debug safety (can remove later)
            pow_val = np.linalg.norm(F_kl, ord="fro") ** 2
            if (not np.isfinite(pow_val)) or (abs(pow_val - self.P[k]) > 1e-6 * max(1.0, self.P[k])):
                raise ValueError(
                    f"Precoder power constraint violated: user {k} block {l} got {pow_val} expected {self.P[k]}"
                )

            self.F[k].append(F_kl)

        # shape checks
        assert self.H[k][l].shape == (self.NR[k], self.NT[k])
        assert self.F[k][l].shape == (self.NT[k], self.dk[k])
        assert self.X[k][l].shape[0] == self.dk[k]
        assert self.X[k][l].shape[1] == int(self.n_kl[k][l])

    # ============================================================
    # Calibration: compute sigma2 once (because sigma2 not provided)
    # ============================================================
    def _calibrate_sigma2_from_target_snr(self, reference: str = "rx_signal_power") -> None:
        """
        Compute sigma2[k] once from target snr_db[k].

        reference:
          - "rx_signal_power" (recommended): estimate P_signal = mean |H F X|^2 over initial blocks.
            Then sigma2 = P_signal / SNR_target.
        """
        if reference != "rx_signal_power":
            raise ValueError("Only reference='rx_signal_power' is implemented.")

        for k in range(self.K):
            # estimate received signal power (noiseless) over existing blocks
            p_blocks = []
            for l in range(self.L[k]):
                Ysig = self.H[k][l] @ (self.F[k][l] @ self.X[k][l])
                p_blocks.append(np.mean(np.abs(Ysig) ** 2))
            P_signal = float(np.mean(p_blocks))

            snr_lin_target = 10.0 ** (float(self.snr_db[k]) / 10.0)
            sigma2_k = P_signal / snr_lin_target

            if not np.isfinite(sigma2_k) or sigma2_k <= 0:
                raise ValueError(f"Bad sigma2 calibration for user {k}: P_signal={P_signal}, target_snr_lin={snr_lin_target}")

            self.sigma2[k] = float(sigma2_k)

        # store the *target* SNR as a reference (measured SNR can drift with new blocks)
        self.SNR_linear = [10.0 ** (float(db) / 10.0) for db in self.snr_db]
        self.SNR_db_measured = list(self.snr_db)
        self.SINR_linear = [10.0 ** (float(db) / 10.0) for db in self.snr_db]
        self.SINR_db_measured = list(self.snr_db)

        # compute initial CNR metrics under this sigma2 (optional)
        self._update_cnr_metrics()

    def _update_cnr_metrics(self) -> None:
        for k in range(self.K):
            H_power = float(np.mean([np.linalg.norm(self.H[k][l], "fro") ** 2 for l in range(self.L[k])]))
            cnr_lin = H_power / float(self.sigma2[k])
            self.CNR_linear[k] = cnr_lin
            self.CNR_db[k] = 10.0 * np.log10(cnr_lin)

    # ============================================================
    # Noise + receive generation (sigma2 fixed)
    # ============================================================
    def _generate_noise_all(self) -> None:
        self.N = [[] for _ in range(self.K)]
        for k in range(self.K):
            sigma = np.sqrt(float(self.sigma2[k]))
            for l in range(self.L[k]):
                n_kl = int(self.n_kl[k][l])
                rng_n = self._rng_for(k, l, self.STREAM_N)
                N_kl = sigma * self._cn(rng_n, (self.NR[k], n_kl))
                self.N[k].append(N_kl)

    def _generate_received_all(self) -> None:
        self.Y = [[] for _ in range(self.K)]
        for k in range(self.K):
            for l in range(self.L[k]):
                Y_kl = self.H[k][l] @ (self.F[k][l] @ self.X[k][l]) + self.N[k][l]
                self.Y[k].append(Y_kl)

        self._recompute_latency()

    def _recompute_latency(self) -> None:
        self.n = [int(sum(self.n_kl[k])) for k in range(self.K)]
        self.latency = [self.n[k] / float(self.fs[k]) for k in range(self.K)]

    # ============================================================
    # Public methods
    # ============================================================
    def add_block(self, user: int, n_block: Optional[int] = None) -> None:
        """
        Add one new block for `user`. sigma2 remains fixed.
        This will append H,F,X (deterministic), then generate N,Y for the new block.
        """
        k = int(user)
        l_new = len(self.H[k])

        # update bookkeeping
        self.L[k] += 1
        if n_block is None:
            n_block = int(self.T[k])
        self.n_kl[k].append(int(n_block))

        # create new H,F,X
        self._ensure_block(k, l_new)

        # create noise for the new block with fixed sigma2
        sigma = np.sqrt(float(self.sigma2[k]))
        rng_n = self._rng_for(k, l_new, self.STREAM_N)
        N_new = sigma * self._cn(rng_n, (self.NR[k], int(self.n_kl[k][l_new])))
        self.N[k].append(N_new)

        # received
        Y_new = self.H[k][l_new] @ (self.F[k][l_new] @ self.X[k][l_new]) + N_new
        self.Y[k].append(Y_new)

        # update metrics
        self._recompute_latency()
        self.update_system()

    def regenerate_received(self, regenerate_noise: bool = False) -> None:
        """
        Recompute Y based on current H,F,X and either:
          - existing N (regenerate_noise=False)
          - regenerated N using deterministic seeds (regenerate_noise=True)
        sigma2 remains fixed either way.
        """
        if regenerate_noise:
            self._generate_noise_all()
        self._generate_received_all()
        
    def update_system(self, F=None, n_kl=None, regenerate_noise_on_nl_change: bool = True):
        """
        Update everything that depends on F or n_kl.

        Rules:
        - If F changes: Y changes, capacities change. Noise N can stay unchanged.
        - If n_kl changes: X/N/Y shapes change. We regenerate X and (optionally) N deterministically.
        - sigma2 stays fixed (per your requirement).
        """
        # ---------- track what changed ----------
        old_n_kl = [list(v) for v in self.n_kl]  # deep-ish copy of lists of ints
        nl_changed = False

        # ---------- apply updates ----------
        if F is not None:
            self.F = F

        if n_kl is not None:
            self.n_kl = [list(v) for v in n_kl]
            # check if any length differs
            if len(old_n_kl) != len(self.n_kl):
                nl_changed = True
            else:
                for k in range(self.K):
                    if len(old_n_kl[k]) != len(self.n_kl[k]):
                        nl_changed = True
                        break
                    for l in range(len(self.n_kl[k])):
                        if int(old_n_kl[k][l]) != int(self.n_kl[k][l]):
                            nl_changed = True
                            break
                    if nl_changed:
                        break

        # ---------- ensure consistency with L ----------
        # Keep L derived from n_kl (recommended). This avoids silent mismatches.
        self.L = [len(self.n_kl[k]) for k in range(self.K)]

        # Ensure H/F/X lists have exactly L[k] blocks; create missing blocks deterministically.
        for k in range(self.K):
            for l in range(self.L[k]):
                self._ensure_block(k, l)  # creates H/F/X if missing, enforces F power

            # If someone reduced L via n_kl, truncate stored arrays
            if len(self.H[k]) > self.L[k]:
                self.H[k] = self.H[k][:self.L[k]]
            if len(self.F[k]) > self.L[k]:
                self.F[k] = self.F[k][:self.L[k]]
            if len(self.X[k]) > self.L[k]:
                self.X[k] = self.X[k][:self.L[k]]
            if len(self.N[k]) > self.L[k]:
                self.N[k] = self.N[k][:self.L[k]]
            if len(self.Y[k]) > self.L[k]:
                self.Y[k] = self.Y[k][:self.L[k]]

        # ---------- if n_kl changed: regenerate X (+/- N) for affected blocks ----------
        if nl_changed:
            for k in range(self.K):
                for l in range(self.L[k]):
                    n_kl = int(self.n_kl[k][l])

                    # Regenerate X with deterministic seed so it is consistent with (k,l) and new n_kl
                    rng_x = self._rng_for(k, l, self.STREAM_X)
                    self.X[k][l] = self._cn(rng_x, (self.dk[k], n_kl))

                    if regenerate_noise_on_nl_change:
                        # Regenerate N deterministically with fixed sigma2 and new n_kl
                        sigma = np.sqrt(float(self.sigma2[k]))
                        rng_n = self._rng_for(k, l, self.STREAM_N)
                        N_new = sigma * self._cn(rng_n, (self.NR[k], n_kl))
                        if l >= len(self.N[k]):
                            self.N[k].append(N_new)
                        else:
                            self.N[k][l] = N_new
                    else:
                        # Keep old N content if possible (truncate/extend deterministically)
                        # If extending, append fresh samples deterministically.
                        if l >= len(self.N[k]):
                            sigma = np.sqrt(float(self.sigma2[k]))
                            rng_n = self._rng_for(k, l, self.STREAM_N)
                            self.N[k].append(sigma * self._cn(rng_n, (self.NR[k], n_kl)))
                        else:
                            N_old = self.N[k][l]
                            n_old = N_old.shape[1]
                            if n_kl == n_old:
                                pass
                            elif n_kl < n_old:
                                self.N[k][l] = N_old[:, :n_kl]
                            else:
                                # extend deterministically
                                sigma = np.sqrt(float(self.sigma2[k]))
                                rng_n = self._rng_for(k, l, self.STREAM_N)
                                N_full = sigma * self._cn(rng_n, (self.NR[k], n_kl))
                                # overwrite with deterministic full so behavior is consistent
                                self.N[k][l] = N_full

        # ---------- recompute Y for all blocks (always needed if F changed; needed if X/N changed too) ----------
        # Ensure N exists for every block even if nl_changed=False and N was never generated (defensive)
        for k in range(self.K):
            sigma = np.sqrt(float(self.sigma2[k]))
            for l in range(self.L[k]):
                n_kl = int(self.n_kl[k][l])

                if l >= len(self.N[k]) or self.N[k][l].shape != (self.NR[k], n_kl):
                    rng_n = self._rng_for(k, l, self.STREAM_N)
                    if l >= len(self.N[k]):
                        self.N[k].append(sigma * self._cn(rng_n, (self.NR[k], n_kl)))
                    else:
                        self.N[k][l] = sigma * self._cn(rng_n, (self.NR[k], n_kl))

                Y_new = self.H[k][l] @ (self.F[k][l] @ self.X[k][l]) + self.N[k][l]
                if l >= len(self.Y[k]):
                    self.Y[k].append(Y_new)
                else:
                    self.Y[k][l] = Y_new

        # ---------- recompute derived n, latency ----------
        self.n = [int(sum(self.n_kl[k])) for k in range(self.K)]
        self.latency = [self.n[k] / float(self.fs[k]) for k in range(self.K)]

        # ---------- recompute C, V, R_fbl ----------
        self.C, self.V, self.R_fbl, self.usr_avg_C = [], [], [], []
        LOG2E_SQ = (np.log2(np.e)) ** 2

        for k in range(self.K):
            Lk = self.L[k]
            Nr = self.NR[k]
            eps = float(self.epsilon[k])

            Tk = np.asarray(self.n_kl[k], dtype=float)
            Ck = np.zeros(Lk, dtype=float)
            Vk = np.zeros(Lk, dtype=float)

            for l in range(Lk):
                A = self._build_effective_metric_matrix(k, l)

                I = np.eye(Nr, dtype=np.complex128)
                sign, logdet = np.linalg.slogdet(I + A)
                if sign <= 0:
                    raise ValueError(f"slogdet sign<=0 at user {k}, block {l}; numerical issue in I+A")
                Ck[l] = float(logdet / np.log(2.0))

                eigvals = np.linalg.eigvalsh(A)
                Vk[l] = float(np.sum(eigvals * (eigvals + 2.0) / (eigvals + 1.0) ** 2) * LOG2E_SQ)

            R_fblk = Ck - np.sqrt(Vk / Tk) * Q_inv(eps)

            self.C.append(Ck)
            self.V.append(Vk)
            self.R_fbl.append(R_fblk)
            self.usr_avg_C.append(float(np.mean(Ck)))

        # optional: refresh CNR metrics (H and sigma2 define it; H might have been truncated/extended)
        self._update_cnr_metrics()

    def get_SNR(self) -> Tuple[List[float], List[float]]:
        """
        Measured SNR per user using current H,F,X and actual generated noise N.
        With fixed sigma2, measured SNR will vary as you add blocks (because signal power varies).
        """
        snr_lin_klist: List[float] = []
        snr_db_list: List[float] = []

        for k in range(self.K):
            p_sig = float(np.mean([
                np.mean(np.abs(self.H[k][l] @ (self.F[k][l] @ self.X[k][l])) ** 2)
                for l in range(self.L[k])
            ]))
            p_noise = float(np.mean([np.mean(np.abs(self.N[k][l]) ** 2) for l in range(self.L[k])]))

            snr_lin = p_sig / max(p_noise, 1e-30)
            snr_db = self._safe_db(snr_lin)

            snr_lin_klist.append(snr_lin)
            snr_db_list.append(snr_db)

        return snr_lin_klist, snr_db_list

    def get_SINR(self) -> Tuple[List[float], List[float]]:
        """
        Measured SINR per user using current H,F,X and generated noise N.
        Interference is modeled as the aggregate received power from all other users
        at the BS and is mapped blockwise using the closest available block index.
        """
        sinr_lin_klist: List[float] = []
        sinr_db_list: List[float] = []

        for k in range(self.K):
            sig_blocks: List[float] = []
            interf_blocks: List[float] = []
            noise_blocks: List[float] = []

            for l in range(self.L[k]):
                H_k = self._as_complex_array(self.H[k][l])
                F_k = self._as_complex_array(self.F[k][l])
                X_k = self._as_complex_array(self.X[k][l])
                desired = H_k @ (F_k @ X_k)

                p_sig = float(np.mean(np.abs(desired) ** 2))
                p_noise = float(np.mean(np.abs(self.N[k][l]) ** 2))
                p_interf = 0.0

                for j in range(self.K):
                    if j == k:
                        continue

                    lj = self._resolve_metric_block_index(j, l, require_x=True)
                    H_j = self._as_complex_array(self.H[j][lj])
                    if H_j.shape[0] != H_k.shape[0]:
                        raise ValueError(
                            "SINR interference model requires a common BS receive dimension NR across users. "
                            f"User {k} has NR={H_k.shape[0]}, user {j} block {lj} has NR={H_j.shape[0]}."
                        )

                    F_j = self._as_complex_array(self.F[j][lj])
                    X_j = self._as_complex_array(self.X[j][lj])
                    interference = H_j @ (F_j @ X_j)
                    p_interf += float(np.mean(np.abs(interference) ** 2))

                sig_blocks.append(p_sig)
                interf_blocks.append(p_interf)
                noise_blocks.append(p_noise)

            mean_sig = float(np.mean(sig_blocks)) if sig_blocks else 0.0
            mean_interf = float(np.mean(interf_blocks)) if interf_blocks else 0.0
            mean_noise = float(np.mean(noise_blocks)) if noise_blocks else 0.0

            sinr_lin = mean_sig / max(mean_interf + mean_noise, 1e-30)
            sinr_lin_klist.append(sinr_lin)
            sinr_db_list.append(self._safe_db(sinr_lin))

        return sinr_lin_klist, sinr_db_list

    def get_CNR(self) -> Tuple[List[float], List[float]]:
        """
        CNR per user using average channel power and actual generated noise power from N.
        """
        cnr_lin_klist: List[float] = []
        cnr_db_list: List[float] = []

        for k in range(self.K):
            H_power = float(np.mean([np.linalg.norm(self.H[k][l], "fro") ** 2 for l in range(self.L[k])]))
            p_noise = float(np.mean([np.mean(np.abs(self.N[k][l]) ** 2) for l in range(self.L[k])]))

            cnr_lin = H_power / p_noise
            cnr_db = 10.0 * np.log10(cnr_lin)

            cnr_lin_klist.append(cnr_lin)
            cnr_db_list.append(cnr_db)

        return cnr_lin_klist, cnr_db_list

    # ============================================================
    # Optional: explicit recalibration if YOU ever want it
    # ============================================================
    def recalibrate_sigma2(self) -> None:
        """
        Explicitly recalibrate sigma2 from target snr_db using current existing blocks.
        Not used by default (because you requested fixed sigma2), but provided for experiments.
        """
        self._calibrate_sigma2_from_target_snr(reference="rx_signal_power")
        self._generate_noise_all()
        self._generate_received_all()
        self.update_system()
        self.SNR_linear, self.SNR_db_measured = self.get_SNR()
        self.SINR_linear, self.SINR_db_measured = self.get_SINR()
        
if __name__ == "__main__":   
    SYSTEM_TEST_PARAMS, SIMULATION_TEST_PARAMS = get_config()
    System = UplinkSystem(SYSTEM_TEST_PARAMS, seed = 0)

    #-----------------Sanity Check System--------------------$
    test_usr, test_block = 0,1
    # print("H")
    # print(System.H[test_usr][test_block])
    # print("X")
    # print(System.X[test_usr][test_block])
    # print("Y")
    # print(System.Y[test_usr][test_block])

    # print("Check Dims")
    # print(f" Y = {System.Y[test_usr].shape}, H = {System.H[test_usr].shape} , F = {System.F[test_usr].shape}, X = {System.X[test_usr].shape}\n")
    #--------------------------------------------------$
    
    print("|------------ System Constants ------------|")
    print()
    
    print("R_fbl: ",System.R_fbl[test_usr][test_block].real)
    print("C: ",System.C[test_usr][test_block].real)
    print("F: ",System.F[test_usr][test_block])

    # System.check_SNR_user()
    # # System.check_SNR_block()
    # System.check_latency_users()
    
    # System.ChannelSystem.plot_magnitude_per_block()
    # System.ChannelSystem.plot_magnitude_over_blocklength()
    
    # for usr in range(test_k):
    #     print(f"Capacity per user per coh block: User {usr} :", System.C[usr])
    #     print(f"Ergodic capacity (avg across coh block) per user: User {usr}: ", System.ergodic_C[usr])
     
    # "Print Rate"
    # for usr in range(System.K):
    #     print(f"Rate_fbl per user : User {usr} :", System.R_fbl[usr])
    # print(System.sigma2[0])

