from __future__ import annotations

from typing import Any, Mapping

import numpy as np


UPLINK_RATE_MODEL_SINR = "sinr"
UPLINK_RATE_MODEL_SNR = "snr"


def normalize_uplink_rate_model(value: str | None) -> str:
    model = str(value or UPLINK_RATE_MODEL_SINR).strip().lower()
    aliases = {
        "interference": UPLINK_RATE_MODEL_SINR,
        "interference_plus_noise": UPLINK_RATE_MODEL_SINR,
        "noise_only": UPLINK_RATE_MODEL_SNR,
    }
    model = aliases.get(model, model)
    if model not in {UPLINK_RATE_MODEL_SINR, UPLINK_RATE_MODEL_SNR}:
        raise ValueError(
            "Unsupported uplink_rate_model. Expected one of "
            f"{UPLINK_RATE_MODEL_SINR!r} or {UPLINK_RATE_MODEL_SNR!r}, got {value!r}."
        )
    return model


def get_uplink_rate_model(sim_cfg: Mapping[str, Any] | None) -> str:
    if sim_cfg is None:
        return UPLINK_RATE_MODEL_SINR
    return normalize_uplink_rate_model(sim_cfg.get("uplink_rate_model", UPLINK_RATE_MODEL_SINR))


def uses_uplink_interference(sim_cfg: Mapping[str, Any] | None) -> bool:
    return get_uplink_rate_model(sim_cfg) == UPLINK_RATE_MODEL_SINR


def build_uplink_rate_covariance(
    uplinksystem,
    sim_cfg: Mapping[str, Any] | None,
    user: int,
    block: int,
    *,
    F_override=None,
) -> np.ndarray | None:
    if not uses_uplink_interference(sim_cfg):
        return None
    return np.asarray(
        uplinksystem.get_interference_plus_noise_covariance(
            int(user),
            int(block),
            F_override=F_override,
        ),
        dtype=np.complex128,
    )
