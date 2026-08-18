from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import torch


LORA_SIGNATURE_KEY = "minimax_h3_fastpath_lora_signature"
LORA_SIGMA_KEY = "minimax_h3_fastpath_sigma"

CACHE_BINDING_KEY = "minimax_h3_fastpath_cache_binding"
CACHE_CALL_KEY = "minimax_h3_fastpath_cache_call"
CACHE_ATTACHMENT_KEY = "minimax_h3_fastpath_cache_attachment"
CACHE_WRAPPER_KEY = "minimax_h3_fastpath_middle_cache"

LORA_ATTACHMENT_KEY = "minimax_h3_fastpath_lora_specs"
LORA_INJECTION_KEY = "minimax_h3_fastpath_scheduled_lora"


def sigma_anchors(sigmas: Any) -> tuple[float, ...]:
    values = torch.as_tensor(sigmas).detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() < 2:
        raise ValueError("SIGMAS must contain at least one model-call sigma and the final endpoint")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SIGMAS contains NaN or infinite values")
    if bool((values < 0.0).any()):
        raise ValueError("SIGMAS must be non-negative")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("SIGMAS must be monotonically descending")

    anchors = values[:-1]
    if anchors.numel() > 1 and bool((anchors[1:] >= anchors[:-1]).any()):
        raise ValueError(
            "Model-call SIGMAS must be strictly descending; repeated values cannot identify calls safely"
        )
    return tuple(float(value) for value in anchors.tolist())


def explicit_strengths(text: str, count: int) -> tuple[float, ...]:
    cleaned = re.sub(r"[\[\](){}]", " ", text.strip())
    tokens = [token for token in re.split(r"[,;\s]+", cleaned) if token]
    if len(tokens) != count:
        raise ValueError(
            f"explicit_strengths needs exactly {count} values for sigmas[:-1]; received {len(tokens)}"
        )
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise ValueError("explicit_strengths contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("explicit_strengths contains NaN or infinite values")
    return values


def curve_strengths(
    count: int,
    start_strength: float,
    end_strength: float,
    curve: str,
    curve_power: float,
    explicit: str,
) -> tuple[float, ...]:
    if curve == "explicit":
        return explicit_strengths(explicit, count)
    if count == 1:
        return (float(start_strength),)

    values: list[float] = []
    for index in range(count):
        progress = index / (count - 1)
        if curve == "linear":
            shaped = progress
        elif curve == "cosine":
            shaped = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif curve == "smoothstep":
            shaped = progress * progress * (3.0 - 2.0 * progress)
        elif curve == "power":
            shaped = progress**curve_power
        else:
            raise ValueError(f"Unknown strength curve: {curve}")
        values.append(start_strength + shaped * (end_strength - start_strength))
    return tuple(float(value) for value in values)


def strength_at_sigma(group: dict[str, Any], sigma: float) -> float:
    anchors: tuple[float, ...] = group["sigma_anchors"]
    strengths: tuple[float, ...] = group["strength_anchors"]
    if sigma >= anchors[0]:
        return strengths[0]
    if sigma <= anchors[-1]:
        return strengths[-1]

    for index in range(len(anchors) - 1):
        sigma_high = anchors[index]
        sigma_low = anchors[index + 1]
        if sigma_high >= sigma >= sigma_low:
            span = sigma_high - sigma_low
            if span == 0.0:
                return strengths[index]
            alpha = (sigma_high - sigma) / span
            return strengths[index] + alpha * (strengths[index + 1] - strengths[index])
    return strengths[-1]


def normalized_items(values: Iterable[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def rounded_strength(value: float) -> float:
    # Exact schedule anchors remain exact while insignificant interpolation noise
    # cannot spuriously invalidate a cache entry.
    return round(float(value), 10)
