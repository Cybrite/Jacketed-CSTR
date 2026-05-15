"""First-order-plus-dead-time approximation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FOPDTModel:
    """Approximate process model G(s) = K exp(-theta s)/(tau s + 1)."""

    K: float
    tau: float
    theta: float



def estimate_fopdt_from_step(time: Iterable[float], response: Iterable[float], du: float = 1.0) -> FOPDTModel:
    """Estimate FOPDT parameters from a monotone step response."""

    t = np.asarray(list(time), dtype=float)
    y = np.asarray(list(response), dtype=float)
    y0 = float(y[0])
    yss = float(y[-1])
    delta_y = yss - y0
    if abs(du) < 1e-12:
        raise ValueError("Input step size du must be nonzero.")
    if abs(delta_y) < 1e-12:
        raise ValueError("Response change is too small to identify a FOPDT model.")

    gain = delta_y / du
    y28 = y0 + 0.283 * delta_y
    y63 = y0 + 0.632 * delta_y

    t28 = _interpolate_crossing_time(t, y, y28)
    t63 = _interpolate_crossing_time(t, y, y63)

    tau = max(1e-6, 1.5 * (t63 - t28))
    theta = max(0.0, t63 - tau)

    return FOPDTModel(K=float(gain), tau=float(tau), theta=float(theta))



def _interpolate_crossing_time(time: np.ndarray, signal: np.ndarray, level: float) -> float:
    for index in range(1, len(signal)):
        y0 = signal[index - 1]
        y1 = signal[index]
        if (y0 - level) == 0:
            return float(time[index - 1])
        if (y0 - level) * (y1 - level) <= 0:
            ratio = (level - y0) / (y1 - y0 + 1e-12)
            return float(time[index - 1] + ratio * (time[index] - time[index - 1]))
    return float(time[-1])