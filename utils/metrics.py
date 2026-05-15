"""Performance metrics for process control evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class ResponseMetrics:
    """Common step-response metrics used in process control."""

    overshoot_percent: float
    rise_time: float
    settling_time: float
    iae: float
    ise: float
    itae: float



def _first_crossing_time(time: np.ndarray, signal: np.ndarray, level: float) -> float:
    for index in range(1, len(signal)):
        y0 = signal[index - 1]
        y1 = signal[index]
        if (y0 - level) == 0:
            return float(time[index - 1])
        if (y0 - level) * (y1 - level) <= 0:
            ratio = (level - y0) / (y1 - y0 + 1e-12)
            return float(time[index - 1] + ratio * (time[index] - time[index - 1]))
    return float(time[-1])



def step_response_metrics(time: Iterable[float], response: Iterable[float], setpoint: float) -> ResponseMetrics:
    """Compute overshoot, rise time, settling time, and integral error metrics."""

    t = np.asarray(list(time), dtype=float)
    y = np.asarray(list(response), dtype=float)
    error = setpoint - y

    final_value = float(y[-1])
    amplitude = abs(setpoint - y[0]) if abs(setpoint - y[0]) > 1e-9 else 1.0

    if setpoint >= y[0]:
        lower_level = y[0] + 0.1 * amplitude
        upper_level = y[0] + 0.9 * amplitude
    else:
        lower_level = y[0] - 0.1 * amplitude
        upper_level = y[0] - 0.9 * amplitude

    rise_time = max(0.0, _first_crossing_time(t, y, lower_level) - _first_crossing_time(t, y, upper_level))

    tolerance = 0.02 * max(abs(final_value), 1.0)
    settling_time = float(t[-1])
    for index in range(len(y)):
        if np.all(np.abs(y[index:] - final_value) <= tolerance):
            settling_time = float(t[index])
            break

    if setpoint >= y[0]:
        peak = float(np.max(y))
        overshoot_percent = max(0.0, (peak - setpoint) / amplitude * 100.0)
    else:
        trough = float(np.min(y))
        overshoot_percent = max(0.0, (setpoint - trough) / amplitude * 100.0)

    iae = float(np.trapezoid(np.abs(error), t))
    ise = float(np.trapezoid(error**2, t))
    itae = float(np.trapezoid(t * error**2, t))

    return ResponseMetrics(
        overshoot_percent=overshoot_percent,
        rise_time=rise_time,
        settling_time=settling_time,
        iae=iae,
        ise=ise,
        itae=itae,
    )



def format_metric_rows(metrics: Dict[str, ResponseMetrics]) -> str:
    """Create a compact plain-text comparison table for console output."""

    header = (
        f"{'Method':<18} {'OS %':>9} {'Rise':>10} {'Settling':>10} "
        f"{'IAE':>12} {'ISE':>12} {'ITAE':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, metric in metrics.items():
        lines.append(
            f"{name:<18} {metric.overshoot_percent:9.2f} {metric.rise_time:10.2f} {metric.settling_time:10.2f} "
            f"{metric.iae:12.4f} {metric.ise:12.4f} {metric.itae:12.4f}"
        )
    return "\n".join(lines)