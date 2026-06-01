"""PI controller design and classical tuning rules."""

from __future__ import annotations

from dataclasses import dataclass

from utils.fopdt import FOPDTModel


@dataclass(frozen=True)
class PIGains:
    """PI controller parameters."""

    Kc: float
    tauI: float


class PIController:
    """Discrete-time PI controller with saturation and basic anti-windup.

    PI control is preferred over PID for thermal chemical processes because the
    derivative term is often noise-sensitive and unnecessary when the dominant
    process dynamics are slow, noisy, and strongly integrating in the energy
    balance. Industrial reactor temperature loops therefore commonly use PI.
    """

    def __init__(self, Kc: float, tauI: float, bias: float, u_min: float = 0.0, u_max: float = 500.0):
        self.gains = PIGains(Kc=float(Kc), tauI=float(tauI))
        self.bias = float(bias)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.reset()

    def reset(self) -> None:
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.last_output = self.bias

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute the manipulated feed-flow command."""

        error = float(setpoint - measurement)
        self.integral_error += error * dt
        proportional = self.gains.Kc * error
        integral = self.gains.Kc / max(self.gains.tauI, 1e-9) * self.integral_error
        output = self.bias + proportional + integral
        saturated = min(max(output, self.u_min), self.u_max)

        if saturated != output:
            self.integral_error -= error * dt

        self.previous_error = error
        self.last_output = saturated
        return saturated



def ziegler_nichols_pi(model: FOPDTModel) -> PIGains:
    """Classic Ziegler-Nichols open-loop PI rule for FOPDT models."""

    theta = max(model.theta, 0.05 * model.tau, 1e-3)
    Kc = 0.9 * model.tau / max(model.K * theta, 1e-9)
    tauI = 3.33 * theta
    return PIGains(Kc=float(Kc), tauI=float(max(tauI, 1e-6)))



def cohen_coon_pi(model: FOPDTModel) -> PIGains:
    """Cohen-Coon PI rule for FOPDT models."""

    theta = max(model.theta, 0.05 * model.tau, 1e-3)
    ratio = theta / max(model.tau, 1e-9)
    Kc = (1.0 / max(model.K, 1e-9)) * (model.tau / theta) * (0.9 + ratio / 12.0) / (1.0 + ratio / 30.0)
    tauI = theta * (30.0 + 3.0 * ratio) / (9.0 + 20.0 * ratio)
    return PIGains(Kc=float(Kc), tauI=float(max(tauI, 1e-6)))



def imc_pi(model: FOPDTModel, lambda_cl: float | None = None) -> PIGains:
    """Internal-model-control PI tuning for FOPDT models."""

    if lambda_cl is None:
        lambda_cl = max(model.theta, 0.25 * model.tau)
    Kc = model.tau / (max(model.K, 1e-9) * (lambda_cl + model.theta))
    tauI = model.tau
    return PIGains(Kc=float(Kc), tauI=float(max(tauI, 1e-6)))