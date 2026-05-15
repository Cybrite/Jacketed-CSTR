"""Plotting helpers for the CSTR project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import control
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import ResponseMetrics


def configure_matplotlib() -> None:
    """Apply a professional plotting style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "font.size": 10,
        }
    )



def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")



def plot_open_loop(time: np.ndarray, temperature: np.ndarray, concentration: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time, temperature, color="#b23a48", lw=2)
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Nonlinear Open-Loop Reactor Response")
    axes[1].plot(time, concentration, color="#264653", lw=2)
    axes[1].set_ylabel("Concentration [mol/L]")
    axes[1].set_xlabel("Time [min]")
    save_figure(fig, path)
    plt.close(fig)



def plot_linear_step(time: np.ndarray, response: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(time, response, color="#1d3557", lw=2)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Temperature deviation [K]")
    ax.set_title("Linearized Open-Loop Step Response")
    save_figure(fig, path)
    plt.close(fig)



def plot_pole_zero_map(poles: Sequence[complex], zeros: Sequence[complex], path: Path) -> None:
    fig, ax = plt.subplots()
    poles = np.asarray(poles)
    zeros = np.asarray(zeros)
    ax.scatter(np.real(poles), np.imag(poles), marker="x", s=90, color="#b23a48", label="Poles")
    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), marker="o", s=70, facecolors="none", edgecolors="#1d3557", label="Zeros")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")
    ax.set_title("Pole-Zero Map")
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)



def plot_bode(transfer_function, path: Path) -> None:
    omega = np.logspace(-3, 1, 500)
    response = control.frequency_response(transfer_function, omega)
    fresp = np.asarray(response.frdata)
    magnitude = np.abs(fresp).squeeze()
    phase = np.unwrap(np.angle(fresp).squeeze()) * 180.0 / np.pi

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].semilogx(omega, 20.0 * np.log10(np.maximum(magnitude, 1e-12)), color="#1d3557", lw=2)
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title("Bode Plot")
    axes[1].semilogx(omega, phase, color="#b23a48", lw=2)
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_xlabel("Frequency [rad/min]")
    save_figure(fig, path)
    plt.close(fig)



def plot_root_locus(transfer_function, path: Path) -> None:
    gains = np.logspace(-3, 3, 250)
    pole_cloud = []
    for gain in gains:
        closed_loop = control.feedback(gain * transfer_function, 1)
        pole_cloud.append(control.poles(closed_loop))
    fig, ax = plt.subplots()
    for poles in pole_cloud:
        ax.scatter(np.real(poles), np.imag(poles), s=4, color="#1d3557", alpha=0.16)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")
    ax.set_title("Root Locus")
    save_figure(fig, path)
    plt.close(fig)



def plot_closed_loop_comparison(results: Dict[str, Dict[str, object]], setpoint: float, path: Path) -> None:
    fig, ax = plt.subplots()
    for name, payload in results.items():
        ax.plot(payload["time"], payload["temperature"], lw=2, label=name)
    ax.axhline(setpoint, color="black", ls="--", lw=1.2, label="Setpoint")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Reactor temperature [K]")
    ax.set_title("Closed-Loop PI Tuning Comparison")
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)



def plot_linear_closed_loop_comparison(results: Dict[str, Dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots()
    for name, payload in results.items():
        ax.plot(payload["time"], payload["response"], lw=2, label=name)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Temperature deviation [K]")
    ax.set_title("Linear Closed-Loop Step Response Comparison")
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)



def plot_gain_history(actions: Sequence[Sequence[float]], path: Path) -> None:
    if not actions:
        return
    arr = np.asarray(actions, dtype=float)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(arr[:, 0], color="#1d3557", lw=2)
    ax[0].set_ylabel("Kc")
    ax[1].plot(arr[:, 1], color="#b23a48", lw=2)
    ax[1].set_ylabel("tauI")
    ax[1].set_xlabel("Training step")
    ax[0].set_title("RL Controller Parameter Evolution")
    save_figure(fig, path)
    plt.close(fig)



def plot_reward_curve(rewards: Sequence[float], path: Path) -> None:
    if not rewards:
        return
    fig, ax = plt.subplots()
    ax.plot(rewards, color="#264653", lw=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DQN Reward Convergence")
    save_figure(fig, path)
    plt.close(fig)



def plot_metrics_table(metrics: Dict[str, ResponseMetrics], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.35 * len(metrics)))
    ax.axis("off")
    rows = []
    for name, metric in metrics.items():
        rows.append([
            name,
            f"{metric.overshoot_percent:.2f}",
            f"{metric.rise_time:.2f}",
            f"{metric.settling_time:.2f}",
            f"{metric.iae:.4f}",
            f"{metric.ise:.4f}",
            f"{metric.itae:.4f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Method", "OS %", "Rise", "Settling", "IAE", "ISE", "ITAE"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Controller Performance Summary", pad=20)
    save_figure(fig, path)
    plt.close(fig)