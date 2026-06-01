"""Plotting helpers for the CSTR project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import control
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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
    fig, axes = plt.subplots(3, 1, sharex=True)
    reactor_temperature = temperature[0]
    jacket_temperature = temperature[1]
    reactant_concentration = concentration[0]
    axes[0].plot(time, reactor_temperature, color="#b23a48", lw=2)
    axes[0].set_ylabel("T_R [K]")
    axes[0].set_title("Nonlinear Open-Loop Disturbance Response")
    axes[1].plot(time, jacket_temperature, color="#457b9d", lw=2)
    axes[1].set_ylabel("T_J [K]")
    axes[2].plot(time, reactant_concentration, color="#264653", lw=2)
    axes[2].set_ylabel("C_A [mol/L]")
    axes[2].set_xlabel("Time [min]")
    save_figure(fig, path)
    plt.close(fig)



def plot_open_vs_closed_loop(
    open_loop_time: np.ndarray,
    open_loop_temperature: np.ndarray,
    closed_loop_time: np.ndarray,
    closed_loop_temperature: np.ndarray,
    setpoint: float,
    path: Path,
) -> None:
    """Compare nonlinear open-loop and closed-loop temperature trajectories."""

    fig, ax = plt.subplots()
    ax.plot(open_loop_time, open_loop_temperature, color="#8d99ae", lw=2.2, label="Open loop")
    ax.plot(closed_loop_time, closed_loop_temperature, color="#1d3557", lw=2.4, label="PI closed loop")
    ax.axhline(setpoint, color="#b23a48", ls="--", lw=1.4, label="Setpoint")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Reactor temperature [K]")
    ax.set_title("Open-Loop vs Closed-Loop Temperature Response")
    ax.legend()
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
    ax.set_title("DDPG Reward Convergence")
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



def plot_process_control_diagram(setpoint: float, path: Path) -> None:
    """Draw a compact process-control diagram for the temperature loop."""

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    reactor = Rectangle((6.1, 2.1), 3.2, 3.7, linewidth=2.0, edgecolor="#1d3557", facecolor="#edf2f7")
    controller = Rectangle((1.1, 5.2), 2.8, 1.2, linewidth=1.8, edgecolor="#b23a48", facecolor="#fdecea")
    valve = Rectangle((3.8, 3.5), 1.3, 1.0, linewidth=1.6, edgecolor="#264653", facecolor="#e0f2f1")
    sensor = Rectangle((9.8, 5.1), 1.4, 0.8, linewidth=1.4, edgecolor="#7a7a7a", facecolor="#f4f4f4")
    jacket = Rectangle((6.0, 0.7), 3.4, 1.0, linewidth=1.6, edgecolor="#457b9d", facecolor="#edf6fb")

    for patch in (reactor, controller, valve, sensor, jacket):
        ax.add_patch(patch)

    ax.text(7.7, 4.1, "Exothermic CSTR", ha="center", va="center", fontsize=14, weight="bold", color="#1d3557")
    ax.text(7.7, 3.4, "States: C_A, T_R, T_J", ha="center", va="center", fontsize=11)
    ax.text(2.5, 5.8, "PI Controller", ha="center", va="center", fontsize=12, weight="bold", color="#b23a48")
    ax.text(4.45, 4.0, "Control\nValve", ha="center", va="center", fontsize=10)
    ax.text(10.5, 5.5, "Temperature\nSensor", ha="center", va="center", fontsize=10)
    ax.text(7.7, 1.15, "Cooling Jacket", ha="center", va="center", fontsize=11, color="#457b9d")
    ax.text(13.1, 4.9, f"Setpoint: {setpoint:.1f} K", ha="center", va="center", fontsize=10)

    arrows = [
        FancyArrowPatch((0.4, 4.0), (3.8, 4.0), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#264653"),
        FancyArrowPatch((5.1, 4.0), (6.1, 4.0), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#264653"),
        FancyArrowPatch((7.7, 2.1), (7.7, 1.7), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#457b9d"),
        FancyArrowPatch((7.7, 5.8), (7.7, 6.7), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#1d3557"),
        FancyArrowPatch((4.5, 5.2), (4.5, 4.5), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#b23a48"),
        FancyArrowPatch((3.9, 5.8), (5.0, 5.8), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#b23a48"),
        FancyArrowPatch((9.2, 4.1), (9.8, 4.1), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#7a7a7a"),
        FancyArrowPatch((11.2, 5.5), (13.2, 5.5), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#7a7a7a"),
    ]
    for arrow in arrows:
        ax.add_patch(arrow)

    ax.text(0.4, 4.3, "T_R setpoint", ha="left", va="bottom", fontsize=10)
    ax.text(12.25, 5.8, "T_R measurement", ha="left", va="center", fontsize=10)
    ax.text(12.25, 1.15, "Coolant flow F_C", ha="left", va="center", fontsize=10)
    ax.text(7.7, 7.0, f"Setpoint T_sp = {setpoint:.1f} K", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.text(1.1, 3.75, "F", ha="left", va="bottom", fontsize=11, weight="bold")
    ax.text(4.05, 2.65, "F_C", ha="center", va="top", fontsize=11, weight="bold", color="#264653")
    ax.text(11.2, 5.95, "Controlled variable: T_R", ha="right", va="bottom", fontsize=10)
    ax.text(11.2, 0.35, "Heat transfer: Q = UA(T_R - T_J)", ha="right", va="bottom", fontsize=10)

    save_figure(fig, path)
    plt.close(fig)