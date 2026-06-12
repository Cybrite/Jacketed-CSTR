"""Plotting helpers for the CSTR project."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import control
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from utils.metrics import ResponseMetrics

COLOR_MAP = {
    "ZN-PI": "#b23a48",      
    "IMC-PI": "#1d3557",     
    "Q-Learning": "#2a9d8f", 
    "DDPG": "#e76f51",       
    "SAC": "#e9c46a",        
}

def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.figsize": (10, 6), "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelsize": 11, "legend.frameon": False, "font.size": 10})

def save_figure(fig, path): path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight")

def plot_open_loop(time, temperature, concentration, path):
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(time, temperature[0], color="#b23a48", lw=2); axes[0].set_ylabel("T_R [K]"); axes[0].set_title("Nonlinear Open-Loop Disturbance Response")
    axes[1].plot(time, temperature[1], color="#457b9d", lw=2); axes[1].set_ylabel("T_J [K]")
    axes[2].plot(time, concentration[0], color="#264653", lw=2); axes[2].set_ylabel("C_A [mol/L]"); axes[2].set_xlabel("Time [min]")
    save_figure(fig, path); plt.close(fig)

def plot_open_vs_closed_loop(ol_t, ol_T, cl_t, cl_T, sp, path):
    fig, ax = plt.subplots()
    ax.plot(ol_t, ol_T, color="#8d99ae", lw=2.2, label="Open loop")
    ax.plot(cl_t, cl_T, color=COLOR_MAP["IMC-PI"], lw=2.4, label="PI closed loop")
    ax.axhline(sp, color="#b23a48", ls="--", lw=1.4, label="Setpoint"); ax.set_xlabel("Time [min]"); ax.set_ylabel("Reactor temperature [K]"); ax.set_title("Open-Loop vs Closed-Loop Temperature Response"); ax.legend()
    save_figure(fig, path); plt.close(fig)

def plot_closed_loop_disturbance_rejection(results, label, setpoint, path):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for name, payload in results.items():
        color = COLOR_MAP.get(name, None)
        axes[0].plot(payload["time"], payload["temperature"], lw=2.2, label=name, color=color)
        axes[1].plot(payload["time"], payload["flow"], lw=2.0, label=name, color=color)
    ref = next(iter(results.values()))
    axes[0].axhline(setpoint, color="black", ls="--", lw=1.2, label="Setpoint")
    axes[2].plot(ref["time"], ref["disturbance"], color="#457b9d", lw=2.0)
    axes[0].set_ylabel("T_R [K]"); axes[0].set_title(f"Closed-Loop Disturbance Rejection: {label}")
    axes[1].set_ylabel("F_C [L/min]"); axes[2].set_ylabel("Disturbance"); axes[2].set_xlabel("Time [min]")
    axes[0].legend(loc="best"); axes[1].legend(loc="best")
    save_figure(fig, path); plt.close(fig)

def plot_linear_step(time, response, path):
    fig, ax = plt.subplots()
    ax.plot(time, response, color="#1d3557", lw=2); ax.set_xlabel("Time [min]"); ax.set_ylabel("Temperature deviation [K]"); ax.set_title("Linearized Open-Loop Step Response")
    save_figure(fig, path); plt.close(fig)

def plot_pole_zero_map(poles, zeros, path):
    fig, ax = plt.subplots()
    ax.scatter(np.real(poles), np.imag(poles), marker="x", s=90, color="#b23a48", label="Poles")
    if len(zeros) > 0: ax.scatter(np.real(zeros), np.imag(zeros), marker="o", s=70, facecolors="none", edgecolors="#1d3557", label="Zeros")
    ax.axhline(0.0, color="black", lw=0.8); ax.axvline(0.0, color="black", lw=0.8); ax.set_xlabel("Real axis"); ax.set_ylabel("Imaginary axis"); ax.set_title("Pole-Zero Map"); ax.legend()
    save_figure(fig, path); plt.close(fig)

def plot_bode(tf, path):
    omega = np.logspace(-3, 1, 500); resp = control.frequency_response(tf, omega); fresp = np.asarray(resp.frdata); mag = np.abs(fresp).squeeze(); phase = np.unwrap(np.angle(fresp).squeeze()) * 180.0 / np.pi
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].semilogx(omega, 20.0 * np.log10(np.maximum(mag, 1e-12)), color="#1d3557", lw=2); axes[0].set_ylabel("Magnitude [dB]"); axes[0].set_title("Bode Plot")
    axes[1].semilogx(omega, phase, color="#b23a48", lw=2); axes[1].set_ylabel("Phase [deg]"); axes[1].set_xlabel("Frequency [rad/min]")
    save_figure(fig, path); plt.close(fig)

def plot_root_locus(tf, path):
    fig, ax = plt.subplots()
    for gain in np.logspace(-3, 3, 250):
        poles = control.poles(control.feedback(gain * tf, 1))
        ax.scatter(np.real(poles), np.imag(poles), s=4, color="#1d3557", alpha=0.16)
    ax.axhline(0.0, color="black", lw=0.8); ax.axvline(0.0, color="black", lw=0.8); ax.set_xlabel("Real axis"); ax.set_ylabel("Imaginary axis"); ax.set_title("Root Locus")
    save_figure(fig, path); plt.close(fig)

def plot_closed_loop_comparison(results, setpoint, path):
    fig, ax = plt.subplots()
    for name, payload in results.items():
        ax.plot(payload["time"], payload["temperature"], lw=2, label=name, color=COLOR_MAP.get(name, None))
    ax.axhline(setpoint, color="black", ls="--", lw=1.2, label="Setpoint"); ax.set_xlabel("Time [min]"); ax.set_ylabel("Reactor temperature [K]"); ax.set_title("Closed-Loop PI Tuning Comparison"); ax.legend()
    save_figure(fig, path); plt.close(fig)

def plot_linear_closed_loop_comparison(results, path):
    fig, ax = plt.subplots()
    for name, payload in results.items():
        ax.plot(payload["time"], payload["response"], lw=2, label=name, color=COLOR_MAP.get(name, None))
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Temperature deviation [K]"); ax.set_title("Linear Closed-Loop Step Response Comparison"); ax.legend()
    save_figure(fig, path); plt.close(fig)

def plot_gain_history(actions, path):
    if actions is None or len(actions) == 0 or np.all(np.isnan(np.asarray(actions))): return
    arr = np.asarray(actions, dtype=float)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(arr[:, 0], color="#1d3557", lw=2); ax[0].set_ylabel("Kc"); ax[0].set_title("RL Controller Parameter Evolution")
    ax[1].plot(arr[:, 1], color="#b23a48", lw=2); ax[1].set_ylabel("tauI"); ax[1].set_xlabel("Time [min]")
    save_figure(fig, path); plt.close(fig)

def plot_reward_curve(rewards, path):
    if rewards is None or len(rewards) == 0: return
    fig, ax = plt.subplots()
    ax.plot(rewards, color="#264653", lw=2); ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Training Convergence")
    save_figure(fig, path); plt.close(fig)

def plot_metrics_table(metrics, path):
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.35 * len(metrics)))
    ax.axis("off")
    rows = [[n, f"{m.overshoot_percent:.2f}", f"{m.rise_time:.2f}", f"{m.settling_time:.2f}", f"{m.iae:.4f}", f"{m.ise:.4f}", f"{m.itae:.4f}"] for n, m in metrics.items()]
    table = ax.table(cellText=rows, colLabels=["Method", "OS %", "Rise", "Settling", "IAE", "ISE", "ITAE"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.35); ax.set_title("Controller Performance Summary", pad=20)
    save_figure(fig, path); plt.close(fig)

def plot_process_control_diagram(setpoint, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    for p in (Rectangle((6.1, 2.1), 3.2, 3.7, linewidth=2.0, edgecolor="#1d3557", facecolor="#edf2f7"), Rectangle((1.1, 5.2), 2.8, 1.2, linewidth=1.8, edgecolor="#b23a48", facecolor="#fdecea"), Rectangle((3.8, 3.5), 1.3, 1.0, linewidth=1.6, edgecolor="#264653", facecolor="#e0f2f1"), Rectangle((9.8, 5.1), 1.4, 0.8, linewidth=1.4, edgecolor="#7a7a7a", facecolor="#f4f4f4"), Rectangle((6.0, 0.7), 3.4, 1.0, linewidth=1.6, edgecolor="#457b9d", facecolor="#edf6fb")): ax.add_patch(p)
    ax.text(7.7, 4.1, "Exothermic CSTR", ha="center", va="center", fontsize=14, weight="bold", color="#1d3557"); ax.text(7.7, 3.4, "States: C_A, T_R, T_J", ha="center", va="center", fontsize=11); ax.text(2.5, 5.8, "PI Controller", ha="center", va="center", fontsize=12, weight="bold", color="#b23a48"); ax.text(4.45, 4.0, "Control\nValve", ha="center", va="center", fontsize=10); ax.text(10.5, 5.5, "Temperature\nSensor", ha="center", va="center", fontsize=10); ax.text(7.7, 1.15, "Cooling Jacket", ha="center", va="center", fontsize=11, color="#457b9d"); ax.text(13.1, 4.9, f"Setpoint: {setpoint:.1f} K", ha="center", va="center", fontsize=10)
    for a in (FancyArrowPatch((0.4, 4.0), (3.8, 4.0), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#264653"), FancyArrowPatch((5.1, 4.0), (6.1, 4.0), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#264653"), FancyArrowPatch((7.7, 2.1), (7.7, 1.7), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#457b9d"), FancyArrowPatch((7.7, 5.8), (7.7, 6.7), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#1d3557"), FancyArrowPatch((4.5, 5.2), (4.5, 4.5), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#b23a48"), FancyArrowPatch((3.9, 5.8), (5.0, 5.8), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#b23a48"), FancyArrowPatch((9.2, 4.1), (9.8, 4.1), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#7a7a7a"), FancyArrowPatch((11.2, 5.5), (13.2, 5.5), arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#7a7a7a")): ax.add_patch(a)
    ax.text(0.4, 4.3, "T_R setpoint", ha="left", va="bottom", fontsize=10); ax.text(12.25, 5.8, "T_R measurement", ha="left", va="center", fontsize=10); ax.text(12.25, 1.15, "Coolant flow F_C", ha="left", va="center", fontsize=10); ax.text(7.7, 7.0, f"Setpoint T_sp = {setpoint:.1f} K", ha="center", va="bottom", fontsize=11, weight="bold"); ax.text(1.1, 3.75, "F", ha="left", va="bottom", fontsize=11, weight="bold"); ax.text(4.05, 2.65, "F_C", ha="center", va="top", fontsize=11, weight="bold", color="#264653"); ax.text(11.2, 5.95, "Controlled variable: T_R", ha="right", va="bottom", fontsize=10); ax.text(11.2, 0.35, "Heat transfer: Q = UA(T_R - T_J)", ha="right", va="bottom", fontsize=10)
    save_figure(fig, path); plt.close(fig)