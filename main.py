"""End-to-end orchestration script for the CSTR control project."""

from __future__ import annotations

from pathlib import Path

import control
import numpy as np

from controllers.pi import PIGains, cohen_coon_pi, imc_pi, ziegler_nichols_pi
from models.cstr import CSTRModel
from models.parameters import default_parameters
from plots.visualization import (
    configure_matplotlib,
    plot_bode,
    plot_closed_loop_comparison,
    plot_linear_closed_loop_comparison,
    plot_gain_history,
    plot_linear_step,
    plot_metrics_table,
    plot_open_loop,
    plot_pole_zero_map,
    plot_reward_curve,
    plot_root_locus,
)
from rl.env import CSTRPITuningEnv, EnvironmentConfig
from rl.train import train_ddpg_agent
from simulation.analysis import (
    closed_loop_linear_response,
    classical_closed_loop_analysis,
    dominant_second_order_characteristics,
    linear_analysis,
    simulate_open_loop_disturbance,
)
from utils.fopdt import estimate_fopdt_from_step
from utils.metrics import format_metric_rows


ARTIFACTS = Path("artifacts")
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"



def select_operating_point(model: CSTRModel) -> np.ndarray:
    """Find a steady-state candidate for analysis and control design."""

    candidates = model.find_steady_state_candidates()
    if not candidates:
        state, _ = model.steady_state()
        return state

    preferred = sorted(candidates, key=lambda item: (abs(item.T - 360.0), np.max(np.real(item.eigenvalues))))[0]
    print("\nSteady-state candidates:")
    for candidate in candidates:
        dominant = float(np.max(np.real(candidate.eigenvalues)))
        print(f"  Ca={candidate.Ca:.4f}, T={candidate.T:.2f} K, dominant eigenvalue={dominant:.4f}")
    return np.array([preferred.Ca, preferred.T], dtype=float)



def main() -> None:
    configure_matplotlib()
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    params = default_parameters()
    model = CSTRModel(params)
    operating_point = select_operating_point(model)
    setpoint = float(operating_point[1] + 5.0)

    print(f"\nSelected operating point: Ca={operating_point[0]:.4f} mol/L, T={operating_point[1]:.2f} K")
    print(f"Control target setpoint: Tsp={setpoint:.2f} K")

    print("\nRunning nonlinear open-loop disturbance simulation...")
    t_ol, Ca_ol, T_ol, _, _ = simulate_open_loop_disturbance(model, operating_point)
    plot_open_loop(t_ol, T_ol, Ca_ol, FIGURES / "open_loop_response.png")

    print("Computing linearized model and open-loop transfer function...")
    linear_result = linear_analysis(model, operating_point, params.Tc0)
    plot_linear_step(linear_result.step_time, linear_result.step_response, FIGURES / "linear_step_response.png")
    plot_pole_zero_map(linear_result.poles, linear_result.zeros, FIGURES / "pole_zero_map.png")
    plot_bode(linear_result.transfer_function, FIGURES / "bode_open_loop.png")
    plot_root_locus(linear_result.transfer_function, FIGURES / "root_locus.png")

    zeta_open, wn_open = dominant_second_order_characteristics(linear_result.poles)
    print(f"Open-loop dominant second-order estimate: zeta={zeta_open:.4f}, wn={wn_open:.4f}")

    print("Estimating FOPDT approximation from the linearized step response...")
    fopdt = estimate_fopdt_from_step(linear_result.step_time, linear_result.step_response, du=1.0)
    print(f"FOPDT estimate: K={fopdt.K:.4f}, tau={fopdt.tau:.4f}, theta={fopdt.theta:.4f}")

    tuning_map = {
        "ZN-PI": ziegler_nichols_pi(fopdt),
        "Cohen-Coon PI": cohen_coon_pi(fopdt),
        "IMC-PI": imc_pi(fopdt),
    }

    print("\nClassical PI tuning results:")
    for name, gains in tuning_map.items():
        print(f"  {name:<14} Kc={gains.Kc:.4f}, tauI={gains.tauI:.4f}")

    classical_results = classical_closed_loop_analysis(
        model=model,
        operating_point=operating_point,
        tuning_map=tuning_map,
        setpoint=setpoint,
    )
    classical_metrics = {name: payload["metrics"] for name, payload in classical_results.items()}
    print("\nClassical controller performance:")
    print(format_metric_rows(classical_metrics))
    plot_closed_loop_comparison(classical_results, setpoint, FIGURES / "classical_closed_loop_comparison.png")
    plot_metrics_table(classical_metrics, FIGURES / "classical_metrics_table.png")

    linear_closed_loop_results = {}
    for name, gains in tuning_map.items():
        closed_loop_tf, cl_time, cl_response = closed_loop_linear_response(
            linear_result.transfer_function, gains, np.linspace(0.0, 60.0, 600)
        )
        zeta, wn = dominant_second_order_characteristics(np.asarray(control.poles(closed_loop_tf), dtype=complex))
        print(f"  Linear {name:<14} zeta={zeta:.4f}, wn={wn:.4f}")
        linear_closed_loop_results[name] = {
            "tf": closed_loop_tf,
            "time": cl_time,
            "response": cl_response,
        }
    plot_linear_closed_loop_comparison(linear_closed_loop_results, FIGURES / "linear_closed_loop_comparison.png")

    print("\nStarting reinforcement learning based PI optimization with DDPG...")
    rl_config = EnvironmentConfig(setpoint_temperature=setpoint)
    env = CSTRPITuningEnv(model, rl_config)
    rl_model, history = train_ddpg_agent(env, total_timesteps=20000, model_path=MODELS / "ddpg_cstr_pi")
    plot_reward_curve(history.episode_rewards, FIGURES / "rl_reward_convergence.png")
    plot_gain_history(history.action_trace, FIGURES / "rl_gain_evolution.png")

    rl_gains = PIGains(*history.best_gains)
    print(f"\nRL optimized gains: Kc={rl_gains.Kc:.4f}, tauI={rl_gains.tauI:.4f}")

    rl_results = classical_closed_loop_analysis(
        model=model,
        operating_point=operating_point,
        tuning_map={"RL-Optimized PI": rl_gains},
        setpoint=setpoint,
    )
    rl_metrics = rl_results["RL-Optimized PI"]["metrics"]

    comparison_metrics = {
        **classical_metrics,
        "RL-Optimized PI": rl_metrics,
    }
    print("\nFinal comparison table:")
    print(format_metric_rows(comparison_metrics))
    plot_closed_loop_comparison({**classical_results, **rl_results}, setpoint, FIGURES / "all_closed_loop_comparison.png")
    plot_metrics_table(comparison_metrics, FIGURES / "all_metrics_table.png")

    print("\nArtifacts saved to:")
    print(f"  Figures: {FIGURES.resolve()}")
    print(f"  Models:  {MODELS.resolve()}")
    print("\nEnd-to-end workflow complete.")


if __name__ == "__main__":
    main()