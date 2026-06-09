"""End-to-end orchestration script for the CSTR control project."""

from __future__ import annotations

from pathlib import Path

import control
import numpy as np

from controllers.pi import imc_pi, ziegler_nichols_pi
from models.cstr import CSTRModel
from models.parameters import default_parameters
from plots.visualization import (
    configure_matplotlib,
    plot_bode,
    plot_closed_loop_disturbance_rejection,
    plot_closed_loop_comparison,
    plot_open_vs_closed_loop,
    plot_linear_closed_loop_comparison,
    plot_linear_step,
    plot_metrics_table,
    plot_open_loop,
    plot_pole_zero_map,
    plot_root_locus,
    plot_process_control_diagram,
    plot_reward_curve,
    plot_gain_history
)
from simulation.analysis import (
    closed_loop_linear_response,
    classical_closed_loop_analysis,
    dominant_second_order_characteristics,
    linear_analysis,
    simulate_closed_loop_disturbance_rejection,
    simulate_open_loop_disturbance,
    simulate_closed_loop_policy,
    closed_loop_metrics
)
from simulation.derivation import build_derivation_report
from utils.fopdt import estimate_fopdt_from_step
from utils.metrics import format_metric_rows

from rl.env import CSTRPITuningEnv, EnvironmentConfig
from rl.q_learning import train_q_learning_agent


ARTIFACTS = Path("artifacts")
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"


def select_operating_point(model: CSTRModel) -> np.ndarray:
    """Find a steady-state candidate for analysis and control design."""
    candidates = model.find_steady_state_candidates(F=model.params.F, Fc=model.params.Fc, Tcin=model.params.Tcin0)
    if not candidates:
        state, _ = model.steady_state(F=model.params.F, Fc=model.params.Fc, Tcin=model.params.Tcin0)
        return state

    stable_candidates = [candidate for candidate in candidates if np.max(np.real(candidate.eigenvalues)) < 0.0]
    ranked_candidates = stable_candidates if stable_candidates else candidates
    preferred = sorted(
        ranked_candidates,
        key=lambda item: (abs(item.TR - 325.0), abs(np.max(np.real(item.eigenvalues)))),
    )[0]
    return np.array([preferred.Ca, preferred.TR, preferred.TJ], dtype=float)


def main() -> None:
    configure_matplotlib()
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    params = default_parameters()
    model = CSTRModel(params)
    operating_point = select_operating_point(model)
    Fss = float(params.F)
    Fcss = float(model.coolant_flow_for_steady_state(operating_point, params.Tcin0))
    setpoint = float(operating_point[1] + 5.0)

    print(f"\nSelected operating point: Ca={operating_point[0]:.4f} mol/L, TR={operating_point[1]:.2f} K")
    print(f"Control target setpoint: Tsp={setpoint:.2f} K")

    report = build_derivation_report(params, operating_point, Fss, Fcss)
    plot_process_control_diagram(setpoint, FIGURES / "process_control_diagram.png")

    print("\nRunning nonlinear open-loop disturbance simulation...")
    t_ol, Ca_ol, TR_ol, TJ_ol, Fc_ol, disturbances = simulate_open_loop_disturbance(model, operating_point)
    plot_open_loop(t_ol, np.vstack([TR_ol, TJ_ol]), np.vstack([Ca_ol, disturbances[0], disturbances[2]]), FIGURES / "open_loop_response.png")

    print("Computing linearized model and open-loop transfer function...")
    linear_result = linear_analysis(model, operating_point, Fss, Fcss, params.Tcin0)
    plot_linear_step(linear_result.step_time, linear_result.step_response, FIGURES / "linear_step_response.png")
    plot_pole_zero_map(linear_result.poles, linear_result.zeros, FIGURES / "pole_zero_map.png")
    plot_bode(linear_result.transfer_function, FIGURES / "bode_open_loop.png")
    plot_root_locus(linear_result.transfer_function, FIGURES / "root_locus.png")

    fopdt = estimate_fopdt_from_step(linear_result.step_time, linear_result.step_response, du=1.0)
    tuning_map = {
        "ZN-PI": ziegler_nichols_pi(fopdt),
        "IMC-PI": imc_pi(fopdt),
    }

    classical_results = classical_closed_loop_analysis(
        model=model,
        operating_point=operating_point,
        tuning_map=tuning_map,
        setpoint=setpoint,
    )
    
    imc_closed_loop_time = classical_results["IMC-PI"]["time"]
    imc_closed_loop_temperature = classical_results["IMC-PI"]["temperature"]
    plot_open_vs_closed_loop(t_ol, TR_ol, imc_closed_loop_time, imc_closed_loop_temperature, setpoint, FIGURES / "open_vs_closed_loop.png")

    # --- REINFORCEMENT LEARNING INTEGRATION (TABULAR Q-LEARNING) ---
    print("\nTraining Tabular Q-Learning Agent. Please wait...")
    rl_config = EnvironmentConfig(
        dt=float(params.dt),
        episode_steps=int(params.episode_steps),
        safety_temperature=float(params.T_safe),
        setpoint_temperature=setpoint,
    )
    q_env = CSTRPITuningEnv(model, rl_config)
    q_agent, q_history = train_q_learning_agent(q_env, episodes=2000)
    
    plot_reward_curve(q_history.episode_rewards, FIGURES / "q_learning_reward_curve.png")

    # Macro-step holding wrapper for evaluation rollouts
    class QPolicyHoldWrapper:
        def __init__(self, agent, hold_steps=10):
            self.agent = agent
            self.hold_steps = hold_steps
            self.current_gains = None
            self.counter = 0

        def __call__(self, obs):
            if self.current_gains is None or self.counter % self.hold_steps == 0:
                state_idx = self.agent.discretize_observation(obs)
                kc_idx, tau_idx = self.agent.select_action_indices(state_idx, explore=False)
                self.current_gains = np.array([self.agent.kc_actions[kc_idx], self.agent.tau_actions[tau_idx]])
            self.counter += 1
            return self.current_gains

    q_policy = QPolicyHoldWrapper(q_agent, hold_steps=10)

    print("Evaluating Q-Learning step response...")
    # FIX: Explicitly set observation_horizon=20.0 to match the training environment scaling bounds
    q_time, q_temp, q_flow, q_gains, _ = simulate_closed_loop_policy(
        model=model,
        policy_fn=q_policy,
        setpoint=setpoint,
        operating_point=operating_point,
        time_final=80.0,
        dt=float(params.dt),
        observation_horizon=20.0,
    )
    plot_gain_history(q_gains, FIGURES / "q_learning_gain_history.png")

    # Combine Classical and RL for Step Response Analysis
    rl_results = {
        "Q-learning": {
            "time": q_time,
            "temperature": q_temp,
            "flow": q_flow,
            "gains": q_gains,
            "metrics": closed_loop_metrics(q_time, q_temp, setpoint),
        }
    }
    
    combined_results = dict(classical_results)
    combined_results.update(rl_results)
    combined_metrics = {name: payload["metrics"] for name, payload in combined_results.items()}
    
    print("\nController Performance Matrix:")
    print(format_metric_rows(combined_metrics))
    
    plot_closed_loop_comparison(combined_results, setpoint, FIGURES / "classical_and_rl_closed_loop_comparison.png")
    plot_metrics_table(combined_metrics, FIGURES / "classical_and_rl_metrics_table.png")

    # --- DISTURBANCE REJECTION ANALYSIS ---
    print("\nRunning Disturbance Rejection Analysis for all controllers...")
    disturbance_cases = {
        "Feed concentration step": "ca0",
        "Feed temperature step": "t0",
    }
    disturbance_plot_map = {
        "Feed concentration step": FIGURES / "closed_loop_disturbance_rejection_ca0.png",
        "Feed temperature step": FIGURES / "closed_loop_disturbance_rejection_t0.png",
    }
    
    for label, disturbance_key in disturbance_cases.items():
        rejection_results = {}
        
        # 1. Simulate ZN and IMC
        for name, gains in tuning_map.items():
            time, temperature, flow, disturbance_trace = simulate_closed_loop_disturbance_rejection(
                model=model,
                gains=gains,
                setpoint=setpoint,
                operating_point=operating_point,
                disturbance=disturbance_key,
            )
            rejection_results[name] = {
                "time": time,
                "temperature": temperature,
                "flow": flow,
                "disturbance": disturbance_trace,
            }
            
        # 2. Simulate RL Q-Learning Agent
        # FIX: Explicitly set observation_horizon=20.0 here as well to balance disturbances safely
        rl_d_time, rl_d_temp, rl_d_flow, _, rl_d_trace = simulate_closed_loop_policy(
            model=model,
            policy_fn=q_policy,
            setpoint=setpoint,
            operating_point=operating_point,
            time_final=40.0,
            dt=float(params.dt),
            disturbance=disturbance_key,
            observation_horizon=20.0,
        )
        
        rejection_results["Q-learning"] = {
            "time": rl_d_time,
            "temperature": rl_d_temp,
            "flow": rl_d_flow,
            "disturbance": rl_d_trace,
        }

        # 3. Plot Combined Disturbance Rejection
        plot_closed_loop_disturbance_rejection(
            rejection_results,
            label,
            setpoint,
            disturbance_plot_map[label],
        )

    # Linear comparison mappings
    linear_closed_loop_results = {}
    for name, gains in tuning_map.items():
        closed_loop_tf, cl_time, cl_response = closed_loop_linear_response(
            linear_result.transfer_function, gains, np.linspace(0.0, 60.0, 600)
        )
        linear_closed_loop_results[name] = {
            "tf": closed_loop_tf,
            "time": cl_time,
            "response": cl_response,
        }
    plot_linear_closed_loop_comparison(linear_closed_loop_results, FIGURES / "linear_closed_loop_comparison.png")

    print("\nArtifacts securely saved to:")
    print(f"  Figures: {FIGURES.resolve()}")
    print(f"  Models:  {MODELS.resolve()}")
    print("\nEnd-to-end workflow complete.")

if __name__ == "__main__":
    main()