"""End-to-end orchestration script for the CSTR control project."""

from __future__ import annotations
from pathlib import Path
import control
import numpy as np
import random
import torch

from controllers.pi import imc_pi, ziegler_nichols_pi, PIGains
from models.cstr import CSTRModel
from models.parameters import default_parameters
from plots.visualization import (
    configure_matplotlib, plot_bode, plot_closed_loop_disturbance_rejection,
    plot_closed_loop_comparison, plot_open_vs_closed_loop,
    plot_linear_closed_loop_comparison, plot_linear_step, plot_metrics_table,
    plot_open_loop, plot_pole_zero_map, plot_root_locus, plot_process_control_diagram,
)
from simulation.analysis import (
    closed_loop_linear_response, classical_closed_loop_analysis,
    linear_analysis, simulate_closed_loop_disturbance_rejection,
    simulate_open_loop_disturbance, simulate_closed_loop_policy, closed_loop_metrics
)
from simulation.derivation import build_derivation_report
from utils.fopdt import estimate_fopdt_from_step
from utils.metrics import format_metric_rows

from rl.env import CSTRPITuningEnv, EnvironmentConfig
from rl.q_learning import train_q_learning_agent
from rl.train import train_ddpg_agent, train_sac_agent

ARTIFACTS = Path("artifacts")
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"

def select_operating_point(model: CSTRModel) -> np.ndarray:
    candidates = model.find_steady_state_candidates(F=model.params.F, Fc=model.params.Fc, Tcin=model.params.Tcin0)
    stable = [c for c in candidates if np.max(np.real(c.eigenvalues)) < 0.0] if candidates else []
    if not stable:
        state, _ = model.steady_state(F=model.params.F, Fc=model.params.Fc, Tcin=model.params.Tcin0)
        return state
    preferred = sorted(stable, key=lambda item: (abs(item.TR - 325.0), abs(np.max(np.real(item.eigenvalues)))))[0]
    return np.array([preferred.Ca, preferred.TR, preferred.TJ], dtype=float)

def main() -> None:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    configure_matplotlib()
    ARTIFACTS.mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)

    params = default_parameters()
    model = CSTRModel(params)
    operating_point = select_operating_point(model)
    Fss, Fcss = float(params.F), float(model.coolant_flow_for_steady_state(operating_point, params.Tcin0))
    setpoint = float(operating_point[1] + 5.0)

    print(f"\nSelected operating point: Ca={operating_point[0]:.4f} mol/L, TR={operating_point[1]:.2f} K\nControl target setpoint: Tsp={setpoint:.2f} K")
    build_derivation_report(params, operating_point, Fss, Fcss)
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
    tuning_map = {"ZN-PI": ziegler_nichols_pi(fopdt), "IMC-PI": imc_pi(fopdt)}
    classical_results = classical_closed_loop_analysis(model, operating_point, tuning_map, setpoint)
    plot_open_vs_closed_loop(t_ol, TR_ol, classical_results["IMC-PI"]["time"], classical_results["IMC-PI"]["temperature"], setpoint, FIGURES / "open_vs_closed_loop.png")

    imc_gains = tuning_map["IMC-PI"]

    # THE FIX: Residual RL anchoring. We allow the RL agent to vary the IMC parameters by +/- 60%
    rl_config = EnvironmentConfig(
        dt=float(params.dt), episode_steps=int(params.episode_steps), 
        safety_temperature=float(params.T_safe), setpoint_temperature=setpoint,
        baseline_Kc=imc_gains.Kc, baseline_tauI=imc_gains.tauI,
        delta_Kc=abs(imc_gains.Kc) * 0.60, delta_tauI=abs(imc_gains.tauI) * 0.60
    )
    rl_env = CSTRPITuningEnv(model, rl_config)

    print("\nTraining Tabular Q-Learning Agent. Please wait...")
    q_agent, q_history = train_q_learning_agent(rl_env, episodes=1500)
    print("Training Continuous DDPG Agent. Please wait...")
    ddpg_agent = train_ddpg_agent(rl_env, total_timesteps=25000)
    print("Training Soft Actor-Critic (SAC) Agent. Please wait...")
    sac_agent = train_sac_agent(rl_env, total_timesteps=25000)

    class PolicyHoldWrapper:
        def __init__(self, policy_fn, hold_steps=10):
            self.policy_fn, self.hold_steps, self.current_gains, self.counter = policy_fn, hold_steps, None, 0
        def __call__(self, obs):
            if self.current_gains is None or self.counter % self.hold_steps == 0:
                self.current_gains = self.policy_fn(obs)
            self.counter += 1
            return self.current_gains

    def q_fn(obs):
        state_idx = q_agent.discretize_observation(obs)
        kc_idx, tau_idx = q_agent.select_action_indices(state_idx, explore=False)
        return np.array([q_agent.kc_actions[kc_idx], q_agent.tau_actions[tau_idx]])
    
    def ddpg_fn(obs): 
        return ddpg_agent.predict(obs, deterministic=True)[0]
        
    def sac_fn(obs): 
        return sac_agent.predict(obs, deterministic=True)[0]

    rl_agents = {"Q-Learning": PolicyHoldWrapper(q_fn), "DDPG": PolicyHoldWrapper(ddpg_fn), "SAC": PolicyHoldWrapper(sac_fn)}
    rl_results = {}
    print("\nEvaluating RL Agents step responses...")
    
    for agent_name, policy in rl_agents.items():
        time_arr, temp_arr, flow_arr, gains_arr, _ = simulate_closed_loop_policy(
            model, policy, setpoint, operating_point, time_final=80.0, dt=float(params.dt), observation_horizon=20.0, 
            baseline_Kc=imc_gains.Kc, baseline_tauI=imc_gains.tauI, 
            delta_Kc=abs(imc_gains.Kc) * 0.60, delta_tauI=abs(imc_gains.tauI) * 0.60
        )
        rl_results[agent_name] = {"time": time_arr, "temperature": temp_arr, "flow": flow_arr, "gains": gains_arr, "metrics": closed_loop_metrics(time_arr, temp_arr, setpoint)}
    
    combined_results = dict(classical_results)
    combined_results.update(rl_results)
    combined_metrics = {name: payload["metrics"] for name, payload in combined_results.items()}
    
    print("\nController Performance Matrix:\n" + format_metric_rows(combined_metrics))
    plot_closed_loop_comparison(combined_results, setpoint, FIGURES / "classical_and_rl_closed_loop_comparison.png")
    plot_metrics_table(combined_metrics, FIGURES / "classical_and_rl_metrics_table.png")

    print("\nRunning Disturbance Rejection Analysis for all controllers...")
    disturbance_cases = {"Feed concentration step": "ca0", "Feed temperature step": "t0"}
    disturbance_plot_map = {"Feed concentration step": FIGURES / "closed_loop_disturbance_rejection_ca0.png", "Feed temperature step": FIGURES / "closed_loop_disturbance_rejection_t0.png"}
    
    for label, disturbance_key in disturbance_cases.items():
        rejection_results = {}
        for name, gains in tuning_map.items():
            time, temp, flow, trace = simulate_closed_loop_disturbance_rejection(model, gains, setpoint, operating_point, disturbance_key)
            rejection_results[name] = {"time": time, "temperature": temp, "flow": flow, "disturbance": trace}
            
        for agent_name in rl_agents.keys():
            # Freezes RL gains for smooth Disturbance Graphs
            final_kc = float(rl_results[agent_name]["gains"][-1][0])
            final_tauI = float(rl_results[agent_name]["gains"][-1][1])
            optimized_gains = PIGains(Kc=final_kc, tauI=final_tauI)

            rl_d_time, rl_d_temp, rl_d_flow, rl_d_trace = simulate_closed_loop_disturbance_rejection(
                model, optimized_gains, setpoint, operating_point, disturbance_key
            )
            rejection_results[agent_name] = {"time": rl_d_time, "temperature": rl_d_temp, "flow": rl_d_flow, "disturbance": rl_d_trace}
            
        plot_closed_loop_disturbance_rejection(rejection_results, label, setpoint, disturbance_plot_map[label])

    linear_closed_loop_results = {}
    for name, gains in tuning_map.items():
        closed_loop_tf, cl_time, cl_response = closed_loop_linear_response(linear_result.transfer_function, gains, np.linspace(0.0, 60.0, 600))
        linear_closed_loop_results[name] = {"tf": closed_loop_tf, "time": cl_time, "response": cl_response}
    plot_linear_closed_loop_comparison(linear_closed_loop_results, FIGURES / "linear_closed_loop_comparison.png")

    print(f"\nArtifacts securely saved to:\n  Figures: {FIGURES.resolve()}\n  Models:  {MODELS.resolve()}\n\nEnd-to-end workflow complete.")

if __name__ == "__main__":
    main()