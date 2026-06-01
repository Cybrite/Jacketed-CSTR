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
    plot_closed_loop_comparison,
    plot_open_vs_closed_loop,
    plot_linear_closed_loop_comparison,
    plot_linear_step,
    plot_metrics_table,
    plot_open_loop,
    plot_pole_zero_map,
    plot_root_locus,
    plot_process_control_diagram,
)
from simulation.analysis import (
    closed_loop_linear_response,
    classical_closed_loop_analysis,
    dominant_second_order_characteristics,
    linear_analysis,
    simulate_open_loop_disturbance,
)
from simulation.derivation import build_derivation_report
from utils.fopdt import estimate_fopdt_from_step
from utils.metrics import format_metric_rows


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
    print("\nSteady-state candidates:")
    for candidate in candidates:
        dominant = float(np.max(np.real(candidate.eigenvalues)))
        stability = "stable" if dominant < 0.0 else "unstable"
        print(f"  Ca={candidate.Ca:.4f}, TR={candidate.TR:.2f} K, TJ={candidate.TJ:.2f} K, dominant eigenvalue={dominant:.4f} ({stability})")
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

    print("\nProcess description:")
    print("  Controlled variable y = T_R (reactor temperature)")
    print("  Manipulated variable u = F_C (coolant flow rate)")
    print("  Intermediate variable = T_J (jacket temperature)")
    print("  Disturbances = C_A0, T0, F, T_C,in")
    print(f"\nSelected operating point: Ca={operating_point[0]:.4f} mol/L, TR={operating_point[1]:.2f} K, TJ={operating_point[2]:.2f} K, Fss={Fss:.2f}, Fcss={Fcss:.2f}")
    print(f"Control target setpoint: Tsp={setpoint:.2f} K")

    report = build_derivation_report(params, operating_point, Fss, Fcss)
    print("\nGoverning equations and derivation:")
    print(report.balances)
    print(report.linearization)
    print(report.transfer_function)

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

    print("\nState-space matrices:")
    print("A =\n", np.array2string(linear_result.A, precision=6, suppress_small=True))
    print("B =\n", np.array2string(linear_result.B, precision=6, suppress_small=True))
    print("C =\n", np.array2string(linear_result.C, precision=6, suppress_small=True))
    print("D =\n", np.array2string(linear_result.D, precision=6, suppress_small=True))
    print("\nTransfer function G(s) = T_R'(s) / F_C'(s):")
    print(linear_result.transfer_function)

    zeta_open, wn_open = dominant_second_order_characteristics(linear_result.poles)
    print(f"Open-loop dominant second-order estimate: zeta={zeta_open:.4f}, wn={wn_open:.4f}")

    print("Estimating FOPDT approximation from the linearized step response...")
    fopdt = estimate_fopdt_from_step(linear_result.step_time, linear_result.step_response, du=1.0)
    print(f"FOPDT estimate: K={fopdt.K:.4f}, tau={fopdt.tau:.4f}, theta={fopdt.theta:.4f}")

    imc_reference = imc_pi(fopdt)
    tuning_map = {
        "ZN-PI": ziegler_nichols_pi(fopdt),
        "IMC-PI": imc_reference,
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

    imc_closed_loop_time = classical_results["IMC-PI"]["time"]
    imc_closed_loop_temperature = classical_results["IMC-PI"]["temperature"]
    plot_open_vs_closed_loop(
        t_ol,
        TR_ol,
        imc_closed_loop_time,
        imc_closed_loop_temperature,
        setpoint,
        FIGURES / "open_vs_closed_loop.png",
    )

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

    print("\nArtifacts saved to:")
    print(f"  Figures: {FIGURES.resolve()}")
    print(f"  Models:  {MODELS.resolve()}")
    print("\nEnd-to-end workflow complete.")


if __name__ == "__main__":
    main()