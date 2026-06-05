# Dynamic Modeling and PI Control of an Exothermic CSTR

This project studies a jacketed exothermic Continuous Stirred Tank Reactor (CSTR) with the reaction $A \rightarrow B$.

The goal is simple:

- control reactor temperature $T_R$
- use coolant flow rate $F_C$ as the manipulated variable
- study the nonlinear process
- linearize the model around a steady state
- design and compare classical PI controllers

The code is written in plain Python and is organized so that each part of the control problem lives in a separate file.

## What The Project Is About

This reactor is hard to control because it is nonlinear and exothermic. That means the reaction releases heat, and the heat release changes with temperature. If the cooling is not enough, the reactor temperature can rise quickly.

The project follows the standard classical process-control workflow:

1. build the nonlinear model from material and energy balances
2. find a steady-state operating point
3. linearize the nonlinear model around that operating point
4. derive the transfer function from coolant flow to reactor temperature
5. test the open-loop response
6. design PI controllers
7. compare closed-loop tracking and disturbance rejection

## Main Files In The Project

| File                                                 | What it does                                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| [main.py](main.py)                                   | Runs the full project from start to finish                                     |
| [models/cstr.py](models/cstr.py)                     | Contains the nonlinear CSTR model, steady-state solver, and linearization code |
| [models/parameters.py](models/parameters.py)         | Stores the physical and kinetic parameters                                     |
| [controllers/pi.py](controllers/pi.py)               | Contains the PI controller and classical PI tuning rules                       |
| [simulation/analysis.py](simulation/analysis.py)     | Contains the simulation and analysis functions explained below                 |
| [simulation/derivation.py](simulation/derivation.py) | Builds the symbolic derivation report with SymPy                               |
| [plots/visualization.py](plots/visualization.py)     | Makes the plots and diagrams                                                   |
| [utils/metrics.py](utils/metrics.py)                 | Computes performance measures like IAE, ISE, and ITAE                          |
| [utils/fopdt.py](utils/fopdt.py)                     | Estimates a FOPDT model from a step response                                   |

## Process Model

The reactor uses these assumptions:

- perfect mixing
- constant volume
- constant density
- constant heat capacity
- one irreversible first-order reaction
- heat transfer through the jacket
- dynamic behavior in time

The nonlinear model is based on the following balances:

$$
\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - r_A
$$

$$
\frac{dT_R}{dt} = \frac{F}{V}(T_0 - T_R) + \frac{-\Delta H}{\rho C_p}r_A - \frac{UA}{\rho C_p V}(T_R - T_J)
$$

$$
\frac{dT_J}{dt} = \frac{F_C}{V_j}(T_{C,in} - T_J) + \frac{UA}{\rho_j C_{p,j}V_j}(T_R - T_J)
$$

$$
r_A = k_0 e^{-E/(RT_R)} C_A
$$

## How `main.py` Works

The [main.py](main.py) file is the entry point.

It does these steps:

1. loads the model parameters
2. creates the reactor model
3. finds a stable steady-state operating point
4. prints the symbolic derivation report
5. simulates nonlinear open-loop behavior
6. computes the linearized model and transfer function
7. estimates a FOPDT model from the step response
8. computes PI gains using Ziegler-Nichols and IMC rules
9. simulates closed-loop tracking
10. simulates closed-loop disturbance rejection
11. saves all figures into `artifacts/figures/`

## Functions In `simulation/analysis.py`

This file contains the main simulation and analysis routines. Each function is explained below in simple English.

| Function                                     | File                                             | What it does                                                                                                           |
| -------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `simulate_open_loop_disturbance`             | [simulation/analysis.py](simulation/analysis.py) | Runs the nonlinear reactor with open-loop disturbances in coolant flow and feed concentration                          |
| `linear_analysis`                            | [simulation/analysis.py](simulation/analysis.py) | Linearizes the model, builds the state-space model, and returns the transfer function, poles, zeros, and step response |
| `build_pi_transfer_function`                 | [simulation/analysis.py](simulation/analysis.py) | Builds the PI controller transfer function $G_c(s)$                                                                    |
| `closed_loop_linear_response`                | [simulation/analysis.py](simulation/analysis.py) | Combines the plant and PI controller in unity feedback and simulates the linear closed-loop step response              |
| `simulate_closed_loop_step`                  | [simulation/analysis.py](simulation/analysis.py) | Runs the nonlinear reactor with PI control for setpoint tracking                                                       |
| `simulate_closed_loop_disturbance_rejection` | [simulation/analysis.py](simulation/analysis.py) | Runs the nonlinear reactor with PI control while a disturbance is applied                                              |
| `closed_loop_metrics`                        | [simulation/analysis.py](simulation/analysis.py) | Computes overshoot, rise time, settling time, IAE, ISE, and ITAE                                                       |
| `classical_closed_loop_analysis`             | [simulation/analysis.py](simulation/analysis.py) | Runs all PI tuning rules and collects the results in one place                                                         |
| `dominant_second_order_characteristics`      | [simulation/analysis.py](simulation/analysis.py) | Estimates damping ratio and natural frequency from the dominant poles                                                  |
| `_rk4_step`                                  | [simulation/analysis.py](simulation/analysis.py) | Takes one fixed RK4 integration step for the nonlinear reactor                                                         |

### 1. `simulate_open_loop_disturbance`

This function runs the nonlinear reactor without a controller.

What it changes over time:

- coolant flow disturbance
- feed concentration disturbance

How it works:

- it creates a time grid
- it defines disturbance profiles as small step changes
- it calls the nonlinear plant model in [models/cstr.py](models/cstr.py)
- it returns time, reactor temperature, jacket temperature, concentration, coolant flow, and the disturbance history

Why it matters:

- it shows how the reactor behaves when no controller is protecting it
- it helps you see the natural nonlinearity of the process

### 2. `linear_analysis`

This function turns the nonlinear model into a linear model around the chosen steady state.

How it works:

- it calls the linearization code in [models/cstr.py](models/cstr.py)
- it builds the state-space model $\dot{x} = Ax + Bu$, $y = Cx + Du$
- it converts that model into a transfer function
- it finds the poles and zeros
- it computes a unit-step response

Why it matters:

- the transfer function is needed for classical PI design
- poles and zeros help you judge stability and speed

### 3. `build_pi_transfer_function`

This function creates the PI controller transfer function:

$$
G_c(s) = K_c\left(1 + \frac{1}{\tau_I s}\right)
$$

How it works:

- it uses the controller gain $K_c$
- it uses the integral time $\tau_I$
- it returns a standard transfer function object from the `control` library

Why it matters:

- this is the mathematical PI controller used in the closed-loop analysis

### 4. `closed_loop_linear_response`

This function connects the PI controller and the linear plant.

How it works:

- it multiplies the controller transfer function by the plant transfer function
- it closes the loop with unity feedback
- it simulates the step response of the closed-loop linear system

Why it matters:

- it shows how the controller changes the plant response in the linear domain

### 5. `simulate_closed_loop_step`

This function runs the full nonlinear reactor with PI control.

How it works:

- it creates a discrete PI controller from [controllers/pi.py](controllers/pi.py)
- at each time step it computes a new coolant flow command
- it advances the nonlinear plant with one RK4 step
- it stores the temperature and control flow history

Why it matters:

- it checks whether the controller really works on the nonlinear model, not just on the linear approximation

### 6. `simulate_closed_loop_disturbance_rejection`

This function tests how well the PI controller rejects load disturbances.

How it works:

- it runs the nonlinear closed-loop simulation
- at a chosen time, it applies one disturbance such as feed concentration, feed temperature, or coolant inlet temperature
- it continues the simulation and records the reactor temperature and controller action

Why it matters:

- it shows how the controller recovers after a disturbance
- this is one of the most important tests in process control

### 7. `closed_loop_metrics`

This function computes standard performance measures.

How it works:

- it compares the temperature response with the setpoint
- it calculates overshoot, rise time, settling time, IAE, ISE, and ITAE

Why it matters:

- it gives an objective way to compare controllers

### 8. `classical_closed_loop_analysis`

This function runs several PI tuning methods and stores the results.

How it works:

- it loops through each tuning rule
- it simulates the closed-loop nonlinear system
- it computes performance metrics for each method
- it returns a dictionary with time histories, gains, and metrics

Why it matters:

- it makes the comparison between Ziegler-Nichols and IMC easy to print and plot

### 9. `dominant_second_order_characteristics`

This function gives a simple second-order summary of the poles.

How it works:

- it looks for the dominant complex poles
- it estimates damping ratio $\zeta$
- it estimates natural frequency $\omega_n$

Why it matters:

- it helps explain whether the response is likely to be underdamped, well damped, or slow

### 10. `_rk4_step`

This is a small numerical integration helper.

How it works:

- it uses the classical fourth-order Runge-Kutta method
- it advances the nonlinear model by one time step

Why it matters:

- it gives a stable and accurate time-domain simulation for the nonlinear reactor

## How The Other Files Work

### `models/cstr.py`

This file contains the physical reactor model.

It provides:

- the nonlinear dynamic equations
- the steady-state solver
- the linearization method
- the transfer-function helper

### `controllers/pi.py`

This file contains the PI controller.

It provides:

- the discrete PI controller class
- Ziegler-Nichols tuning
- IMC tuning

### `simulation/derivation.py`

This file builds the symbolic derivation report.

It uses SymPy to print:

- the material balance
- the energy balance
- the steady-state equations
- the Taylor linearization
- the transfer function form

### `plots/visualization.py`

This file makes the figures.

It creates:

- open-loop response plots
- closed-loop tracking plots
- closed-loop disturbance rejection plots
- pole-zero maps
- Bode plots
- root locus plots
- process-control diagrams
- performance tables

### `utils/metrics.py`

This file computes performance metrics.

It gives:

- overshoot
- rise time
- settling time
- IAE
- ISE
- ITAE

### `utils/fopdt.py`

This file fits a first-order-plus-dead-time model from a step response.

That fitted model is then used for classical PI tuning.

## How The Full Workflow Runs

The main script [main.py](main.py) connects all of the pieces.

In plain English, it does this:

1. get the reactor parameters from [models/parameters.py](models/parameters.py)
2. solve for a steady-state operating point with [models/cstr.py](models/cstr.py)
3. print the derivation report from [simulation/derivation.py](simulation/derivation.py)
4. simulate the open-loop reactor with [simulation/analysis.py](simulation/analysis.py)
5. linearize the plant and extract the transfer function
6. tune PI controllers with [controllers/pi.py](controllers/pi.py)
7. simulate closed-loop tracking and disturbance rejection
8. create the plots with [plots/visualization.py](plots/visualization.py)
9. print the performance table using [utils/metrics.py](utils/metrics.py)

## Current Outputs

The project saves figures in `artifacts/figures/`.

Important outputs include:

- open-loop response
- open-loop vs closed-loop response
- closed-loop disturbance rejection for feed concentration step
- closed-loop disturbance rejection for feed temperature step
- linear step response
- pole-zero map
- Bode plot
- root locus
- classical closed-loop comparison
- performance table
- process-control diagram

## Requirements

The project uses:

- `numpy`
- `scipy`
- `matplotlib`
- `control`
- `sympy`

Install them with:

```bash
pip install -r requirements.txt
```

## Run The Project

From the project root:

```bash
python main.py
```

This will:

- print the balances and derivation steps
- print the transfer function and state-space matrices
- generate the plots
- compare Ziegler-Nichols PI and IMC PI
- save the results automatically

## Final Note

This project is written to be easy to follow. The code separates the reactor model, controller design, simulation, derivation, and plotting into different files so each part can be understood on its own.
