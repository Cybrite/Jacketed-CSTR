# Dynamic Modeling and Reinforcement Learning Based Optimal PI Control of an Exothermic CSTR

This project implements a graduate-level process control study of a jacketed exothermic Continuous Stirred Tank Reactor (CSTR). It follows a standard workflow used in advanced process control:

`PROCESS -> DYNAMIC MODEL -> OPEN LOOP ANALYSIS -> CONTROLLER DESIGN -> CLASSICAL TUNING -> RL OPTIMIZATION`

The objective is to compare classical PI tuning methods with a reinforcement-learning-based PI tuning strategy for temperature control of an unsafe and highly nonlinear reactor.

## What The Project Does

- Builds a nonlinear dynamic model of a jacketed exothermic CSTR
- Simulates open-loop behavior under feed disturbances
- Linearizes the plant around a steady-state operating point
- Computes state-space and transfer-function representations
- Analyzes poles, zeros, eigenvalues, Bode plots, root locus, and step response
- Designs and compares classical PI controllers
- Trains a DDPG agent to optimize PI gains
- Generates plots, tables, and saved model artifacts automatically

## Process Model

The reactor models the irreversible reaction:

$$
A \rightarrow B
$$

with the standard assumptions of:

- perfect mixing
- constant volume
- constant density
- constant heat capacity
- irreversible first-order exothermic kinetics

The nonlinear balances used in the project are:

$$
\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - r_A
$$

$$
\frac{dT}{dt} = \frac{F}{V}(T_0 - T) + \frac{-\Delta H}{\rho C_p} r_A - \frac{UA}{\rho C_p V}(T - T_c)
$$

$$
r_A = k_0 e^{-E/(RT)} C_A
$$

## Control Workflow

The implementation follows the classical process-control sequence:

1. Simulate the nonlinear open-loop reactor
2. Introduce disturbances in feed temperature and concentration
3. Linearize the process around a steady-state operating point
4. Identify the open-loop dynamics and transfer function
5. Design a PI controller
6. Tune PI parameters using classical rules
7. Compare closed-loop responses and performance indices
8. Use Q-learning with DQN to search for improved PI gains
9. Compare RL-optimized and classical controllers

## Classical PI Tuning Methods

The project includes three standard PI tuning approaches:

- Ziegler-Nichols PI
- Cohen-Coon PI
- IMC PI

For each method, the code computes:

- controller gain $K_c$
- integral time $\tau_I$
- overshoot
- rise time
- settling time
- IAE, ISE, and ITAE

## Reinforcement Learning Approach

The RL portion uses a continuous action formulation that directly searches over PI gains.

- State: error, integral error, derivative of error
- Action: continuous PI gains $(K_c, \tau_I)$
- Algorithm: DDPG from Stable-Baselines3
- Reward: penalizes tracking error, oscillation, overshoot, and control effort

This is a practical way to tune PI controllers when the goal is to optimize continuous controller parameters rather than select from a discrete gain table.

## Project Structure

- `models/` - reactor model and physical parameters
- `controllers/` - PI controller and classical tuning rules
- `simulation/` - open-loop and closed-loop analysis tools
- `rl/` - Gymnasium environment and DDPG training code
- `plots/` - plotting and visualization utilities
- `utils/` - metrics and FOPDT approximation helpers
- `artifacts/` - generated figures and saved models

## Requirements

Main dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `control`
- `gymnasium`
- `stable-baselines3`
- `torch`

Install them with:

```bash
pip install -r requirements.txt
```

## Run the Project

Execute the main script from the project root:

```bash
python main.py
```

The script will:

- simulate the nonlinear open-loop reactor
- analyze the linearized model
- compare classical PI tuning methods
- train a DDPG agent for PI gain optimization
- generate all comparison plots and performance tables
- save the trained RL model in `artifacts/models/`

## Outputs

The project automatically saves:

- open-loop response plots
- linearized response plots
- pole-zero maps
- Bode plots
- root locus plots
- closed-loop comparison plots
- RL reward convergence plots
- PI gain evolution plots
- performance summary tables
- RL model checkpoints

## Notes

- The reactor parameters are chosen to be representative of a realistic exothermic CSTR example used in advanced process control.
- PI control is used instead of PID because thermal chemical processes are often noisy, slow, and well served by PI action.
- The RL agent searches over continuous PI gains, which is well suited to DDPG.
- The project is modular so the same nonlinear plant model can support future controller design or learning experiments.

## Purpose

The project is intended to resemble a graduate-level process control and intelligent systems study based on classical reactor control theory combined with reinforcement learning.
