# Dynamic Modeling and Reinforcement Learning Based Optimal PI Control of an Exothermic CSTR

This project implements a graduate-level process control workflow for a jacketed exothermic CSTR following the sequence:

`PROCESS -> DYNAMIC MODEL -> OPEN LOOP ANALYSIS -> CONTROLLER DESIGN -> CLASSICAL TUNING -> RL OPTIMIZATION`

## Features

- Nonlinear material and energy balance model for an irreversible exothermic CSTR
- Open-loop disturbance analysis with solve_ivp simulation
- Jacobian linearization and state-space/transfer-function generation
- PI controller design and classical tuning via Ziegler-Nichols, Cohen-Coon, and IMC rules
- Closed-loop metrics: overshoot, rise time, settling time, IAE, ISE, ITAE
- Custom Gymnasium environment for Q-learning based PI gain optimization
- DQN training with Stable-Baselines3 and PyTorch
- Automatic plot generation and artifact saving

## Run

Install dependencies, then execute:

```bash
python main.py
```

Outputs are saved in `artifacts/`.

## Notes

- The model uses standard process control textbook parameters in consistent engineering units.
- The RL environment discretizes PI gains so Q-learning can be applied with DQN, which is the practical approach for a beginner-friendly implementation.
- The project is intentionally modular so the same model can be reused for process analysis, tuning, and learning-based optimization.
