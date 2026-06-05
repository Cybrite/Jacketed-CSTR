# Jacketed Exothermic CSTR: Complete Classical Process Control Derivation

## 1. Problem Statement

This project studies a jacketed exothermic continuous stirred-tank reactor (CSTR) for the irreversible reaction

$$A \rightarrow B$$

The objective is to regulate the reactor temperature with classical process control methods.

The process variable to be controlled is the reactor temperature, the manipulated variable is the coolant flow rate through the jacket, and the jacket temperature acts as an intermediate thermal state between the coolant stream and the reactor.

This report follows classical process-control methodology: nonlinear modeling, steady-state analysis, deviation-variable formulation, Taylor linearization, state-space form, transfer-function derivation, open-loop analysis, and PI control.

## 2. Process Diagram

The project includes a process-control diagram showing:

- the exothermic CSTR
- the cooling jacket
- the feed stream
- the product stream
- the temperature sensor
- the PI controller
- the control valve
- the coolant flow rate

The main labels are:

- Controlled variable: $T_R$
- Manipulated variable: $F_C$
- Jacket temperature: $T_J$
- Heat transfer: $Q = UA(T_R - T_J)$

The diagram is saved as [process_control_diagram.png](artifacts/figures/process_control_diagram.png).

## 3. Nomenclature and Units

| Symbol     | Meaning                                      | Units       |
| ---------- | -------------------------------------------- | ----------- |
| $V$        | reactor volume                               | L           |
| $V_j$      | jacket volume                                | L           |
| $F$        | reactor feed flow rate                       | L/min       |
| $F_C$      | coolant flow rate through jacket             | L/min       |
| $C_A$      | concentration of reactant A in reactor       | mol/L       |
| $C_{A0}$   | feed concentration of A                      | mol/L       |
| $T_R$      | reactor temperature                          | K           |
| $T_J$      | jacket temperature                           | K           |
| $T_0$      | reactor feed temperature                     | K           |
| $T_{C,in}$ | coolant inlet temperature                    | K           |
| $UA$       | overall heat-transfer coefficient times area | cal/(min·K) |
| $\Delta H$ | heat of reaction                             | cal/mol     |
| $\rho$     | reactor density                              | g/L         |
| $C_p$      | reactor heat capacity                        | cal/(g·K)   |
| $\rho_j$   | jacket density                               | g/L         |
| $C_{p,j}$  | jacket heat capacity                         | cal/(g·K)   |
| $k_0$      | Arrhenius pre-exponential factor             | 1/min       |
| $E$        | activation energy                            | cal/mol     |
| $R$        | gas constant                                 | cal/(mol·K) |

## 4. Modeling Assumptions

The derivation uses the standard idealized CSTR assumptions:

1. Perfect mixing in the reactor and jacket.
2. Constant reactor volume $V$.
3. Constant jacket volume $V_j$.
4. Constant density and heat capacity in each lumped volume.
5. Single irreversible first-order reaction.
6. Arrhenius temperature dependence of the reaction rate.
7. Heat transfer only through the jacket wall.
8. No shaft work and no phase change.

These assumptions give the standard nonlinear dynamic model used in chemical process control texts.

## 5. Reaction Rate Expression

The reaction rate is

$$r_A = k_0 e^{-E/(RT_R)} C_A$$

where

- $k_0$ is the pre-exponential factor,
- $E$ is the activation energy,
- $R$ is the gas constant,
- $T_R$ is the reactor temperature,
- $C_A$ is the concentration of reactant A in the reactor.

Because the reaction is exothermic, the reactor temperature strongly affects the rate, and the rate strongly affects the reactor temperature. This positive feedback is the source of thermal nonlinearity and possible multiplicity of steady states.

## 6. Material Balance for Reactant A

Start from the general balance:

$$\text{Accumulation} = \text{In} - \text{Out} - \text{Consumption}$$

For reactant A in a perfectly mixed CSTR:

$$V\frac{dC_A}{dt} = F C_{A0} - F C_A - V r_A$$

Rearrange the flow terms:

$$V\frac{dC_A}{dt} = F(C_{A0} - C_A) - V r_A$$

Substitute the Arrhenius rate expression:

$$V\frac{dC_A}{dt} = F(C_{A0} - C_A) - V k_0 e^{-E/(RT_R)} C_A$$

Divide through by $V$:

$$\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - k_0 e^{-E/(RT_R)} C_A$$

### Physical meaning

- $\frac{F}{V}(C_{A0} - C_A)$ is the net convective contribution from feed dilution.
- $-k_0 e^{-E/(RT_R)} C_A$ is the disappearance of A by reaction.

## 7. Reactor Energy Balance

Start from energy conservation:

$$\text{Accumulation} = \text{In} - \text{Out} + \text{Heat generated} - \text{Heat removed}$$

For the reactor control volume:

$$\rho C_p V\frac{dT_R}{dt} = F\rho C_p(T_0 - T_R) + (-\Delta H)V r_A - UA(T_R - T_J)$$

Substitute the reaction rate:

$$\rho C_p V\frac{dT_R}{dt} = F\rho C_p(T_0 - T_R) + (-\Delta H)V k_0 e^{-E/(RT_R)} C_A - UA(T_R - T_J)$$

Divide through by $\rho C_p V$:

$$\frac{dT_R}{dt} = \frac{F}{V}(T_0 - T_R) + \frac{-\Delta H}{\rho C_p}k_0 e^{-E/(RT_R)} C_A - \frac{UA}{\rho C_p V}(T_R - T_J)$$

### Physical meaning

- $\frac{F}{V}(T_0 - T_R)$ is the convective enthalpy exchange with the feed.
- $\frac{-\Delta H}{\rho C_p}k_0 e^{-E/(RT_R)} C_A$ is heat release due to the exothermic reaction.
- $-\frac{UA}{\rho C_p V}(T_R - T_J)$ is heat removed to the jacket.

## 8. Jacket Energy Balance

The jacket is treated as a separate thermal control volume. Start from energy conservation:

$$\text{Accumulation} = \text{In} - \text{Out} + \text{Heat received from reactor}$$

The jacket balance is

$$\rho_j C_{p,j} V_j\frac{dT_J}{dt} = F_C\rho_j C_{p,j}(T_{C,in} - T_J) + UA(T_R - T_J)$$

Divide through by $\rho_j C_{p,j}V_j$:

$$\frac{dT_J}{dt} = \frac{F_C}{V_j}(T_{C,in} - T_J) + \frac{UA}{\rho_j C_{p,j}V_j}(T_R - T_J)$$

### Why coolant flow appears

The coolant flow rate $F_C$ determines how quickly coolant enters and leaves the jacket. A larger coolant flow increases the rate at which the jacket temperature can be moved toward the coolant inlet temperature.

### Why jacket dynamics add a state

The jacket stores thermal energy, so its temperature cannot adjust instantaneously. That storage term creates a separate differential equation and therefore a third state variable.

## 9. State Vector and Inputs

Define the state vector as

$$x = \begin{bmatrix} C_A \\ T_R \\ T_J \end{bmatrix}$$

The manipulated input is

$$u = F_C$$

The disturbances are

$$C_{A0}, \quad T_0, \quad F, \quad T_{C,in}$$

The measured output is

$$y = T_R$$

## 10. Nonlinear State Equations

Collecting the three balances gives the nonlinear model:

$$\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - k_0 e^{-E/(RT_R)} C_A$$

$$\frac{dT_R}{dt} = \frac{F}{V}(T_0 - T_R) + \frac{-\Delta H}{\rho C_p}k_0 e^{-E/(RT_R)} C_A - \frac{UA}{\rho C_p V}(T_R - T_J)$$

$$\frac{dT_J}{dt} = \frac{F_C}{V_j}(T_{C,in} - T_J) + \frac{UA}{\rho_j C_{p,j}V_j}(T_R - T_J)$$

This is the complete nonlinear model used for steady-state analysis and control design.

## 11. Steady-State Analysis

At steady state:

$$\frac{dC_A}{dt} = 0, \qquad \frac{dT_R}{dt} = 0, \qquad \frac{dT_J}{dt} = 0$$

### 11.1 Concentration at steady state

From the material balance:

$$0 = \frac{F}{V}(C_{A0} - C_{A,s}) - k_s C_{A,s}$$

where

$$k_s = k_0 e^{-E/(RT_{R,s})}$$

Rearrange:

$$\frac{F}{V}C_{A0} = \left(\frac{F}{V} + k_s\right)C_{A,s}$$

Therefore

$$C_{A,s} = \frac{F C_{A0}}{F + Vk_s}$$

### 11.2 Reactor and jacket steady-state equations

The reactor balance gives

$$0 = \frac{F}{V}(T_0 - T_{R,s}) + \frac{-\Delta H}{\rho C_p}k_s C_{A,s} - \frac{UA}{\rho C_p V}(T_{R,s} - T_{J,s})$$

The jacket balance gives

$$0 = \frac{F_{C,s}}{V_j}(T_{C,in} - T_{J,s}) + \frac{UA}{\rho_j C_{p,j}V_j}(T_{R,s} - T_{J,s})$$

Rearrange the jacket equation to solve for the steady coolant flow rate:

$$F_{C,s}(T_{C,in} - T_{J,s}) = -\frac{UA}{\rho_j C_{p,j}}(T_{R,s} - T_{J,s})$$

Therefore

$$F_{C,s} = -\frac{UA(T_{R,s} - T_{J,s})}{\rho_j C_{p,j}(T_{C,in} - T_{J,s})}$$

### 11.3 Numerical operating point used in the project

The steady-state candidate selected by the code is:

$$C_{A,s} = 0.0317$$

$$T_{R,s} = 405.47\ \text{K}$$

$$T_{J,s} = 335.16\ \text{K}$$

$$F_{C,s} = 100.00\ \text{L/min}$$

## 12. Deviation Variables

Define deviation variables around the steady state:

$$C_A' = C_A - C_{A,s}$$

$$T_R' = T_R - T_{R,s}$$

$$T_J' = T_J - T_{J,s}$$

$$F_C' = F_C - F_{C,s}$$

### Why deviation variables are used

Deviation variables shift the equilibrium to the origin. This simplifies linearization, makes the state-space model easier to write, and allows the transfer function to represent small perturbations around the operating point.

## 13. Taylor Linearization

Let the nonlinear system be written as

$$\dot{x} = f(x,u)$$

where

$$x = \begin{bmatrix} C_A \\ T_R \\ T_J \end{bmatrix}, \qquad u = F_C$$

The first-order Taylor expansion around the steady state is

$$\Delta \dot{x} \approx \left.\frac{\partial f}{\partial x}\right|_{s}\Delta x + \left.\frac{\partial f}{\partial u}\right|_{s}\Delta u$$

This gives the linearized model

$$\dot{x}' = A x' + B u'$$

where $x' = [C_A', T_R', T_J']^T$ and $u' = F_C'$.

### 13.1 Required partial derivatives

The reaction-rate derivatives are

$$\frac{\partial r_A}{\partial C_A} = k_0 e^{-E/(RT_R)}$$

$$\frac{\partial r_A}{\partial T_R} = k_0 C_A e^{-E/(RT_R)}\frac{E}{RT_R^2}$$

Evaluate these at the steady state to obtain

$$\left.\frac{\partial r_A}{\partial C_A}\right|_s = k_s$$

$$\left.\frac{\partial r_A}{\partial T_R}\right|_s = k_s C_{A,s}\frac{E}{RT_{R,s}^2}$$

### 13.2 Linearized concentration equation

Linearizing the material balance gives

$$\frac{dC_A'}{dt} = a_{11}C_A' + a_{12}T_R'$$

with

$$a_{11} = -\frac{F}{V} - k_s$$

$$a_{12} = -k_s C_{A,s}\frac{E}{RT_{R,s}^2}$$

### 13.3 Linearized reactor energy equation

The linearized reactor temperature equation is

$$\frac{dT_R'}{dt} = a_{21}C_A' + a_{22}T_R' + a_{23}T_J'$$

where

$$a_{21} = \frac{-\Delta H}{\rho C_p}k_s$$

$$a_{22} = -\frac{F}{V} + \frac{-\Delta H}{\rho C_p}k_s C_{A,s}\frac{E}{RT_{R,s}^2} - \frac{UA}{\rho C_pV}$$

$$a_{23} = \frac{UA}{\rho C_pV}$$

### 13.4 Linearized jacket energy equation

The linearized jacket equation is

$$\frac{dT_J'}{dt} = a_{32}T_R' + a_{33}T_J' + b_3F_C'$$

with

$$a_{32} = \frac{UA}{\rho_j C_{p,j}V_j}$$

$$a_{33} = -\frac{F_{C,s}}{V_j} - \frac{UA}{\rho_j C_{p,j}V_j}$$

$$b_3 = \frac{T_{C,in} - T_{J,s}}{V_j}$$

## 14. State-Space Model

The linearized state-space model is

$$\dot{x}' = A x' + B u'$$

$$y' = Cx' + Du'$$

with

$$x' = \begin{bmatrix} C_A' \\ T_R' \\ T_J' \end{bmatrix}, \qquad u' = F_C', \qquad y' = T_R'$$

The matrices are

$$
A = \begin{bmatrix}
a_{11} & a_{12} & 0 \\
a_{21} & a_{22} & a_{23} \\
0 & a_{32} & a_{33}
\end{bmatrix}
$$

$$B = \begin{bmatrix} 0 \\ 0 \\ b_3 \end{bmatrix}$$

$$C = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}$$

$$D = \begin{bmatrix} 0 \end{bmatrix}$$

### Physical meaning of the matrices

- $A$ contains the internal dynamic couplings between concentration, reactor temperature, and jacket temperature.
- $B$ shows how coolant flow affects the jacket temperature directly.
- $C$ selects reactor temperature as the controlled variable.
- $D = 0$ because coolant flow does not instantaneously change reactor temperature in this lumped model.

### Numerical state-space matrices from the project run

$$
A = \begin{bmatrix}
-31.578003 & -0.051535 & 0 \\
6397.071797 & 7.689404 & 2.092050 \\
0 & 2.5 & -7.5
\end{bmatrix}
$$

$$B = \begin{bmatrix} 0 \\ 0 \\ -1.75791 \end{bmatrix}$$

$$C = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}$$

$$D = \begin{bmatrix} 0 \end{bmatrix}$$

## 15. Transfer Function Derivation

Starting from the linearized state equations:

$$sC_A'(s) = a_{11}C_A'(s) + a_{12}T_R'(s)$$

$$sT_R'(s) = a_{21}C_A'(s) + a_{22}T_R'(s) + a_{23}T_J'(s)$$

$$sT_J'(s) = a_{32}T_R'(s) + a_{33}T_J'(s) + b_3F_C'(s)$$

### 15.1 Solve the concentration equation

From the first equation:

$$C_A'(s) = \frac{a_{12}}{s-a_{11}}T_R'(s)$$

### 15.2 Solve the jacket equation

From the third equation:

$$T_J'(s) = \frac{a_{32}}{s-a_{33}}T_R'(s) + \frac{b_3}{s-a_{33}}F_C'(s)$$

### 15.3 Substitute into the reactor equation

Substitute $C_A'(s)$ and $T_J'(s)$ into the reactor equation:

$$sT_R'(s) = a_{21}\frac{a_{12}}{s-a_{11}}T_R'(s) + a_{22}T_R'(s) + a_{23}\left(\frac{a_{32}}{s-a_{33}}T_R'(s) + \frac{b_3}{s-a_{33}}F_C'(s)\right)$$

Collect the $T_R'(s)$ terms on the left:

$$\left[s - a_{22} - \frac{a_{21}a_{12}}{s-a_{11}} - \frac{a_{23}a_{32}}{s-a_{33}}\right]T_R'(s) = \frac{a_{23}b_3}{s-a_{33}}F_C'(s)$$

Multiply through by $(s-a_{11})(s-a_{33})$:

$$\left[(s-a_{11})(s-a_{22})(s-a_{33}) - a_{21}a_{12}(s-a_{33}) - a_{23}a_{32}(s-a_{11})\right]T_R'(s) = a_{23}b_3(s-a_{11})F_C'(s)$$

Therefore the transfer function is

$$G_p(s) = \frac{T_R'(s)}{F_C'(s)} = \frac{a_{23}b_3(s-a_{11})}{(s-a_{11})(s-a_{22})(s-a_{33}) - a_{21}a_{12}(s-a_{33}) - a_{23}a_{32}(s-a_{11})}$$

## 16. System Order

This plant is third order because there are three independent state variables:

1. reactant concentration $C_A$
2. reactor temperature $T_R$
3. jacket temperature $T_J$

### Physical interpretation

- The concentration state introduces chemical-reaction dynamics.
- The reactor temperature state introduces thermal storage in the reactor.
- The jacket temperature state introduces thermal storage in the jacket.

In some operating regions the concentration mode is much faster than the thermal modes, so the input-output behavior may appear approximately second order in practice. However, the full mechanistic model remains third order.

## 17. Open-Loop Analysis

Using the linearized transfer function $G_p(s)$, the project generates:

- poles
- zeros
- unit step response
- Bode plot
- root locus
- stability check

### Project result

The linearized open-loop transfer function obtained in the run is approximately

$$G_p(s) = \frac{3.908\times 10^{-14}s^2 - 3.678s - 116.1}{s^3 + 31.39s^2 + 260.8s + 486.3}$$

The dominant second-order estimate from the code is:

$$\zeta \approx 1.0000, \qquad \omega_n \approx 2.6242$$

The fitted FOPDT approximation is:

$$K \approx -0.2388, \qquad \tau \approx 0.4143, \qquad \theta \approx 0.1083$$

The open-loop step response is saved as [linear_step_response.png](artifacts/figures/linear_step_response.png).

## 18. PI Controller

The classical PI controller is

$$G_c(s) = K_c\left(1 + \frac{1}{\tau_I s}\right)$$

### Proportional action

The proportional term responds immediately to the current error.

### Integral action

The integral term accumulates error over time and removes steady-state offset.

### Reverse-acting control

For this plant, the process gain from coolant flow to reactor temperature is negative. Therefore the PI controller must be reverse acting so that an increase in controller output increases cooling and lowers reactor temperature.

## 19. Closed-Loop Transfer Function

With unity feedback, the closed-loop transfer function is

$$\frac{T_R(s)}{R(s)} = \frac{G_c(s)G_p(s)}{1 + G_c(s)G_p(s)}$$

This is the standard closed-loop form used in classical process control.

## 20. PI Tuning

The project evaluates two classical PI tuning rules based on the FOPDT approximation.

### 20.1 Ziegler-Nichols PI

For an FOPDT model with gain $K$, time constant $\tau$, and dead time $\theta$:

$$K_c = -\frac{0.9\tau}{|K|\theta}$$

$$\tau_I = 3.33\theta$$

The negative sign is used because the plant gain is negative and the controller must be reverse acting.

### 20.2 IMC PI

For IMC tuning:

$$K_c = -\frac{\tau}{|K|(\lambda + \theta)}$$

$$\tau_I = \tau$$

where $\lambda$ is the closed-loop filter parameter.

IMC is usually more robust for exothermic reactors because it gives a slower but safer response and preserves stability margin.

### 20.3 Numerical tuning results

From the project run:

$$K_c^{ZN} = -14.4175, \qquad \tau_I^{ZN} = 0.3607$$

$$K_c^{IMC} = -8.0097, \qquad \tau_I^{IMC} = 0.4143$$

## 21. Performance Comparison

The project compares:

- rise time
- settling time
- overshoot
- steady-state error
- IAE
- ISE
- ITAE

### Numerical results

| Method |  OS % | Rise | Settling |    IAE |    ISE |   ITAE |
| ------ | ----: | ---: | -------: | -----: | -----: | -----: |
| ZN-PI  | 42.69 | 0.00 |     4.10 | 3.5925 | 5.9083 | 3.4203 |
| IMC-PI | 29.72 | 0.00 |     1.30 | 1.6243 | 4.1197 | 0.6453 |

The IMC PI tuning is more conservative and gives the smaller cumulative error measures and shorter settling time in the reported nonlinear simulation.

### 21.1 Error-integral definitions

For clarity, the integral performance indices reported above are defined as follows (with $e(t)=r(t)-y(t)$):

- IAE (Integral of Absolute Error):

$$IAE = \int_0^{T} |e(t)|\,dt$$

- ISE (Integral of Squared Error):

$$ISE = \int_0^{T} e(t)^2\,dt$$

- ITAE (Integral of Time-weighted Absolute Error):

$$ITAE = \int_0^{T} t\,|e(t)|\,dt$$

All integrals are evaluated numerically over the simulation horizon using trapezoidal integration (see [utils/metrics.py](utils/metrics.py)). These indices penalize error in different ways: IAE weights all errors equally, ISE penalizes large deviations more heavily, and ITAE emphasizes steady-state and long-duration errors.

### 21.2 Discussion of the numeric results

The numerical table above (ZN-PI vs IMC-PI) shows that the IMC-tuned PI controller yields better overall performance on the nonlinear reactor: it has lower IAE, lower ISE, much lower ITAE, and a shorter measured settling time. This is consistent with IMC tuning being more conservative (slower initial action but less overshoot). The Ziegler–Nichols tuning produces a faster but underdamped response with larger overshoot (42.7%) and larger cumulative error integrals.

Practical interpretation:

- IMC-PI: smaller cumulative errors and faster effective settling — preferred when avoiding thermal excursions is critical.
- ZN-PI: aggressive tuning, larger overshoot — may be useful when speed is prioritized but risks instability in nonlinear regions.

When using these results for design, verify on the full nonlinear model (as done here) and consider further robustness testing (disturbance scenarios, sensor noise, and actuator limits).

## 22. Figures Generated by the Project

- [Process-control diagram](artifacts/figures/process_control_diagram.png)
- [Open-loop response](artifacts/figures/open_loop_response.png)
- [Linear step response](artifacts/figures/linear_step_response.png)
- [Pole-zero map](artifacts/figures/pole_zero_map.png)
- [Bode plot](artifacts/figures/bode_open_loop.png)
- [Root locus](artifacts/figures/root_locus.png)
- [Classical closed-loop comparison](artifacts/figures/classical_closed_loop_comparison.png)
- [Linear closed-loop comparison](artifacts/figures/linear_closed_loop_comparison.png)
- [Open vs closed loop](artifacts/figures/open_vs_closed_loop.png)
- [Performance table](artifacts/figures/classical_metrics_table.png)

## 23. Conclusion

This jacketed exothermic CSTR is a classical third-order thermal-chemical process with a reverse-acting coolant-flow control loop. The reactor concentration, reactor temperature, and jacket temperature together determine the dynamics. Linearization around the selected steady state provides a transfer function for classical PI tuning. The IMC PI controller is the better choice when the objective is to reduce overshoot and maintain robust behavior for the exothermic reactor.

The saved analysis file is intended to be read directly by a professor or copied into a notebook without needing additional derivation steps.# Jacketed Exothermic CSTR: Full Classical Process Control Derivation

## 1. Objective

This report presents a classical process control analysis for a jacketed exothermic continuous stirred-tank reactor (CSTR). The goal is to control reactor temperature with coolant flow through the jacket, following the standard steps used in chemical process control:

1. write the nonlinear material and energy balances,
2. determine the steady state,
3. define deviation variables,
4. linearize the nonlinear model,
5. derive a state-space model,
6. derive the transfer function,
7. analyze the open-loop plant,
8. design PI controllers,
9. compare closed-loop performance.

The manipulated variable is the coolant flow rate, and the controlled variable is the reactor temperature.

## 2. Process description and assumptions

The chemical reaction is:

$$A \rightarrow B$$

The reactor is exothermic, so the reaction releases heat. A cooling jacket removes heat from the reactor. The jacket temperature is a dynamic intermediate variable between the coolant stream and the reactor.

### Assumptions

The derivation uses the standard classical assumptions:

1. Perfect mixing in the reactor and in the jacket.
2. Constant reactor volume and constant jacket volume.
3. Constant density and constant heat capacity in each control volume.
4. Single irreversible first-order reaction.
5. Arrhenius temperature dependence of the reaction rate.
6. Heat transfer between reactor and jacket is proportional to the temperature difference.
7. The coolant flow rate is the manipulated variable.
8. Feed flow rate, feed concentration, feed temperature, and coolant inlet temperature are treated as disturbances unless stated otherwise.

## 3. Nomenclature

| Symbol     | Meaning                                      | Units       |
| ---------- | -------------------------------------------- | ----------- |
| $V$        | reactor volume                               | L           |
| $V_j$      | jacket volume                                | L           |
| $F$        | reactor feed flow rate                       | L/min       |
| $F_C$      | coolant flow rate through jacket             | L/min       |
| $C_A$      | reactant concentration in reactor            | mol/L       |
| $C_{A0}$   | feed concentration                           | mol/L       |
| $T_R$      | reactor temperature                          | K           |
| $T_J$      | jacket temperature                           | K           |
| $T_0$      | feed temperature                             | K           |
| $T_{C,in}$ | coolant inlet temperature                    | K           |
| $UA$       | overall heat-transfer coefficient times area | cal/(min·K) |
| $\Delta H$ | reaction enthalpy                            | cal/mol     |
| $\rho$     | reactor density                              | g/L         |
| $C_p$      | reactor heat capacity                        | cal/(g·K)   |
| $\rho_j$   | jacket density                               | g/L         |
| $C_{p,j}$  | jacket heat capacity                         | cal/(g·K)   |
| $k_0$      | pre-exponential factor                       | 1/min       |
| $E$        | activation energy                            | cal/mol     |
| $R$        | gas constant                                 | cal/(mol·K) |

### State, input, and output variables

| Symbol | Meaning |
| ------ | ------- |
| $x_1$  | $C_A$   |
| $x_2$  | $T_R$   |
| $x_3$  | $T_J$   |
| $u$    | $F_C$   |
| $y$    | $T_R$   |

For linearization, the deviation variables are:

$$C_A' = C_A - C_{A,s}$$

$$T_R' = T_R - T_{R,s}$$

$$T_J' = T_J - T_{J,s}$$

$$F_C' = F_C - F_{C,s}$$

## 4. Nonlinear reaction rate

The reaction rate is first order in reactant concentration and follows Arrhenius temperature dependence:

$$r_A = k_0 e^{-E/(R T_R)} C_A$$

The rate increases as the reactor temperature increases because the exponential factor becomes larger.

## 5. Material balance

### 5.1 General balance

For reactant $A$ in a well-mixed CSTR:

$$\text{Accumulation} = \text{In} - \text{Out} - \text{Consumption}$$

Since the reactor volume is constant:

$$V\frac{dC_A}{dt} = F C_{A0} - F C_A - V r_A$$

This can be written as:

$$V\frac{dC_A}{dt} = F(C_{A0} - C_A) - V r_A$$

### 5.2 Substitute the reaction rate

Substituting the Arrhenius form gives the full nonlinear concentration balance:

$$\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - k_0 e^{-E/(R T_R)} C_A$$

### 5.3 Physical meaning

The first term, $\frac{F}{V}(C_{A0} - C_A)$, is the net dilution term. If the feed concentration is larger than the reactor concentration, the concentration rises. If the reactor concentration is larger than the feed concentration, the concentration falls.

The second term, $k_0 e^{-E/(R T_R)} C_A$, is the rate of reactant consumption by chemical reaction.

## 6. Reactor energy balance

### 6.1 General balance

For the reactor energy balance:

$$\text{Accumulation} = \text{In} - \text{Out} + \text{Heat generated} - \text{Heat removed}$$

The standard constant-property energy balance is:

$$\rho C_p V\frac{dT_R}{dt} = F\rho C_p(T_0 - T_R) + (-\Delta H) V r_A - UA(T_R - T_J)$$

### 6.2 Substitute the reaction rate

Substituting $r_A$ gives:

$$\rho C_p V\frac{dT_R}{dt} = F\rho C_p(T_0 - T_R) + (-\Delta H) V k_0 e^{-E/(R T_R)} C_A - UA(T_R - T_J)$$

### 6.3 Physical meaning

The three terms on the right-hand side are:

1. sensible heat brought in by the feed,
2. heat released by the exothermic reaction,
3. heat transferred from reactor to jacket.

For an exothermic reaction, $\Delta H < 0$, so the factor $(-\Delta H)$ is positive.

## 7. Jacket energy balance

### 7.1 General balance

The jacket is also a thermal storage volume, so it has its own dynamic balance:

$$\text{Accumulation} = \text{In} - \text{Out} + \text{Heat received from reactor}$$

The jacket balance is:

$$\rho_j C_{p,j} V_j\frac{dT_J}{dt} = F_C \rho_j C_{p,j}(T_{C,in} - T_J) + UA(T_R - T_J)$$

### 7.2 Physical meaning

The coolant flow rate $F_C$ appears because coolant enters and exits the jacket carrying enthalpy. The term $UA(T_R - T_J)$ appears with a positive sign because heat leaving the reactor enters the jacket.

### 7.3 Why the jacket adds a state

The jacket is not an algebraic constraint. It stores thermal energy, so it has dynamics of its own. That is why the plant has three states instead of two.

## 8. Steady-state analysis

At steady state:

$$\frac{dC_A}{dt} = 0, \qquad \frac{dT_R}{dt} = 0, \qquad \frac{dT_J}{dt} = 0$$

### 8.1 Steady-state concentration

From the concentration balance:

$$0 = \frac{F}{V}(C_{A0} - C_{A,s}) - k_0 e^{-E/(R T_{R,s})} C_{A,s}$$

Define:

$$k_s = k_0 e^{-E/(R T_{R,s})}$$

Then:

$$\frac{F}{V}(C_{A0} - C_{A,s}) = k_s C_{A,s}$$

$$\frac{F}{V}C_{A0} = \left(\frac{F}{V} + k_s\right) C_{A,s}$$

$$C_{A,s} = \frac{F C_{A0}}{F + V k_s}$$

### 8.2 Steady-state reactor temperature

At steady state the reactor balance becomes:

$$0 = F\rho C_p(T_0 - T_{R,s}) + (-\Delta H)V k_s C_{A,s} - UA(T_{R,s} - T_{J,s})$$

This equation links the reactor temperature to the jacket temperature and the reaction rate.

### 8.3 Steady-state jacket temperature

The jacket steady-state equation is:

$$0 = F_{C,s}\rho_j C_{p,j}(T_{C,in} - T_{J,s}) + UA(T_{R,s} - T_{J,s})$$

Solving for coolant flow gives:

$$F_{C,s} = -\frac{UA(T_{R,s} - T_{J,s})}{\rho_j C_{p,j}(T_{C,in} - T_{J,s})}$$

### 8.4 Numerical operating point used in the project

The steady state found by the code is:

$$C_{A,s} = 0.0317$$

$$T_{R,s} = 405.47\,K$$

$$T_{J,s} = 335.16\,K$$

$$F_{C,s} = 100.00\,L/min$$

## 9. Deviation-variable form

Deviation variables measure departures from the operating point:

$$C_A' = C_A - C_{A,s}$$

$$T_R' = T_R - T_{R,s}$$

$$T_J' = T_J - T_{J,s}$$

$$F_C' = F_C - F_{C,s}$$

The reason for deviation variables is that they move the operating point to the origin. This makes linearization and control design much simpler.

## 10. Linearization by Taylor expansion

Let the nonlinear system be written as:

$$\dot{x} = f(x,u)$$

with:

$$x = \begin{bmatrix} C_A \\ T_R \\ T_J \end{bmatrix}, \qquad u = F_C$$

The first-order Taylor expansion about the steady state is:

$$\Delta \dot{x} \approx \left.\frac{\partial f}{\partial x}\right|_s \Delta x + \left.\frac{\partial f}{\partial u}\right|_s \Delta u$$

### 10.1 Define the nonlinear functions

$$f_1 = \frac{F}{V}(C_{A0} - C_A) - k_0 e^{-E/(R T_R)} C_A$$

$$f_2 = \frac{F}{V}(T_0 - T_R) + \frac{-\Delta H}{\rho C_p}k_0 e^{-E/(R T_R)} C_A - \frac{UA}{\rho C_p V}(T_R - T_J)$$

$$f_3 = \frac{F_C}{V_j}(T_{C,in} - T_J) + \frac{UA}{\rho_j C_{p,j} V_j}(T_R - T_J)$$

### 10.2 Partial derivatives of the reaction rate

The reaction rate is:

$$r_A = k_0 e^{-E/(R T_R)} C_A$$

Its derivatives are:

$$\frac{\partial r_A}{\partial C_A} = k_0 e^{-E/(R T_R)}$$

$$\frac{\partial r_A}{\partial T_R} = k_0 C_A e^{-E/(R T_R)} \frac{E}{R T_R^2}$$

### 10.3 Jacobian matrix $A$

The state matrix is:

$$A = \left.\frac{\partial f}{\partial x}\right|_s$$

Its entries are:

$$a_{11} = -\frac{F}{V} - k_0 e^{-E/(R T_{R,s})}$$

$$a_{12} = -k_0 C_{A,s} e^{-E/(R T_{R,s})} \frac{E}{R T_{R,s}^2}$$

$$a_{13} = 0$$

$$a_{21} = \frac{-\Delta H}{\rho C_p} k_0 e^{-E/(R T_{R,s})}$$

$$a_{22} = -\frac{F}{V} + \frac{-\Delta H}{\rho C_p} k_0 C_{A,s} e^{-E/(R T_{R,s})} \frac{E}{R T_{R,s}^2} - \frac{UA}{\rho C_p V}$$

$$a_{23} = \frac{UA}{\rho C_p V}$$

$$a_{31} = 0$$

$$a_{32} = \frac{UA}{\rho_j C_{p,j} V_j}$$

$$a_{33} = -\frac{F_{C,s}}{V_j} - \frac{UA}{\rho_j C_{p,j} V_j}$$

Therefore:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

### 10.4 Input matrix $B$

Since the manipulated input is coolant flow, only the jacket equation contains $F_C$ directly. The derivative with respect to $F_C$ is:

$$\frac{\partial f_3}{\partial F_C} = \frac{T_{C,in} - T_J}{V_j}$$

At steady state:

$$b_3 = \frac{T_{C,in} - T_{J,s}}{V_j}$$

Thus:

$$B = \begin{bmatrix} 0 \\ 0 \\ b_3 \end{bmatrix}$$

### 10.5 Output matrices $C$ and $D$

The controlled output is reactor temperature deviation:

$$y = T_R'$$

Therefore:

$$C = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}, \qquad D = \begin{bmatrix} 0 \end{bmatrix}$$

## 11. State-space model

The linearized model is:

$$\dot{x}' = A x' + B u'$$

$$y' = C x' + D u'$$

where:

$$x' = \begin{bmatrix} C_A' \\ T_R' \\ T_J' \end{bmatrix}, \qquad u' = F_C', \qquad y' = T_R'$$

At the operating point used in the project, the numerical matrices are:

$$
A = \begin{bmatrix}
-31.578003 & -0.051535 & 0 \\
6397.071797 & 7.689404 & 2.092050 \\
0 & 2.5 & -7.5
\end{bmatrix}
$$

$$B = \begin{bmatrix} 0 \\ 0 \\ -1.75791 \end{bmatrix}$$

$$C = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}$$

$$D = \begin{bmatrix} 0 \end{bmatrix}$$

## 12. Transfer-function derivation

### 12.1 Laplace transform

Assume zero initial conditions for the deviation variables. The Laplace transform of the state equations is:

$$sX(s) = AX(s) + BU(s)$$

Rearranging gives:

$$\left(sI - A\right)X(s) = BU(s)$$

Therefore:

$$X(s) = \left(sI - A\right)^{-1}BU(s)$$

The output is:

$$Y(s) = CX(s) + DU(s)$$

Substituting the state solution:

$$Y(s) = C\left(sI - A\right)^{-1}BU(s) + DU(s)$$

So the plant transfer function is:

$$G_p(s) = \frac{Y(s)}{U(s)} = C\left(sI - A\right)^{-1}B + D$$

### 12.2 Explicit algebra with the three states

Write the Laplace-domain equations directly:

$$\left(s-a_{11}\right)X_1 - a_{12}X_2 = 0$$

$$-a_{21}X_1 + \left(s-a_{22}\right)X_2 - a_{23}X_3 = 0$$

$$-a_{32}X_2 + \left(s-a_{33}\right)X_3 = b_3 U$$

From the first equation:

$$X_1 = \frac{a_{12}}{s-a_{11}}X_2$$

From the third equation:

$$X_3 = \frac{b_3 U + a_{32}X_2}{s-a_{33}}$$

Substitute both expressions into the second equation:

$$-a_{21}\left(\frac{a_{12}}{s-a_{11}}X_2\right) + (s-a_{22})X_2 - a_{23}\left(\frac{b_3 U + a_{32}X_2}{s-a_{33}}\right) = 0$$

Multiply by $(s-a_{11})(s-a_{33})$:

$$-a_{21}a_{12}(s-a_{33})X_2 + (s-a_{22})(s-a_{11})(s-a_{33})X_2 - a_{23}a_{32}(s-a_{11})X_2 = a_{23}b_3(s-a_{11})U$$

Thus:

$$\frac{X_2(s)}{U(s)} = \frac{a_{23}b_3(s-a_{11})}{(s-a_{22})(s-a_{11})(s-a_{33}) - a_{21}a_{12}(s-a_{33}) - a_{23}a_{32}(s-a_{11})}$$

Since $y = T_R' = X_2$, the plant transfer function is:

$$G_p(s) = \frac{T_R'(s)}{F_C'(s)} = \frac{a_{23}b_3(s-a_{11})}{(s-a_{22})(s-a_{11})(s-a_{33}) - a_{21}a_{12}(s-a_{33}) - a_{23}a_{32}(s-a_{11})}$$

### 12.3 Numerical transfer function obtained by the code

The computed linearized plant at the selected operating point is approximately:

$$G_p(s) = \frac{3.908\times 10^{-14}s^2 - 3.678s - 116.1}{s^3 + 31.39s^2 + 260.8s + 486.3}$$

The tiny $s^2$ coefficient in the numerator is numerical roundoff. The physically meaningful behavior is dominated by the first-order numerator terms.

## 13. System order

The system is third order because it contains three dynamic states:

1. reactant concentration $C_A$,
2. reactor temperature $T_R$,
3. jacket temperature $T_J$.

The physical interpretation is that concentration dynamics affect the rate of heat generation, the reactor stores thermal energy, and the jacket stores thermal energy separately. This creates three poles in the linearized plant.

In many operating regions the concentration mode is fast, so the dominant thermal response may look approximately second order.

## 14. Open-loop analysis

The open-loop plant is the transfer function from coolant-flow deviation to reactor-temperature deviation.

### 14.1 Poles and zeros

At the operating point used in the project, the open-loop transfer function is stable. The computed response has left-half-plane poles and a stable step response.

### 14.2 Step response

The open-loop step response is used to estimate process gain, time constant, and dead time for FOPDT approximation.

### 14.3 FOPDT estimate

The estimated FOPDT model is:

$$G_{FOPDT}(s) = \frac{K e^{-\theta s}}{\tau s + 1}$$

with numerical estimates:

$$K = -0.2388$$

$$\tau = 0.4143$$

$$\theta = 0.1083$$

The negative gain means that increasing coolant flow lowers reactor temperature, which is physically correct for a cooling loop.

## 15. PI controller

The classical PI controller is:

$$G_c(s) = K_c\left(1 + \frac{1}{\tau_I s}\right)$$

In time domain:

$$u(t) = K_c\left[e(t) + \frac{1}{\tau_I}\int_0^t e(\tau)\,d\tau\right]$$

where:

$$e(t) = T_{R,sp}(t) - T_R(t)$$

### 15.1 Physical meaning

Proportional action reacts immediately to error. Integral action removes steady-state offset.

### 15.2 Reverse-acting sign

Because the plant gain from coolant flow to temperature is negative, the controller gain must also be negative for the loop to act in the correct direction.

## 16. PI tuning

### 16.1 Ziegler–Nichols PI

For an FOPDT model, the Ziegler–Nichols open-loop PI rules are:

$$K_c = -0.9\frac{\tau}{|K|(\theta)}$$

$$\tau_I = 3.33\theta$$

Using the fitted model:

$$K_c = -14.4175$$

$$\tau_I = 0.3607$$

### 16.2 IMC PI

The IMC PI rules are:

$$K_c = -\frac{\tau}{|K|(\lambda + \theta)}$$

$$\tau_I = \tau$$

Using the fitted model:

$$K_c = -8.0097$$

$$\tau_I = 0.4143$$

The IMC controller is less aggressive and therefore more robust.

## 17. Closed-loop system

For unity feedback:

$$\frac{T_R(s)}{R(s)} = \frac{G_c(s)G_p(s)}{1 + G_c(s)G_p(s)}$$

This is the standard closed-loop transfer function used in classical process control.

## 18. Performance comparison

The project evaluates the following metrics:

1. rise time,
2. settling time,
3. overshoot,
4. steady-state error,
5. IAE,
6. ISE,
7. ITAE.

### 18.1 Closed-loop results from the code

| Method | Overshoot % | Rise time | Settling time |    IAE |    ISE |   ITAE |
| ------ | ----------: | --------: | ------------: | -----: | -----: | -----: |
| ZN-PI  |       42.69 |      0.00 |          4.10 | 3.5925 | 5.9083 | 3.4203 |
| IMC-PI |       29.72 |      0.00 |          1.30 | 1.6243 | 4.1197 | 0.6453 |

### 18.2 Interpretation

The ZN controller is more aggressive and gives larger overshoot. The IMC controller is more conservative and gives better damping and lower integrated error.

## 19. Figures generated by the project

The following figures are created automatically by the code:

1. process-control diagram,
2. open-loop response,
3. linear step response,
4. pole-zero map,
5. Bode plot,
6. root locus,
7. nonlinear closed-loop comparison,
8. linear closed-loop comparison,
9. open-loop vs closed-loop comparison,
10. controller performance table.

## 20. Conclusion

The jacketed exothermic CSTR is a third-order nonlinear process with reactor concentration, reactor temperature, and jacket temperature as dynamic states. The coolant flow rate is an appropriate manipulated variable because it directly changes the jacket energy balance and indirectly changes reactor temperature.

The derivation shows the full classical workflow used in chemical process control: nonlinear modeling, steady-state analysis, deviation variables, linearization, transfer-function derivation, open-loop analysis, PI controller design, and closed-loop performance comparison.

## 24. Closed-Loop Disturbance Rejection

The project now also includes closed-loop disturbance-rejection plots for the PI-controlled reactor. Two load disturbances are evaluated:

- a feed concentration step
- a feed temperature step

For each disturbance, the response is simulated for both ZN-PI and IMC-PI and the results are saved as:

- [Closed-loop disturbance rejection: feed concentration](artifacts/figures/closed_loop_disturbance_rejection_ca0.png)
- [Closed-loop disturbance rejection: feed temperature](artifacts/figures/closed_loop_disturbance_rejection_t0.png)

The new plots show that the IMC tuning remains the more conservative and better-damped controller when the reactor is subjected to load disturbances, while ZN-PI reacts more aggressively and tends to produce larger temperature excursions.

## 25. Updated Results Summary

The complete figure set now includes the setpoint-tracking plots and the new disturbance-rejection plots. In the latest project run, the IMC-PI controller continued to provide the best tradeoff between speed and robustness, especially when the reactor experienced feed-side disturbances. The ZN-PI controller remained usable, but it was more aggressive and less well damped.

Latest setpoint-tracking metrics from the current run:

| Method |  OS % | Settling [min] |    IAE |    ISE |   ITAE |
| ------ | ----: | -------------: | -----: | -----: | -----: |
| ZN-PI  | 42.69 |           4.10 | 3.5925 | 5.9083 | 3.4203 |
| IMC-PI | 29.72 |           1.30 | 1.6243 | 4.1197 | 0.6453 |

For disturbance rejection, the new figures show the same qualitative ordering: IMC-PI returns the reactor temperature to the setpoint faster and with smaller excursion than ZN-PI for both the feed concentration step and the feed temperature step.

Relevant generated figures now include:

- [Open-loop response](artifacts/figures/open_loop_response.png)
- [Open vs closed loop](artifacts/figures/open_vs_closed_loop.png)
- [Closed-loop disturbance rejection: feed concentration](artifacts/figures/closed_loop_disturbance_rejection_ca0.png)
- [Closed-loop disturbance rejection: feed temperature](artifacts/figures/closed_loop_disturbance_rejection_t0.png)
- [Classical closed-loop comparison](artifacts/figures/classical_closed_loop_comparison.png)

The IMC PI controller is the better choice here because it gives lower overshoot and lower integral error than the Ziegler–Nichols controller.
