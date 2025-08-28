# N-Body Problem Simulation

## 1. Introduction
The **N-body problem** deals with predicting the motion of `N` interacting bodies under their mutual gravitational forces.  
- The **two-body problem** has an exact solution (Kepler’s laws).  
- For **N > 2**, the system becomes nonlinear and chaotic, with **no closed-form solution**.  

This problem is central in **astrophysics** (planetary systems, galaxies), **molecular dynamics**, and **computational simulations**.  
Its main challenges include:  
- Rapidly increasing computational cost as N grows.  
- Sensitive dependence on initial conditions.  
- Necessity of **numerical approximation methods**.  

This project implements **numerical integration techniques**, coupled with **parallel computing** and **real-time visualization**, to study the N-body problem.

---

## 2. Methodology

### 2.1 Equations of Motion
For two bodies with masses `m₁` and `m₂`, the gravitational force is:

\[
F_{ij} = G \frac{m_i m_j}{|r_j - r_i|^3} (r_j - r_i)
\]

Newton’s second law gives:

\[
m_i \frac{d^2 r_i}{dt^2} = \sum_j F_{ij}
\]

This leads to a coupled system of **ordinary differential equations (ODEs)**.  
- For 2 bodies → closed-form elliptical orbits.  
- For N > 2 → must be solved numerically.

---

### 2.2 Runge-Kutta Integration
We use the **4th-order Runge-Kutta (RK4)** method to approximate solutions. For each timestep Δt:

\[
\begin{aligned}
k_1 &= f(t, y) \\
k_2 &= f(t + \tfrac{\Delta t}{2}, y + \tfrac{\Delta t}{2} k_1) \\
k_3 &= f(t + \tfrac{\Delta t}{2}, y + \tfrac{\Delta t}{2} k_2) \\
k_4 &= f(t + \Delta t, y + \Delta t \cdot k_3) \\
y(t + \Delta t) &= y(t) + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
\]

This method balances **accuracy and efficiency** for simulating motion.

---

### 2.3 Visualization & Parallelization
- **OpenGL (C++)**: Renders trajectories of bodies in real-time for intuitive analysis.  
- **OpenMP**: Parallelizes force computations across CPU cores, significantly reducing runtime for larger `N`.

---

## 3. Project Progress
✅ Completed:  
- Two-body simulation using Runge-Kutta.  
- Real-time visualization with OpenGL.  
- CPU parallelization with OpenMP.  

🔜 Remaining:  
- Extend to **N > 3** bodies.  
- Experiment with **Verlet integration** for better long-term energy conservation.  
- **GPU acceleration** for large-scale simulations.  

---

## 4. Conclusion
The N-body problem highlights how simple gravitational laws lead to complex, chaotic systems.  
This project demonstrates:  
- Formulation of the problem as ODEs.  
- Numerical solution via Runge-Kutta.  
- Visualization with OpenGL.  
- Performance gains with OpenMP.  

The successful two-body simulation provides a strong foundation for scaling to more bodies and leveraging advanced integrators and GPUs for realistic astrophysical simulations.