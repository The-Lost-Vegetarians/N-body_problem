N-Body Problem Simulation
1. Introduction
The N-body problem deals with predicting the motion of N interacting bodies under their mutual gravitational forces.

The two-body problem has an exact analytical solution (Kepler’s laws).

For N > 2, the system becomes nonlinear and chaotic, with no closed-form solution, requiring numerical methods.

This problem is central in astrophysics (simulating planetary systems, star clusters, and galaxies), molecular dynamics, and other areas of computational science. Its main challenges include:

A computational cost that increases rapidly as N grows.

A sensitive dependence on initial conditions, a hallmark of chaotic systems.

The necessity of numerical approximation methods to solve the equations of motion.

This project implements numerical integration techniques, coupled with parallel computing and real-time visualization, to study the N-body problem.

2. Files
two_body_simulation.html: The main interactive, browser-based simulation of the Earth-Moon system. It includes the complete physics engine and WebGL rendering in a single file.

main.cpp: The original C++ implementation of the RK4 integrator and the two-body ODEs. This is a command-line program that calculates the initial state derivatives.

2body.py: An early Python prototype that uses the RK4 integrator to pre-calculate an orbit and animate it using matplotlib.

3. Methodology
3.1 Equations of Motion
For any body i with mass m_i and position vector r_i, the gravitational force exerted on it by another body j is given by Newton's law of universal gravitation:

$$ F_{ij} = G \frac{m_i m_j}{|r_j - r_i|^3} (r_j - r_i) $$

Applying Newton’s second law, the acceleration of body i is the sum of all forces from other bodies:

$$ m_i \frac{d^2 r_i}{dt^2} = \sum_{j \neq i} F_{ij} $$

This results in a coupled system of ordinary differential equations (ODEs) that must be solved numerically for N > 2.

3.2 Runge-Kutta Integration
We use the 4th-order Runge-Kutta (RK4) method to approximate the solution to these ODEs. For each time step 
Deltat, the state vector y (containing positions and velocities) is updated as follows:

k_1=f(t,y)

k_2=f(t+
Deltat/2,y+
Deltat/2
cdotk_1)

k_3=f(t+
Deltat/2,y+
Deltat/2
cdotk_2)

k_4=f(t+
Deltat,y+
Deltat
cdotk_3)

y(t+
Deltat)=y(t)+
fracDeltat6(k_1+2k_2+2k_3+k_4)

This method offers a good balance between computational efficiency and accuracy for simulating orbital mechanics.

3.3 Visualization & Parallelization
OpenGL (C++): Renders the trajectories and positions of the bodies in real-time for intuitive analysis of the simulation results.

OpenMP: Parallelizes the force computation loop across multiple CPU cores, which significantly reduces runtime for simulations with a larger number of bodies (N).

4. How to Run
Interactive WebGL Visualization
Open two_body_simulation.html in any modern web browser (like Chrome, Firefox, or Safari).

C++ Version
Compile the C++ code:

g++ main.cpp -o 2body_cpp

Run the executable:

./2body_cpp

Python Version
Make sure you have numpy and matplotlib installed:

pip install numpy matplotlib

Run the script:

python 2body.py

5. Project Progress
✅ Completed:

Two-body simulation solver using the Runge-Kutta method.

Real-time visualization of the 2-body system with OpenGL.

CPU parallelization of the physics calculations using OpenMP.

🔜 Remaining:

Extend the simulation to handle N > 3 bodies.

Experiment with Verlet integration, which offers better long-term energy conservation.

Implement GPU acceleration (e.g., with CUDA or OpenCL) for large-scale simulations.

6. Conclusion
The N-body problem is a classic example of how simple physical laws can lead to complex, chaotic, and computationally intensive systems. This project successfully demonstrates the entire pipeline for tackling such a problem:

Formulating the physics as a system of ODEs.

Solving the system numerically with the Runge-Kutta method.

Visualizing the results with OpenGL for clear interpretation.

Improving performance with parallel computing via OpenMP.

The completed two-body simulation provides a strong foundation for scaling to a larger number of bodies and leveraging more advanced techniques for realistic astrophysical simulations.