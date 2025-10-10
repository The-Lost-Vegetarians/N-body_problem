# N-Body Problem Simulation with High-Performance Computing

## 1\. Introduction

The N-body problem is a classic challenge in computational physics that involves predicting the individual motions of a group of celestial objects interacting gravitationally. While the two-body problem has an exact analytical solution (as described by Kepler’s laws), systems with three or more bodies exhibit chaotic behavior and lack a closed-form solution, necessitating the use of numerical methods.

This problem is fundamental to astrophysics, where it's used to simulate planetary systems, star clusters, and galaxies. The primary challenges are:

  * **Computational Complexity**: The calculation cost grows quadratically with the number of bodies (N), making simulations computationally expensive.
  * **Sensitivity**: The system is highly sensitive to initial conditions, a key characteristic of chaotic systems.
  * **Numerical Approximation**: Solving the equations of motion requires robust numerical integration techniques.

This project implements and compares different high-performance computing (HPC) techniques to solve the N-body problem, from simple multi-core CPU parallelization to massively parallel GPU acceleration.

## 2\. Files

  * `main.cpp`: An OpenMP-based simulation of a stable 3-body "figure-8" orbit.
  * `multi.cpp`: A more general OpenMP simulation that initializes N bodies at rest to model gravitational collapse.
  * `multi_nvcc.cpp`: A CUDA-based implementation that offloads the most intensive calculations to the GPU for a significant performance increase with a large number of bodies.
  * `2body.py`: A Python prototype that uses Matplotlib to animate a pre-calculated 2-body orbit.
  * `main.exe`: a Windows executable file.

## 3\. Methodology

### 3.1 Equations of Motion

For any body *i* with mass *mᵢ* and position vector *rᵢ*, the gravitational force exerted on it by another body *j* is given by Newton's law of universal gravitation:

$$F_{ij} = G \frac{m_i m_j}{|r_j - r_i|^3} (r_j - r_i)$$

From Newton's second law, the acceleration of body *i* is the sum of all forces from other bodies:

$$m_i \frac{d^2 r_i}{dt^2} = \sum_{j \neq i} F_{ij}$$

This creates a coupled system of ordinary differential equations (ODEs) that must be solved numerically for N \> 2.

### 3.2 Runge-Kutta Integration

The 4th-order Runge-Kutta (RK4) method is used to integrate the ODEs and approximate the state of the system over time. For each time step, the state vector *y* (containing positions and velocities) is updated using a weighted average of four intermediate "slopes," providing a good balance between accuracy and computational cost for orbital mechanics.

### 3.3 High-Performance Computing Approaches

The force calculation loop is the computational bottleneck. To accelerate it, two parallel computing paradigms are implemented:

#### a. CPU Parallelism with OpenMP

In `main.cpp` and `multi.cpp`, the `#pragma omp parallel for` directive is used to distribute the main loop of the `calculateAccelerations` function across multiple CPU cores. This shared-memory approach significantly speeds up the simulation by allowing each core to compute the accelerations for a subset of the bodies simultaneously.

#### b. GPU Parallelism with CUDA

The `multi_nvcc.cpp` file implements a more advanced approach using NVIDIA's CUDA platform.

  * **Host and Device**: The CPU (host) manages the simulation, while the GPU (device) is responsible for the heavy calculations.
  * **Kernel Function**: A CUDA kernel, `calculateAccelerationsKernel`, is launched on the GPU, where thousands of threads are spawned. Each thread is assigned to a single body and computes its acceleration in parallel.
  * **Data Management**: Body data (mass, position, velocity) is explicitly transferred between host RAM and GPU VRAM using `cudaMalloc` and `cudaMemcpy`. This method achieves massive parallelism, making it ideal for simulations with thousands of bodies.

## 4\. How to Run

### C++ (OpenMP Version)

1.  **Compile:**
    ```bash
    g++ -fopenmp multi.cpp -o nbody_omp
    ```
2.  **Run:**
    ```bash
    ./nbody_omp
    ```

### C++ (CUDA Version)

1.  **Compile:** You will need the NVIDIA CUDA Toolkit.
    ```bash
    nvcc multi_nvcc.cpp -o nbody_cuda
    ```
2.  **Run:**
    ```bash
    ./nbody_cuda
    ```

### Python Version

1.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    ```
2.  **Run the script:**
    ```bash
    python 2body.py
    ```

## 5\. Project Status

  * [x] N-body simulation solver using the Runge-Kutta method.
  * [x] CPU parallelization of the physics calculations using OpenMP.
  * [x] GPU acceleration for a large number of bodies using CUDA.
  * [x] Python prototype for 2-body visualization.
  * [ ] **Remaining:**
      * Experiment with Verlet integration for better long-term energy conservation.
      * Implement real-time visualization for the C++ simulations (e.g., with OpenGL or a plotting library).

## 6\. Conclusion

This project successfully demonstrates a complete pipeline for tackling the N-body problem, from formulating the physics to applying advanced HPC techniques for performance optimization. By implementing both multi-core CPU (OpenMP) and massively parallel GPU (CUDA) solutions, this work provides a practical comparison of different approaches to solving computationally intensive scientific problems. The foundation is now laid for scaling these simulations to an even larger number of bodies for more realistic astrophysical modeling.
