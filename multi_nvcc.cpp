#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper macro for checking CUDA API errors
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

const double G = 6.67430e-11;
const double SOFTENING = 1e-9;

// The CUDA kernel for calculating accelerations on the GPU
__global__ void calculateAccelerationsKernel(double *m, double *x, double *y, double *ax, double *ay, int n) {
    // Calculate the unique global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the bounds of our body count
    if (tid < n) {
        double my_x = x[tid];
        double my_y = y[tid];
        double total_ax = 0.0;
        double total_ay = 0.0;

        // Loop through all other bodies to calculate gravitational force
        for (int j = 0; j < n; j++) {
            if (tid == j) continue;

            double dx = x[j] - my_x;
            double dy = y[j] - my_y;
            double distSq = dx * dx + dy * dy + SOFTENING;
            double invDist = rsqrt(distSq); // Use fast GPU intrinsic for 1/sqrt
            double invDist3 = invDist * invDist * invDist;

            total_ax += G * m[j] * dx * invDist3;
            total_ay += G * m[j] * dy * invDist3;
        }

        // Write the final calculated acceleration back to global memory
        ax[tid] = total_ax;
        ay[tid] = total_ay;
    }
}

// Host function to orchestrate the GPU calculation
void calculateAccelerationsGPU(double *d_m, double *d_x, double *d_y, double *d_ax, double *d_ay, int n) {
    // Define the number of threads per block
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel on the GPU!
    calculateAccelerationsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_x, d_y, d_ax, d_ay, n);
    
    // Check for any errors during kernel launch or execution
    CUDA_CHECK(cudaGetLastError());
    // Wait for the GPU to finish its work before proceeding
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Initializer remains on the CPU
void initializeFromRest(std::vector<double>& m, std::vector<double>& x, std::vector<double>& y, std::vector<double>& vx, std::vector<double>& vy, int n) {
    m.resize(n); x.resize(n); y.resize(n); vx.resize(n); vy.resize(n);
    
    std::mt19937 rng(time(0));
    const double SYSTEM_SCALE = 1.5e12;
    std::uniform_real_distribution<double> pos_dist(-SYSTEM_SCALE, SYSTEM_SCALE);
    std::uniform_real_distribution<double> mass_dist(1.0e26, 5.0e28);

    for (int i = 0; i < n; ++i) {
        m[i] = mass_dist(rng);
        x[i] = pos_dist(rng);
        y[i] = pos_dist(rng);
        vx[i] = 0.0;
        vy[i] = 0.0;
    }
}


int main() {
    // --- Simulation Setup ---
    const int N_BODIES = 4096; // Good number to see GPU advantage
    const double dt = 86400;   // 1 day timestep
    const int num_steps = 1000;

    // Host-side vectors (Structure of Arrays)
    std::vector<double> h_m, h_x, h_y, h_vx, h_vy, h_ax(N_BODIES), h_ay(N_BODIES);
    
    initializeFromRest(h_m, h_x, h_y, h_vx, h_vy, N_BODIES);

    // Device-side pointers
    double *d_m, *d_x, *d_y, *d_vx, *d_vy, *d_ax, *d_ay;
    size_t dataSize = N_BODIES * sizeof(double);

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&d_m, dataSize));
    CUDA_CHECK(cudaMalloc(&d_x, dataSize));
    CUDA_CHECK(cudaMalloc(&d_y, dataSize));
    CUDA_CHECK(cudaMalloc(&d_vx, dataSize));
    CUDA_CHECK(cudaMalloc(&d_vy, dataSize));
    CUDA_CHECK(cudaMalloc(&d_ax, dataSize));
    CUDA_CHECK(cudaMalloc(&d_ay, dataSize));

    // Copy initial data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), dataSize, cudaMemcpyHostToDevice));

    std::cout << "Starting " << N_BODIES << "-body simulation with CUDA." << std::endl;

    // --- Simulation Loop ---
    for (int i = 0; i < num_steps; ++i) {
        // Simple Euler integration for clarity (RK4 is more complex with data transfers)
        // 1. Calculate accelerations based on current positions
        calculateAccelerationsGPU(d_m, d_x, d_y, d_ax, d_ay, N_BODIES);

        // 2. Copy accelerations back to host to update velocities and positions
        // Note: A more advanced implementation would do this update in another kernel
        // to avoid the Host <-> Device transfer inside the loop.
        CUDA_CHECK(cudaMemcpy(h_ax.data(), d_ax, dataSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ay.data(), d_ay, dataSize, cudaMemcpyDeviceToHost));
        
        // 3. Update velocities and positions on the CPU
        for(int j = 0; j < N_BODIES; ++j) {
            h_vx[j] += h_ax[j] * dt;
            h_vy[j] += h_ay[j] * dt;
            h_x[j] += h_vx[j] * dt;
            h_y[j] += h_vy[j] * dt;
        }

        // 4. Copy the updated positions and velocities back to the GPU for the next step
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), dataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), dataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), dataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), dataSize, cudaMemcpyHostToDevice));

        // Print status
        if (i % 100 == 0) {
            std::cout << "Step " << i << std::endl;
        }
    }

    // Free GPU memory
    cudaFree(d_m); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_ax); cudaFree(d_ay);

    std::cout << "Simulation finished." << std::endl;
    return 0;
}