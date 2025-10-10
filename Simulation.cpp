#include "simulation.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>

// --- Global Variable Definitions ---
State bodies;
std::vector<std::vector<std::pair<float, float>>> trajectories;
const double G = 6.67430e-11;
const double SOFTENING = 1e9;
bool is_paused = false;
double time_step = 3600;

// --- Camera Control Definitions ---
float zoom = 1.0f / (4.5e11);
float pan_x = 0.0f;
float pan_y = 0.0f;

// Calculates the net acceleration on each body
void calculateAccelerations(State &current_bodies) {
    int n = current_bodies.size();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        current_bodies[i].ax = 0.0;
        current_bodies[i].ay = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = current_bodies[j].x - current_bodies[i].x;
            double dy = current_bodies[j].y - current_bodies[i].y;
            double distSq = dx * dx + dy * dy + SOFTENING;
            double invDist = 1.0 / sqrt(distSq);
            double invDist3 = invDist * invDist * invDist;
            
            current_bodies[i].ax += G * current_bodies[j].m * dx * invDist3;
            current_bodies[i].ay += G * current_bodies[j].m * dy * invDist3;
        }
    }
}

// Helper functions for the RK4 ODE solver
std::vector<double> stateToVector(const State& current_bodies) {
    std::vector<double> vec;
    for (const auto& b : current_bodies) {
        vec.push_back(b.x); vec.push_back(b.y);
        vec.push_back(b.vx); vec.push_back(b.vy);
    }
    return vec;
}

State vectorToState(const std::vector<double>& vec, const State& originalBodies) {
    State newState = originalBodies;
    for(size_t i = 0; i < newState.size(); ++i) {
        newState[i].x = vec[i * 4 + 0]; newState[i].y = vec[i * 4 + 1];
        newState[i].vx = vec[i * 4 + 2]; newState[i].vy = vec[i * 4 + 3];
    }
    return newState;
}

std::vector<double> nBodyODE(const std::vector<double>& vec, const State& bodiesTemplate) {
    State current_bodies = vectorToState(vec, bodiesTemplate);
    calculateAccelerations(current_bodies);
    
    std::vector<double> dydt(vec.size());
    for (size_t i = 0; i < current_bodies.size(); i++) {
        dydt[i * 4 + 0] = current_bodies[i].vx; dydt[i * 4 + 1] = current_bodies[i].vy;
        dydt[i * 4 + 2] = current_bodies[i].ax; dydt[i * 4 + 3] = current_bodies[i].ay;
    }
    return dydt;
}

// Runge-Kutta 4th order integration step
State rk4Step(State &current_bodies, double dt) {
    std::vector<double> y0 = stateToVector(current_bodies);
    std::vector<double> k1 = nBodyODE(y0, current_bodies);
    
    std::vector<double> temp(y0.size());
    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k1[i];
    std::vector<double> k2 = nBodyODE(temp, current_bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k2[i];
    std::vector<double> k3 = nBodyODE(temp, current_bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + dt * k3[i];
    std::vector<double> k4 = nBodyODE(temp, current_bodies);

    std::vector<double> y1(y0.size());
    for (size_t i = 0; i < y0.size(); i++) {
        y1[i] = y0[i] + dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    return vectorToState(y1, current_bodies);
}

// Velocity Verlet integration step
State verletStep(State& current_bodies, double dt) {
    State next_bodies = current_bodies;

    for (size_t i = 0; i < current_bodies.size(); ++i) {
        next_bodies[i].vx += current_bodies[i].ax * 0.5 * dt;
        next_bodies[i].vy += current_bodies[i].ay * 0.5 * dt;
        next_bodies[i].x += next_bodies[i].vx * dt;
        next_bodies[i].y += next_bodies[i].vy * dt;
    }

    calculateAccelerations(next_bodies);

    for (size_t i = 0; i < current_bodies.size(); ++i) {
        next_bodies[i].vx += next_bodies[i].ax * 0.5 * dt;
        next_bodies[i].vy += next_bodies[i].ay * 0.5 * dt;
    }
    
    return next_bodies;
}

// Initializes the 3-Body Figure-8 stable orbit
void initFigure8Simulation() {
    const int N_BODIES = 3;
    bodies.assign(N_BODIES, Body());
    trajectories.assign(N_BODIES, std::vector<std::pair<float, float>>());

    bodies[0].m = 1.5e30; bodies[1].m = 1.5e30; bodies[2].m = 1.5e30;
    
    bodies[0].x = -0.97000436 * 1.5e11; bodies[0].y = 0.24308753 * 1.5e11;
    bodies[1].x = 0.0;                  bodies[1].y = 0.0;
    bodies[2].x = 0.97000436 * 1.5e11;  bodies[2].y = -0.24308753 * 1.5e11;

    double v_x = 0.4662036850, v_y = 0.4323657300;
    double vel_scale = 30000;
    bodies[0].vx = v_x * vel_scale;     bodies[0].vy = v_y * vel_scale;
    bodies[1].vx = -2 * v_x * vel_scale; bodies[1].vy = -2 * v_y * vel_scale;
    bodies[2].vx = v_x * vel_scale;     bodies[2].vy = v_y * vel_scale;
    
    calculateAccelerations(bodies);
}

// Initializes a random N-body simulation
void initRandomSimulation(int n) {
    bodies.assign(n, Body());
    trajectories.assign(n, std::vector<std::pair<float, float>>());
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> pos_dist(-2.0e11, 2.0e11);
    std::uniform_real_distribution<double> vel_dist(-5000, 5000);
    std::uniform_real_distribution<double> mass_dist(1.0e28, 1.0e30);

    // Central massive body
    bodies[0].m = 1.989e30 * 100;
    bodies[0].x = 0; bodies[0].y = 0;
    bodies[0].vx = 0; bodies[0].vy = 0;

    for (int i = 1; i < n; ++i) {
        bodies[i].m = mass_dist(rng);
        bodies[i].x = pos_dist(rng);
        bodies[i].y = pos_dist(rng);
        bodies[i].vx = vel_dist(rng);
        bodies[i].vy = vel_dist(rng);
    }
    
    calculateAccelerations(bodies);
}