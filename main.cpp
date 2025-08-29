#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h> // Include OpenMP for parallelization

using namespace std;

// A structure to hold the state of a single body
struct Body {
    double m;      // mass
    double x, y;   // position
    double vx, vy; // velocity
    double ax, ay; // acceleration
};

typedef vector<Body> State;

const double G = 6.67430e-11; // Gravitational constant
const double SOFTENING = 1e-9; // Softening factor to prevent division by zero

/*
    This function calculates the net acceleration on each body in the system.
    It iterates through every pair of bodies (i, j) and calculates the gravitational
    force between them, summing the forces on each body.
    The outer loop is parallelized with OpenMP to speed up computation.
*/
void calculateAccelerations(State &bodies) {
    int n = bodies.size();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        bodies[i].ax = 0.0;
        bodies[i].ay = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double distSq = dx * dx + dy * dy + SOFTENING;
            double invDist = 1.0 / sqrt(distSq);
            double invDist3 = invDist * invDist * invDist;
            
            bodies[i].ax += G * bodies[j].m * dx * invDist3;
            bodies[i].ay += G * bodies[j].m * dy * invDist3;
        }
    }
}


// Converts the State of structs into a flat vector for the ODE solver
vector<double> stateToVector(const State& bodies) {
    vector<double> vec;
    for (const auto& b : bodies) {
        vec.push_back(b.x);
        vec.push_back(b.y);
        vec.push_back(b.vx);
        vec.push_back(b.vy);
    }
    return vec;
}

// Converts a flat vector back into a State of structs
State vectorToState(const vector<double>& vec, const State& originalBodies) {
    State newState = originalBodies;
    int n = newState.size();
    for(int i = 0; i < n; ++i) {
        newState[i].x = vec[i * 4 + 0];
        newState[i].y = vec[i * 4 + 1];
        newState[i].vx = vec[i * 4 + 2];
        newState[i].vy = vec[i * 4 + 3];
    }
    return newState;
}

// The ODE function for the N-body problem.
vector<double> nBodyODE(const vector<double>& vec, const State& bodiesTemplate) {
    State bodies = vectorToState(vec, bodiesTemplate);
    calculateAccelerations(bodies);
    
    vector<double> dydt(vec.size());
    for (size_t i = 0; i < bodies.size(); i++) {
        dydt[i * 4 + 0] = bodies[i].vx;
        dydt[i * 4 + 1] = bodies[i].vy;
        dydt[i * 4 + 2] = bodies[i].ax;
        dydt[i * 4 + 3] = bodies[i].ay;
    }
    return dydt;
}


/*
    RK4 (Runge-Kutta 4th order) method adapted for the N-body state.
    - It estimates the next state using four "slopes" (k1, k2, k3, k4).
    - The weighted average of these slopes provides an accurate update.
*/
State rk4Step(State &bodies, double dt) {
    vector<double> y0 = stateToVector(bodies);
    vector<double> k1 = nBodyODE(y0, bodies);
    
    vector<double> temp(y0.size());
    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k1[i];
    vector<double> k2 = nBodyODE(temp, bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k2[i];
    vector<double> k3 = nBodyODE(temp, bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + dt * k3[i];
    vector<double> k4 = nBodyODE(temp, bodies);

    vector<double> y1(y0.size());
    for (size_t i = 0; i < y0.size(); i++) {
        y1[i] = y0[i] + dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    return vectorToState(y1, bodies);
}


int main() {
    // --- Simulation setup for a 3-Body Figure-8 stable orbit ---
    const int N_BODIES = 3;
    State bodies(N_BODIES);

    // Set masses (in kg)
    bodies[0].m = 1.5e30;
    bodies[1].m = 1.5e30;
    bodies[2].m = 1.5e30;
    
    // Set initial positions (in meters)
    bodies[0].x = -0.97000436; bodies[0].y = 0.24308753;
    bodies[1].x = 0.0;         bodies[1].y = 0.0;
    bodies[2].x = 0.97000436;  bodies[2].y = -0.24308753;

    // Scale positions to a larger system size
    for(auto& b : bodies) {
        b.x *= 1.5e11; // 1 AU
        b.y *= 1.5e11;
    }

    // Set initial velocities (in m/s)
    double v_x = 0.4662036850;
    double v_y = 0.4323657300;
    double vel_scale = 30000; // ~ Earth orbital speed

    bodies[0].vx = v_x * vel_scale; bodies[0].vy = v_y * vel_scale;
    bodies[1].vx = -2 * v_x * vel_scale; bodies[1].vy = -2 * v_y * vel_scale;
    bodies[2].vx = v_x * vel_scale; bodies[2].vy = v_y * vel_scale;

    // --- Simulation Loop ---
    double t = 0;
    double dt = 3600; // 1 hour timestep
    int num_steps = 10000;

    cout << "Starting " << N_BODIES << "-body simulation with OpenMP." << endl;

    for (int i = 0; i < num_steps; ++i) {
        bodies = rk4Step(bodies, dt);
        t += dt;

        // Print positions every 100 steps
        if (i % 100 == 0) {
            cout << "Step " << i << ", Time: " << t / (3600*24) << " days" << endl;
            for(int j=0; j<N_BODIES; ++j){
                cout << "  Body " << j << ": x=" << bodies[j].x << ", y=" << bodies[j].y << endl;
            }
        }
    }

    cout << "Simulation finished." << endl;
    return 0;
}

