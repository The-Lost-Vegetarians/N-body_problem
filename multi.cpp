#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <omp.h>

using namespace std;

// Body struct and other functions remain the same...
struct Body {
    double m;      // mass
    double x, y;   // position
    double vx, vy; // velocity
    double ax, ay; // acceleration
};
typedef vector<Body> State;
const double G = 6.67430e-11;
const double SOFTENING = 1e-9;

/*
    NEW FUNCTION
    Initializes a system where all bodies start from a standstill (zero velocity).
    This setup demonstrates gravitational collapse.
*/
void initializeFromRest(State& bodies, int numBodies) {
    bodies.resize(numBodies);

    // Setup for random number generation
    mt19937 rng(time(0));
    const double SYSTEM_SCALE = 1.5e12; // ~10 AU
    uniform_real_distribution<double> pos_dist(-SYSTEM_SCALE, SYSTEM_SCALE);
    uniform_real_distribution<double> mass_dist(1.0e26, 5.0e28);

    // Assign random positions and masses, but zero velocity
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].m = mass_dist(rng);
        bodies[i].x = pos_dist(rng);
        bodies[i].y = pos_dist(rng);
        
        // Initialize velocity to zero
        bodies[i].vx = 0.0;
        bodies[i].vy = 0.0;
    }
}

// --- The rest of the simulation code (calculateAccelerations, rk4Step, etc.) is unchanged ---

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

vector<double> stateToVector(const State& bodies) {
    vector<double> vec;
    vec.reserve(bodies.size() * 4);
    for (const auto& b : bodies) {
        vec.push_back(b.x); vec.push_back(b.y);
        vec.push_back(b.vx); vec.push_back(b.vy);
    }
    return vec;
}

State vectorToState(const vector<double>& vec, const State& originalBodies) {
    State newState = originalBodies;
    int n = newState.size();
    for(int i = 0; i < n; ++i) {
        newState[i].x = vec[i * 4 + 0]; newState[i].y = vec[i * 4 + 1];
        newState[i].vx = vec[i * 4 + 2]; newState[i].vy = vec[i * 4 + 3];
    }
    return newState;
}

vector<double> nBodyODE(const vector<double>& vec, const State& bodiesTemplate) {
    State bodies = vectorToState(vec, bodiesTemplate);
    calculateAccelerations(bodies);
    vector<double> dydt(vec.size());
    for (size_t i = 0; i < bodies.size(); i++) {
        dydt[i * 4 + 0] = bodies[i].vx; dydt[i * 4 + 1] = bodies[i].vy;
        dydt[i * 4 + 2] = bodies[i].ax; dydt[i * 4 + 3] = bodies[i].ay;
    }
    return dydt;
}

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
    // --- Simulation setup ---
    const int N_BODIES = 5;
    State bodies;

    // Use the new initialization function for a "from rest" system
    initializeFromRest(bodies, N_BODIES);

    // --- Simulation Loop ---
    double t = 0;
    double dt = 86400; // 1 day timestep
    int num_steps = 200;

    cout << "Starting " << N_BODIES << "-body simulation from rest (gravitational collapse)." << endl;

    for (int i = 0; i < num_steps; ++i) {
        bodies = rk4Step(bodies, dt);
        t += dt;

        if (i % 1 == 0) {
            cout << "Step " << i << ", Time: " << t / (3600*24) << " days" << endl;
            for(int j=0; j < min(N_BODIES, 5); ++j){
                cout << "   Body " << j << ": x=" << bodies[j].x << ", y=" << bodies[j].y << endl;
            }
        }
    }

    cout << "Simulation finished." << endl;
    return 0;
}