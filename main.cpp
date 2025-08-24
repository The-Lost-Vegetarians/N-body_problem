#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

/*
    This code section defines the ODEs for a simple two-body system and computes the derivatives.
    The input state vector contains the initial positions and velocities of the two bodies.
    The output state vector contains the derivatives of the positions and velocities which provide 
    the velocities and accelerations of the bodies for numerical integration for the next time step.
    Details of the ODEs are given in the comments of the issue assigned to this task.
*/

const double G = 6.67430e-11; // Gravitational constant

typedef vector<double> State;

State twoBodyODE(const State &state, double m1, double m2) {
    State dydt(8);

    // Unpack state vector for initial positions and velocities {x1, y1, vx1, vy1, x2, y2, vx2, vy2}
    double x1 = state[0], y1 = state[1]; 
    double vx1 = state[2], vy1 = state[3]; 
    double x2 = state[4], y2 = state[5]; 
    double vx2 = state[6], vy2 = state[7];

    //distance components (x, y) and distance between the two bodies
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dist = sqrt(dx * dx + dy * dy); // Euclidean distance
    double invDist = 1.0 / (dist * dist * dist); // Inverse cube of distance
    
    //acceleration components (x, y) for both bodies (1, 2)
    double ax1 = G * m2 * dx * invDist; 
    double ay1 = G * m2 * dy * invDist; 
    double ax2 = -G * m1 * dx * invDist; 
    double ay2 = -G * m1 * dy * invDist; 

    dydt = {vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2}; // Derivatives: [dx1/dt, dy1/dt, dvx1/dt, dvy1/dt, dx2/dt, dy2/dt, dvx2/dt, dvy2/dt]
    return dydt;
}

// rk4 solver to be implemented here. provide a concise multi-line comment explaining the method and the code following it.

int main() {

    double m1 = 5.972e24; // Mass of Earth
    double m2 = 7.348e22; // Mass of Moon
    State state = {0, 0, 0, 0, 384400000, 0, 0, 1022}; // Initial state
    
    State dydt = twoBodyODE(state, m1, m2);

    // Print result
    cout << "Derivatives (d(state)/dt):\n";
    for (size_t i = 0; i < dydt.size(); i++) {
        cout << dydt[i] << " ";
    }
    cout << endl;

    return 0;
}