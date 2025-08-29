#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <utility> // For std::pair

// Forward declaration
struct Body;

// Define State as a vector of Body objects
typedef std::vector<Body> State;

// A structure to hold the state of a single body
struct Body {
    double m;      // mass
    double x, y;   // position
    double vx, vy; // velocity
    double ax, ay; // acceleration
};

// --- Global Variable Declarations ---
extern State bodies;
extern std::vector<std::vector<std::pair<float, float>>> trajectories;
extern bool is_paused;
extern double time_step;

// --- Camera Control Variable Declarations ---
extern float zoom;
extern float pan_x;
extern float pan_y;

// --- Function Prototypes ---
void initFigure8Simulation();
void initRandomSimulation(int n);
void calculateAccelerations(State &current_bodies);
State rk4Step(State &current_bodies, double dt);
State verletStep(State &current_bodies, double dt);

#endif // SIMULATION_H