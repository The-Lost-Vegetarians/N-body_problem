#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h> // Include OpenMP for parallelization
#include <GL/glut.h> // Include the GLUT library for OpenGL windowing

using namespace std;

// A structure to hold the state of a single body
struct Body {
    double m;      // mass
    double x, y;   // position
    double vx, vy; // velocity
    double ax, ay; // acceleration
};

typedef vector<Body> State;

// --- Global Variables ---
State bodies; // Global state for all bodies
vector<vector<pair<float, float>>> trajectories; // To store paths
const double G = 6.67430e-11; // Gravitational constant
const double SOFTENING = 1e9; // Softening factor to prevent extreme forces at close range

// Simulation control
bool is_paused = false;
double time_step = 3600; // 1 hour timestep

// Camera control
float zoom = 1.0f / (4.5e11); // Start zoomed out to see the whole system
float pan_x = 0.0f;
float pan_y = 0.0f;


// --- Physics Engine (Identical to previous version) ---

/*
    This function calculates the net acceleration on each body in the system.
    It is parallelized with OpenMP to speed up computation.
*/
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

// Converts the State of structs into a flat vector for the ODE solver
vector<double> stateToVector(const State& current_bodies) {
    vector<double> vec;
    for (const auto& b : current_bodies) {
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
    State current_bodies = vectorToState(vec, bodiesTemplate);
    calculateAccelerations(current_bodies);
    
    vector<double> dydt(vec.size());
    for (size_t i = 0; i < current_bodies.size(); i++) {
        dydt[i * 4 + 0] = current_bodies[i].vx;
        dydt[i * 4 + 1] = current_bodies[i].vy;
        dydt[i * 4 + 2] = current_bodies[i].ax;
        dydt[i * 4 + 3] = current_bodies[i].ay;
    }
    return dydt;
}

/*
    RK4 (Runge-Kutta 4th order) method adapted for the N-body state.
*/
State rk4Step(State &current_bodies, double dt) {
    vector<double> y0 = stateToVector(current_bodies);
    vector<double> k1 = nBodyODE(y0, current_bodies);
    
    vector<double> temp(y0.size());
    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k1[i];
    vector<double> k2 = nBodyODE(temp, current_bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + 0.5 * dt * k2[i];
    vector<double> k3 = nBodyODE(temp, current_bodies);

    for (size_t i = 0; i < y0.size(); i++) temp[i] = y0[i] + dt * k3[i];
    vector<double> k4 = nBodyODE(temp, current_bodies);

    vector<double> y1(y0.size());
    for (size_t i = 0; i < y0.size(); i++) {
        y1[i] = y0[i] + dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    return vectorToState(y1, current_bodies);
}


// --- OpenGL Visualization ---

void initGL() {
    glClearColor(0.0f, 0.0f, 0.05f, 1.0f); // Dark blue background
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Apply camera transformations
    glScalef(zoom, zoom, 1.0f);
    glTranslatef(pan_x, pan_y, 0.0f);

    // Draw trajectories
    glColor3f(0.5f, 0.5f, 0.5f); // Grey color for trails
    for (const auto& traj : trajectories) {
        glBegin(GL_LINE_STRIP);
        for (const auto& pos : traj) {
            glVertex2f(pos.first, pos.second);
        }
        glEnd();
    }

    // Draw bodies
    for (size_t i = 0; i < bodies.size(); ++i) {
        // Assign a color based on body index
        glColor3f( (i==0) ? 1.0 : 0.4, (i==1) ? 1.0 : 0.4, (i==2) ? 1.0 : 0.4);
        
        glPushMatrix();
        glTranslatef(bodies[i].x, bodies[i].y, 0.0f);
        glutSolidSphere(1.5e10, 20, 20); // Render bodies as spheres
        glPopMatrix();
    }

    glutSwapBuffers();
}

void reshape(GLsizei width, GLsizei height) {
    if (height == 0) height = 1;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (width >= height) {
        double aspect = (double)width / height;
        gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);
    } else {
        double aspect = (double)height / width;
        gluOrtho2D(-1.0, 1.0, -1.0 * aspect, 1.0 * aspect);
    }
}

void update(int value) {
    if (!is_paused) {
        bodies = rk4Step(bodies, time_step);
        // Store trajectory points
        for(size_t i = 0; i < bodies.size(); ++i) {
            trajectories[i].push_back({(float)bodies[i].x, (float)bodies[i].y});
            if(trajectories[i].size() > 500) { // Limit trail length
                trajectories[i].erase(trajectories[i].begin());
            }
        }
    }
    glutPostRedisplay();
    glutTimerFunc(16, update, 0); // ~60 FPS
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case ' ': is_paused = !is_paused; break; // Space to pause/play
        case 'r': /* Reset logic would go here */ break;
        case '+': zoom *= 1.1f; break;
        case '-': zoom /= 1.1f; break;
        case 27: exit(0); break; // ESC to exit
    }
}

void specialKeys(int key, int x, int y) {
    float pan_speed = 2.0e11 / zoom;
    switch (key) {
        case GLUT_KEY_UP: pan_y -= pan_speed; break;
        case GLUT_KEY_DOWN: pan_y += pan_speed; break;
        case GLUT_KEY_LEFT: pan_x += pan_speed; break;
        case GLUT_KEY_RIGHT: pan_x -= pan_speed; break;
    }
}

void initSimulation() {
    const int N_BODIES = 3;
    bodies.assign(N_BODIES, Body());
    trajectories.assign(N_BODIES, vector<pair<float, float>>());

    // --- Simulation setup for a 3-Body Figure-8 stable orbit ---
    bodies[0].m = 1.5e30; bodies[1].m = 1.5e30; bodies[2].m = 1.5e30;
    
    bodies[0].x = -0.97000436; bodies[0].y = 0.24308753;
    bodies[1].x = 0.0;         bodies[1].y = 0.0;
    bodies[2].x = 0.97000436;  bodies[2].y = -0.24308753;

    for(auto& b : bodies) { b.x *= 1.5e11; b.y *= 1.5e11; }

    double v_x = 0.4662036850, v_y = 0.4323657300;
    double vel_scale = 30000;
    bodies[0].vx = v_x * vel_scale; bodies[0].vy = v_y * vel_scale;
    bodies[1].vx = -2 * v_x * vel_scale; bodies[1].vy = -2 * v_y * vel_scale;
    bodies[2].vx = v_x * vel_scale; bodies[2].vy = v_y * vel_scale;
}


int main(int argc, char** argv) {
    // --- Initialize Simulation State ---
    initSimulation();
    
    // --- Initialize GLUT and Create Window ---
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(800, 800);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("N-Body Simulation with OpenGL & OpenMP");

    // --- Register Callbacks ---
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutTimerFunc(16, update, 0);

    // --- OpenGL Initialization ---
    initGL();
    
    // --- Enter the Event-Processing Loop ---
    cout << "Starting N-body simulation with OpenGL." << endl;
    cout << "Controls:" << endl;
    cout << "  Spacebar: Pause / Play" << endl;
    cout << "  +/-     : Zoom In / Out" << endl;
    cout << "  Arrows  : Pan View" << endl;
    cout << "  ESC     : Exit" << endl;

    glutMainLoop();
    
    return 0;
}

