#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include "simulation.h"

// --- Simulation Settings ---
enum Integrator { RK4, VERLET };
Integrator current_integrator = RK4;
int num_bodies = 50;
GLFWwindow* window; // Keep a global reference to the window

// --- GLFW Callback for Keyboard Input ---
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return; // Only handle key presses

    switch (key) {
        case GLFW_KEY_SPACE: is_paused = !is_paused; break;
        case GLFW_KEY_R: initRandomSimulation(num_bodies); break;
        case GLFW_KEY_F: initFigure8Simulation(); break;
        case GLFW_KEY_I:
            current_integrator = (current_integrator == RK4) ? VERLET : RK4;
            std::cout << "Switched to " << (current_integrator == RK4 ? "RK4" : "Verlet") << " integrator." << std::endl;
            break;
        case GLFW_KEY_EQUAL: zoom *= 1.1f; break; // Use '=' for '+'
        case GLFW_KEY_MINUS: zoom /= 1.1f; break;
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
    }
}

// Function to handle panning with arrow keys
void handle_panning() {
    float pan_speed = 2.0e11 / zoom;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) pan_y -= pan_speed;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) pan_y += pan_speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) pan_x += pan_speed;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) pan_x -= pan_speed;
}

// --- OpenGL Display Function ---
void display() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    
    // Set projection matrix based on aspect ratio
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)width / height;
    gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);

    // Switch back to model-view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glClear(GL_COLOR_BUFFER_BIT);
    glScalef(zoom, zoom, 1.0f);
    glTranslatef(pan_x, pan_y, 0.0f);

    glColor3f(0.5f, 0.5f, 0.5f);
    for (const auto& traj : trajectories) {
        glBegin(GL_LINE_STRIP);
        for (const auto& pos : traj) {
            glVertex2f(pos.first, pos.second);
        }
        glEnd();
    }

    for (size_t i = 0; i < bodies.size(); ++i) {
        if (i == 0) glColor3f(1.0f, 1.0f, 0.0f);
        else glColor3f(0.7f, 0.7f, 1.0f);
        
        glPushMatrix();
        glTranslatef(bodies[i].x, bodies[i].y, 0.0f);
        GLUquadric* quad = gluNewQuadric();
        gluSphere(quad, 1.5e10, 20, 20);
        gluDeleteQuadric(quad);
        glPopMatrix();
    }
}


int main(int argc, char** argv) {
    if (argc > 1) num_bodies = std::stoi(argv[1]);

    // --- Initialize GLFW ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    window = glfwCreateWindow(800, 800, "N-Body Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    initRandomSimulation(num_bodies);
    glClearColor(0.0f, 0.0f, 0.05f, 1.0f); // Set background color

    std::cout << "Starting " << num_bodies << "-body simulation with GLFW." << std::endl;
    // Print controls...

    // --- Main Loop ---
    while (!glfwWindowShouldClose(window)) {
        if (!is_paused) {
            if (current_integrator == RK4) bodies = rk4Step(bodies, time_step);
            else bodies = verletStep(bodies, time_step);

            for(size_t i = 0; i < bodies.size(); ++i) {
                trajectories[i].push_back({(float)bodies[i].x, (float)bodies[i].y});
                if(trajectories[i].size() > 500) trajectories[i].erase(trajectories[i].begin());
            }
        }

        handle_panning();
        display(); // Call display function

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}