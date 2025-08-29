import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------------------
# 1. RK4 Integrator
# -----------------------------------------
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h,   y + h * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------------------
# 2. Two-body ODEs (relative COM frame)
# State: [x, y, vx, vy]
# -----------------------------------------
def two_body(t, y, G=4*np.pi**2, M=1.0):
    x, y_pos, vx, vy = y
    r = np.sqrt(x**2 + y_pos**2)
    ax = -G * M * x / r**3
    ay = -G * M * y_pos / r**3
    return np.array([vx, vy, ax, ay])

# -----------------------------------------
# 3. Simulation parameters
# -----------------------------------------
G = 4 * np.pi**2   # AU^3 / yr^2 / Msun
M = 1.0            # Sun mass

r0 = 1.0           # AU
v0 = 2 * np.pi     # AU/yr (circular velocity)
y0 = np.array([r0, 0.0, 0.0, v0])  # Earth initial state

t0, tf = 0.0, 2.0      # simulate 2 years
h = 0.001              # step size (years)
steps = int((tf - t0)/h)

# Precompute trajectory
positions = np.zeros((steps, 2))
y = y0.copy()
t = t0
for i in range(steps):
    positions[i] = [y[0], y[1]]
    y = rk4_step(lambda t_, y_: two_body(t_, y_, G, M), t, y, h)
    t += h

