import numpy as np

def linear_AB(p):
    """State s = [x, xdot, theta, thetadot], input tau (total wheel torque)."""
    a, b, J, D = p.a, p.b, p.J, p.D
    m, l, g, r = p.body_mass_kg, p.axle_to_com_m, p.gravity_ms2, p.wheel_radius_m
    k = m * g * l
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -b * k / D, 0],
                  [0, 0, 0, 1],
                  [0, 0, a * k / D, 0]], dtype=float)
    B = np.array([[0],
                  [(J / r + b) / D],
                  [0],
                  [-(b / r + a) / D]], dtype=float)
    return A, B

def deriv(p, s, tau):
    """Nonlinear dynamics. Returns ds/dt."""
    x, xd, th, thd = s
    a, b, J = p.a, p.b, p.J
    m, l, g, r = p.body_mass_kg, p.axle_to_com_m, p.gravity_ms2, p.wheel_radius_m
    k = m * g * l
    M = np.array([[a, b * np.cos(th)],
                  [b * np.cos(th), J]])
    rhs = np.array([tau / r + b * np.sin(th) * thd**2,
                    -tau + k * np.sin(th)])
    xdd, thdd = np.linalg.solve(M, rhs)
    return np.array([xd, xdd, thd, thdd])

def rk4(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def energy(p, s):
    x, xd, th, thd = s
    a, b, J = p.a, p.b, p.J
    k = p.body_mass_kg * p.gravity_ms2 * p.axle_to_com_m
    T = 0.5 * a * xd**2 + b * np.cos(th) * xd * thd + 0.5 * J * thd**2
    return T + k * np.cos(th)