import numpy as np
from scipy.linalg import solve_continuous_are
from model import linear_AB

def design_lqr(p, Q, R):
    """LQR on the frictionless model from Step 2. State [z, v, theta, thetadot].
    z is the integral of velocity error (driving mode) or the position x (hold mode).
    The matrices A and B are the same as Step 2, so the same function serves both modes.
    Returns K (1x4) with control law u = -K @ state, where u is total wheel torque."""
    A, B = linear_AB(p)
    Q = np.diag(Q)
    R = np.array([[R]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, A - B @ K

def controllability_rank(p):
    A, B = linear_AB(p)
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
    return np.linalg.matrix_rank(C)