"""Step 2 checks: symbolic derivation, open loop poles, energy conservation, linear vs nonlinear."""
import numpy as np
import sympy as sp
from params import load_params
from model import linear_AB, deriv, rk4, energy

p = load_params()
A, B = linear_AB(p)

# ---- Check 1: independent symbolic derivation (from body coordinates) ----
t = sp.symbols('t')
Me, m, l, I, g, r = sp.symbols('Me m l I g r', positive=True)
tau = sp.symbols('tau', real=True)
x = sp.Function('x')(t)
th = sp.Function('th')(t)
xc = x + l * sp.sin(th)
yc = l * sp.cos(th)
T = (sp.Rational(1, 2) * Me * x.diff(t)**2
     + sp.Rational(1, 2) * m * (xc.diff(t)**2 + yc.diff(t)**2)
     + sp.Rational(1, 2) * I * th.diff(t)**2)
V = m * g * yc
L = T - V

def EL(q, Q):
    return sp.diff(L.diff(q.diff(t)), t) - L.diff(q) - Q

eqs = [EL(x, tau / r), EL(th, -tau)]
sol = sp.solve(eqs, [x.diff(t, 2), th.diff(t, 2)], dict=True)[0]
X, XD, TH, THD = sp.symbols('X XD TH THD', real=True)

def sub(e):
    e = e.subs({x.diff(t): XD, th.diff(t): THD})
    return e.subs({x: X, th: TH})

f = sp.Matrix([XD, sub(sol[x.diff(t, 2)]), THD, sub(sol[th.diff(t, 2)])])
state = [X, XD, TH, THD]
zero = {X: 0, XD: 0, TH: 0, THD: 0}
vals = {Me: p.m_eff, m: p.body_mass_kg, l: p.axle_to_com_m, I: p.body_pitch_inertia_kgm2,
        g: p.gravity_ms2, r: p.wheel_radius_m}
A_sym = np.array(f.jacobian(state).subs(zero).subs(vals).evalf(), dtype=float)
B_sym = np.array(f.diff(tau).subs(zero).subs(vals).evalf(), dtype=float)
print("Check 1: max |A - A_symbolic| =", np.max(np.abs(A - A_sym)))
print("         max |B - B_symbolic| =", np.max(np.abs(B - B_sym)))

# ---- Check 2: open loop poles ----
print("Check 2: open loop poles =", np.round(np.linalg.eigvals(A), 3))

# ---- Check 3: energy conservation with zero input ----
s = np.array([0.0, 0.5, 0.3, -0.2])
E0 = energy(p, s)
dt, worst = 1e-4, 0.0
for _ in range(10000):
    s = rk4(lambda z: deriv(p, z, 0.0), s, dt)
    worst = max(worst, abs(energy(p, s) - E0))
print("Check 3: max energy drift over 1 s (J) =", worst)

# ---- Check 4: linear vs nonlinear for a small tilt ----
s_nl = np.array([0.0, 0.0, 0.002, 0.0])
s_li = s_nl.copy()
dt, worst = 1e-4, 0.0
for _ in range(3000):
    s_nl = rk4(lambda z: deriv(p, z, 0.0), s_nl, dt)
    s_li = rk4(lambda z: A @ z, s_li, dt)
    worst = max(worst, abs(s_nl[2] - s_li[2]))
print("Check 4: max tilt difference over 0.3 s (rad) =", worst,
      "relative to final tilt:", worst / abs(s_nl[2]))
