import math
# three basic black box ode solver are defined at here
# they are Euler, mid_point and RK4
def ode_solve_euler(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/steps
    t = t0
    z = z0

    for i_step in range(steps):
        z = z + h * f(z, t)
        t = t + h
    return z


def ode_solve_rk4(z0,t0,t1,f):
    """
    Simplest RK4 ODE initial value solver
    """
    t = t0
    z = z0
    h_max = 0.05
    steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    h = (t1-t0)/steps
    half_dt = h * 0.5
    for i_step in range(steps):
      k1 = f(z,t)
      k2 = f(z + half_dt * k1,t + half_dt)
      k3 = f(z + half_dt * k2,t + half_dt)
      k4 = f(z + h * k3,t+h)
      z += ((k1 + 2 * (k2 + k3) + k4) * h / 6)
      t += h
    return z


def ode_solve_mid(z0,t0,t1,f):
    """
    Simplest RK4 ODE initial value solver
    """
    t = t0
    z = z0
    h_max = 0.05
    steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    h = (t1 - t0)/steps
    half_dt = h * 0.5
    for i_step in range(steps):
        z += h * f(z + half_dt * f(z,t), t + half_dt)
        t += h
    return z