def rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + dt * k1 / 2.0)
    k3 = f(x + dt * k2 / 2.0)
    k4 = f(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)