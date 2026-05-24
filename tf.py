from integrator import rk4
import numpy as np

class tf1:
    def __init__(self, tau, K, dt_s=None, x0=0.0):
        self.tau = tau
        self.K = K
        self.dt_s = dt_s
        self.x = x0
        self.y = 0.0
        self.u = 0.0

        self.A = -1.0 / self.tau
        self.B = self.K / self.tau

    def dstate(self, x):
        return self.A * x + self.B * self.u

    def step(self, u):
        self.u = u
        self.x = rk4(self.dstate, self.x, self.dt_s)
        self.y = self.x
        return self.y, self.dstate(self.x)