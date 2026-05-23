from integrator import rk4

def dstate(x):
    return x**2

dt_s = 0.1
x = 1

new_x = rk4(dstate,x,dt_s)

bro = 1