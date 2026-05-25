import numpy as np
import dill

# test loading of the lamdified A and B matrices
with open('bicopter_sdre_matrices.pkl', 'rb') as f:
    plant = dill.load(f)

A_func = plant['A_func']
B_func = plant['B_func']

# Pass the 7 states and your 12 vehicle parameters in the exact order of the signature
# (omega_bx, omega_by, omega_bz, phi_px, phi_nx, omega_rpx, omega_rnx, Ixx, Iyy ...)
A_live = A_func(0.1, 0.0, -0.05, 0.0, 0.0, 500.0, 500.0, 0.05, 0.05, 0.08, 0.02, 0.02, 1.0, 1.0, 1e-4, 0.1, 0.0, -0.1, 0.0)

print(A_live) # Boom. Numpy array ready for math.