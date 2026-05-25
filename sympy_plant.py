import sympy as sp

"""
Scalar variables
"""

# tilt mechanism angles/rates
phi_px, phi_nx = sp.symbols('phi_px phi_nx')
phidot_px, phidot_nx = sp.symbols('phidot_px phidot_nx')

# motor rates/accelerations
omega_rpx, omega_rnx = sp.symbols('omega_rpx omega_rnx')
omegadot_rpx, omegadot_rnx = sp.symbols('omegadot_rpx omegadot_rnx')

# motor forces
f_px, f_nx = sp.symbols('f_px f_nx')

# body rate
omega_bx, omega_by, omega_bz = sp.symbols('omega_bx omega_by omega_bz')

# thruster positions in body-coordinates
r_pxx, r_pxz = sp.symbols('r_pxx r_pxz')
r_nxx, r_nxz = sp.symbols('r_nxx r_nxz')

# rotor inertia
j_rotor = sp.symbols('j_rotor')

"""
Vectors
"""

# tilt mechanism pointing direction & direction rate change
t_px = sp.Matrix([[0], [sp.sin(phi_px)], [-sp.cos(phi_px)]])
t_nx = sp.Matrix([[0], [sp.sin(phi_nx)], [-sp.cos(phi_nx)]])
tdot_px = sp.Matrix([[0], [sp.sin(phidot_px)], [-sp.cos(phidot_px)]])
tdot_nx = sp.Matrix([[0], [sp.sin(phidot_nx)], [-sp.cos(phidot_nx)]])

# body rate
omega_b = sp.Matrix([[omega_bx], [omega_by], [omega_bz]])

# rotor positions
r_px = sp.Matrix([[r_pxx], [0], [r_pxz]])
r_nx = sp.Matrix([[r_nxx], [0], [r_nxz]])

"""
Individual Moments
"""

M_thrust = r_px.cross(t_px * f_px) + r_nx.cross(t_nx * f_px)

M_wheel_px = -j_rotor * (tdot_px * omega_rpx + t_px * omegadot_rpx + omega_rpx * t_px.cross(omega_b))
M_wheel_nx = -j_rotor * (tdot_nx * omega_rnx + t_nx * omegadot_rnx + omega_rnx * t_nx.cross(omega_b))

"""
Net moment
"""

M_net = M_thrust + M_wheel_px + M_wheel_nx