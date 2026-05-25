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

sp.pprint(M_net)

# This equation is the problem and the solution. It maps the actuator and vehicle state directly to the moment on the vehicle.
# The million dollar question: how can we get control inputs u such that this equation equals a desired moment M?
# The thing is, this is a differential equation, not a normal equation. It contains derivatives with respect to time of the actuators,
# as well as their current state. This equation cannot be solved instantaneously; we cannot make the actuators do whatever we want.
# The actuators have their own lag; their own physics. But, we know that too. So, we can also get differential equations for the actuator
# states and state derivatives as a function of U. Let's do it.

"""
Transient parameters
"""

tau_r = sp.symbols('tau_r') # rotor/motor time constant
tau_s = sp.symbols('tau_s') # servo time constant
k_r, k_s = sp.symbols('k_r k_s') # DC gains

"""
Inputs
"""

phi_px_cmd, phi_nx_cmd = sp.symbols('phi_px_cmd phi_nx_cmd')
omega_rpx_cmd, omega_rnx_cmd = sp.symbols('omega_rpx_cmd omega_rnx_cmd')

"""
Outputs
"""

phidot_px = -1 / tau_s * phi_px + k_s / tau_s * phi_px_cmd
phidot_nx = -1 / tau_s * phi_nx + k_s / tau_s * phi_nx_cmd
omegadot_rpx = -1 / tau_r * omega_rpx + k_r / tau_r * omega_rpx_cmd
omegadot_rnx = -1 / tau_r * omega_rnx + k_r / tau_r * omega_rnx_cmd

sp.pprint([phidot_px,phidot_nx,omegadot_rpx,omegadot_rnx])

# now we have the whole story. We have
# - 3 differential equations describing the moment on the body as a function of the vehicle state and its derivatives
# - 4 differential equations mapping control input u to rate change of the vehicle states

# here's an idea. we have an M_cmd. we want it to go TOWARDS M_actual.
# 1) for everything we DO NOT have control over (which is just omega_b) plug it in.
# 2) obtain the derivative of the moment equation with respect to time.
# 3) zero all terms beyond 1st order.
# 4) solve for control inputs u that minimize dM/dt towards M_desired.

# alternatively, consider that we have an omega_b_desired and an omega_b_current. we want to drive omega_b towards the desired state.
# to that end, we wish to maximzie omegadot_b towards omega_b. omegadot_b can be approximated as B * M. And M is the big equation we
# already have.

