import sympy as sp
import dill
from sympy.utilities.lambdify import lambdify

print("--- Initializing Bicopter SDRE SymPy Derivation ---")

"""
1. Scalar Variables & Symbols
"""
# Tilt mechanism angles
phi_px, phi_nx = sp.symbols('phi_px phi_nx')
# Pure symbols for derivatives (Do NOT overwrite these!)
phidot_px, phidot_nx = sp.symbols('phidot_px phidot_nx')

# Motor rates
omega_rpx, omega_rnx = sp.symbols('omega_rpx omega_rnx')
# Pure symbols for accelerations
omegadot_rpx, omegadot_rnx = sp.symbols('omegadot_rpx omegadot_rnx')

# Motor thrusts (To be provided by LUT during evaluation)
f_px, f_nx = sp.symbols('f_px f_nx')

# Body rates and accelerations
omega_bx, omega_by, omega_bz = sp.symbols('omega_bx omega_by omega_bz')
omegadot_bx, omegadot_by, omegadot_bz = sp.symbols('omegadot_bx omegadot_by omegadot_bz')

# Thruster positions in body-coordinates
r_pxx, r_pxz = sp.symbols('r_pxx r_pxz')
r_nxx, r_nxz = sp.symbols('r_nxx r_nxz')

# Inertias and transient parameters
j_rotor = sp.symbols('j_rotor')
Ixx, Iyy, Izz = sp.symbols('Ixx Iyy Izz')
tau_r, tau_s = sp.symbols('tau_r tau_s') # Time constants
k_r, k_s = sp.symbols('k_r k_s')         # DC gains

"""
2. Inputs (Commands)
"""
phi_px_cmd, phi_nx_cmd = sp.symbols('phi_px_cmd phi_nx_cmd')
omega_rpx_cmd, omega_rnx_cmd = sp.symbols('omega_rpx_cmd omega_rnx_cmd')

"""
3. Kinematic Vectors
"""
# Tilt mechanism pointing direction
t_px = sp.Matrix([[0], [sp.sin(phi_px)], [-sp.cos(phi_px)]])
t_nx = sp.Matrix([[0], [sp.sin(phi_nx)], [-sp.cos(phi_nx)]])

# Direction rate change (Using correct Chain Rule for gyroscopic coupling)
tdot_px = sp.Matrix([[0], [phidot_px * sp.cos(phi_px)], [phidot_px * sp.sin(phi_px)]])
tdot_nx = sp.Matrix([[0], [phidot_nx * sp.cos(phi_nx)], [phidot_nx * sp.sin(phi_nx)]])

# Body rate vector and rotor positions
omega_b = sp.Matrix([[omega_bx], [omega_by], [omega_bz]])
r_px = sp.Matrix([[r_pxx], [0], [r_pxz]])
r_nx = sp.Matrix([[r_nxx], [0], [r_nxz]])

"""
4. Moments & Equations of Motion
"""
M_thrust = r_px.cross(t_px * f_px) + r_nx.cross(t_nx * f_px)

M_wheel_px = -j_rotor * (tdot_px * omega_rpx + t_px * omegadot_rpx + omega_rpx * t_px.cross(omega_b))
M_wheel_nx = -j_rotor * (tdot_nx * omega_rnx + t_nx * omegadot_rnx + omega_rnx * t_nx.cross(omega_b))

M_net = M_thrust + M_wheel_px + M_wheel_nx

I_b = sp.Matrix([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]])
omegadot_b = sp.Matrix([[omegadot_bx],[omegadot_by],[omegadot_bz]])

# Nonlinear Euler coupling: omega x (I * omega)
euler_coupling = omega_b.cross(I_b * omega_b)

# EOM: xdot = f(x,u)
eom = sp.Eq(omegadot_b, sp.Inverse(I_b) * (M_net - euler_coupling))

"""
5. Actuator Dynamics & Substitution Dictionary
"""
# Maps pure symbols to functional expressions to safely inject U into the EOMs
actuator_subs = {
    phidot_px: -1 / tau_s * phi_px + k_s / tau_s * phi_px_cmd,
    phidot_nx: -1 / tau_s * phi_nx + k_s / tau_s * phi_nx_cmd,
    omegadot_rpx: -1 / tau_r * omega_rpx + k_r / tau_r * omega_rpx_cmd,
    omegadot_rnx: -1 / tau_r * omega_rnx + k_r / tau_r * omega_rnx_cmd
}

"""
6. B(x) Matrix Extraction
"""
print("-> Extracting B(x) Matrix...")
omegadot_b_rhs = sp.simplify(eom.rhs.subs(actuator_subs))

X = sp.Matrix([omega_bx, omega_by, omega_bz, phi_px, phi_nx, omega_rpx, omega_rnx])
U = sp.Matrix([phi_px_cmd, phi_nx_cmd, omega_rpx_cmd, omega_rnx_cmd])

Xdot_full = sp.Matrix([
    omegadot_b_rhs[0],
    omegadot_b_rhs[1],
    omegadot_b_rhs[2],
    actuator_subs[phidot_px],   
    actuator_subs[phidot_nx],
    actuator_subs[omegadot_rpx],
    actuator_subs[omegadot_rnx]
])

B_sdc = sp.simplify(Xdot_full.jacobian(U))

"""
7. A(x) Matrix Construction (Sparse Engineering Factorization + LUT Thrust)
"""
print("-> Constructing Sparse A(x) Matrix...")
# Unforced drift moments (cmds = 0)
drift_subs = {phi_px_cmd: 0, phi_nx_cmd: 0, omega_rpx_cmd: 0, omega_rnx_cmd: 0}
M_drift = M_net.subs(actuator_subs).subs(drift_subs)

# A11: 3x3 Euler Coupling Block (Skew-Symmetric)
S_omega = sp.Matrix([
    [0, -omega_bz, omega_by],
    [omega_bz, 0, -omega_bx],
    [-omega_by, omega_bx, 0]
])
A11 = -sp.Inverse(I_b) * S_omega * I_b

# A22: 4x4 Actuator Drift Block
A22 = sp.diag(-1/tau_s, -1/tau_s, -1/tau_r, -1/tau_r)

# A12: 3x4 Actuator-to-Body Coupling Block (Dumping complexity into omega columns)
M_drift_px = M_drift.subs({phi_nx: 0, omega_rnx: 0, f_nx: 0})
M_drift_nx = M_drift.subs({phi_px: 0, omega_rpx: 0, f_px: 0})

col_omega_rpx = sp.simplify(sp.Inverse(I_b) * M_drift_px / omega_rpx)
col_omega_rnx = sp.simplify(sp.Inverse(I_b) * M_drift_nx / omega_rnx)
col_zero = sp.Matrix.zeros(3, 1)

A12 = sp.Matrix.hstack(col_zero, col_zero, col_omega_rpx, col_omega_rnx)

# A21: 4x3 Body-to-Actuator Coupling Block
A21 = sp.Matrix.zeros(4, 3)

# Assemble Full A(x)
A_top = sp.Matrix.hstack(A11, A12)
A_bot = sp.Matrix.hstack(A21, A22)
A_sdc = sp.Matrix.vstack(A_top, A_bot)

"""
8. Lambdify and Export (Dill)
"""
print("-> Lambdifying Matrices for Fast Evaluation...")

# Define the exact input signature. 
# We explicitly add f_px and f_nx so you can pass in your LUT thrust values!
vehicle_params = (Ixx, Iyy, Izz, tau_s, tau_r, k_s, k_r, j_rotor, r_pxx, r_pxz, r_nxx, r_nxz)
eval_args = tuple(X) + (f_px, f_nx) + vehicle_params

# Lambdify with numpy backend for instant 2D array generation
A_func = lambdify(eval_args, A_sdc, modules='numpy')
B_func = lambdify(eval_args, B_sdc, modules='numpy')

export_data = {
    'A_func': A_func,
    'B_func': B_func,
    'state_signature': X,
    'thrust_signature': (f_px, f_nx),
    'param_signature': vehicle_params
}

file_name = 'bicopter_sdre_matrices.pkl'

print(f"-> Pickling to '{file_name}'...")
with open(file_name, 'wb') as f:
    dill.dump(export_data, f)

print("\nSuccess! Plant derivation complete and exported.")