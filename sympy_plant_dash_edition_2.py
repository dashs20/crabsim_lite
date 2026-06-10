import sympy as sp

"""
Inputs
"""
# phi
u_phi_px = sp.symbols('u_phi_px')
u_phi_nx = sp.symbols('u_phi_nx')

# wr
u_wr_px = sp.symbols('u_wr_px')
u_wr_nx = sp.symbols('u_wr_nx')

"""
Actuator dynamics
"""
tau_s = sp.symbols('tau_s')
tau_m = sp.symbols('tau_m')


"""
State, dState
"""
# Wr
wr_px = sp.symbols('wr_px')
dwr_px = sp.symbols('dwr_px')

wr_nx = sp.symbols('wr_nx')
dwr_nx = sp.symbols('dwr_nx')

# f(Wr)
fwr_px = sp.symbols('fwr_px')
fwr_nx = sp.symbols('fwr_nx')

# phi
phi_px = sp.symbols('phi_px')
dphi_px = sp.symbols('dphi_px')

phi_nx = sp.symbols('phi_nx')
dphi_nx = sp.symbols('dphi_nx')

# That
That_px = sp.Matrix([[0],[sp.sin(phi_px)],[-sp.cos(phi_px)]])
dThat_px = sp.Matrix([[0],[dphi_px * sp.cos(phi_px)],[dphi_px * sp.sin(phi_px)]])

That_nx = sp.Matrix([[0],[sp.sin(phi_nx)],[-sp.cos(phi_nx)]])
dThat_nx = sp.Matrix([[0],[dphi_nx * sp.cos(phi_nx)],[dphi_nx * sp.sin(phi_nx)]])

# Wb
wb_x, wb_y, wb_z = sp.symbols('wb_x wb_y wb_z')
dwb_x, dwb_y, dwb_z = sp.symbols('dwb_x dwb_y dwb_z')
wb = sp.Matrix([[wb_x], [wb_y], [wb_z]])
dwb = sp.Matrix([[dwb_x], [dwb_y], [dwb_z]])

"""
Params
"""
# j_rotor
j_rotor = sp.symbols('j_rotor')

# r
r_px_x, r_px_y, r_px_z = sp.symbols('r_px_x r_px_y r_px_z')
r_px = sp.Matrix([[r_px_x],[r_px_y],[r_px_z]])

r_nx_x, r_nx_y, r_nx_z = sp.symbols('r_nx_x r_nx_y r_nx_z')
r_nx = sp.Matrix([[r_nx_x],[r_nx_y],[r_nx_z]])

# I_b
ib_xx, ib_yy, ib_zz = sp.symbols('ib_xx ib_yy ib_zz')
Ib = sp.Matrix([[ib_xx,0,0],[0,ib_yy,0],[0,0,ib_zz]])


"""
Define chunks
"""
## dynamics
M_wheel_px = -j_rotor * (dThat_px * wr_px + dwr_px * That_px.cross(wb) + wr_px * That_px.cross(wb))
M_wheel_nx = -j_rotor * (dThat_nx * wr_nx + dwr_nx * That_nx.cross(wb) + wr_nx * That_nx.cross(wb))
M_thrust_px = r_px.cross(That_px * fwr_px)
M_thrust_nx = r_nx.cross(That_nx * fwr_nx)
rhs = Ib * dwb + wb.cross(Ib * wb)

# actuator models


"""
Define raw equations
"""
fucked_dynamics = sp.Eq(rhs, M_wheel_px + M_wheel_nx + M_thrust_px + M_thrust_nx)

dphi_px_eq = sp.Eq(dphi_px,-1/tau_s * phi_px + 1/tau_s * u_phi_px)
dphi_nx_eq = sp.Eq(dphi_nx,-1/tau_s * phi_nx + 1/tau_s * u_phi_nx)
dwr_px_eq = sp.Eq(dwr_px,-1/tau_m * wr_px + 1/tau_m * u_wr_px)
dwr_nx_eq = sp.Eq(dwr_nx,-1/tau_m * wr_nx + 1/tau_m * u_wr_nx)

"""
Make Sympy my bitch
"""
# 1. Define the actuator derivative expressions
dphi_px_expr = -phi_px/tau_s + u_phi_px/tau_s
dphi_nx_expr = -phi_nx/tau_s + u_phi_nx/tau_s
dwr_px_expr = -wr_px/tau_m + u_wr_px/tau_m
dwr_nx_expr = -wr_nx/tau_m + u_wr_nx/tau_m

# 2. Decouple the dynamics
dyn_subs = fucked_dynamics.subs({
    dphi_px: dphi_px_expr,
    dphi_nx: dphi_nx_expr,
    dwr_px: dwr_px_expr,
    dwr_nx: dwr_nx_expr
})

# 3. Solve for dwb components
dwb_system = [dyn_subs.lhs[i] - dyn_subs.rhs[i] for i in range(3)]
dwb_solution = sp.solve(dwb_system, [dwb_x, dwb_y, dwb_z])

# 4. Create the final "manly" dictionary for LaTeX
# We merge the solved dynamics with the raw actuator ODEs
final_system = {
    dwb_x: dwb_solution[dwb_x],
    dwb_y: dwb_solution[dwb_y],
    dwb_z: dwb_solution[dwb_z],
    dphi_px: dphi_px_expr,
    dphi_nx: dphi_nx_expr,
    dwr_px: dwr_px_expr,
    dwr_nx: dwr_nx_expr
}

# 5. Output the LaTeX block
print(r"\begin{align*}")
for key, expr in final_system.items():
    lhs = sp.latex(key)
    rhs_expr = sp.latex(sp.simplify(expr))
    
    # Use multlined for the heavy dynamics, standard format for actuator ODEs
    if key in [dwb_x, dwb_y, dwb_z]:
        # Extract the axis (x, y, or z) from the symbol name for the ib denominator
        axis = str(key)[-1]
        print(f"    {lhs} &= \\frac{{1}}{{ib_{{{axis}}}}} \\left( \\begin{{multlined}}[t] {rhs_expr} \\end{{multlined}} \\right) \\\\")
    else:
        print(f"    {lhs} &= {rhs_expr} \\\\")
print(r"\end{align*}")