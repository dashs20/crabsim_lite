import sympy as sp

# 1. Define your symbols
fpx_y, fpx_z = sp.symbols('fpx_y fpx_z')
fnx_y, fnx_z = sp.symbols('fnx_y fnx_z')
rpx_x, rpx_z = sp.symbols('rpx_x rpx_z')
rnx_x, rnx_z = sp.symbols('rnx_x rnx_z')

# 2. Setup symbolic matrices
rpx = sp.Matrix([[rpx_x], [0], [rpx_z]])
rnx = sp.Matrix([[rnx_x], [0], [rnx_z]])
fpx = sp.Matrix([[0], [fpx_y], [fpx_z]])
fnx = sp.Matrix([[0], [fnx_y], [fnx_z]])

# 3. Calculate the resulting moments
M_calc = rpx.cross(fpx) + rnx.cross(fnx)

# 4. Define the 4th constraint equation for Total Thrust (f)
f_calc = fpx_z + fnx_z

# 5. Combine the moments and the thrust constraint into a 4x1 vector
# This represents the right side of: [Mx, My, Mz, f]^T = A * S
System_Eqs = sp.Matrix([M_calc[0], M_calc[1], M_calc[2], f_calc])

# 6. Define the vector S you want to factor out
S = [fpx_y, fpx_z, fnx_y, fnx_z]

# 7. Extract the 4x4 coefficient matrix (A)
A, _ = sp.linear_eq_to_matrix(System_Eqs, S)

print("4x4 Control Allocation Matrix A:")
sp.pprint(A)