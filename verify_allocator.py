from gnc import allocator
import yaml
from util import *
from plants import compute_T_hat

# open yaml
with open('crabcopter_gnc.yaml', 'r') as file:
    data = yaml.safe_load(file)

r_cg2px_m = mm2m(np.array(data['r_cg2px_mm']))
r_cg2nx_m = mm2m(np.array(data['r_cg2nx_mm']))

# build allocator
alloc = allocator(r_cg2px_m,
                  r_cg2nx_m,
                  data['motor_lookup'],
                  data['min_thr_frac'],
                  np.deg2rad(data['phi_max_deg']))

# fire off some tests
M_des_Nm = np.array([0.5,-0.1,0])
act = alloc.allocate(M_des_Nm,0.3)

# Get T_hats
Tpx = compute_T_hat(act[2])
Tnx = compute_T_hat(act[3])

# manually recompute moment as a function of actuator commands
M_px = np.cross(r_cg2px_m,np.transpose(Tpx * act[0] * alloc.f_max_N))
M_nx = np.cross(r_cg2nx_m,np.transpose(Tnx * act[1] * alloc.f_max_N))
M_recompute = M_px + M_nx

phi_px_deg = np.rad2deg(act[2])
phi_nx_deg = np.rad2deg(act[3])

print(M_recompute - M_des_Nm)