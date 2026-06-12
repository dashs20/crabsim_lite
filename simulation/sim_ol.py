from models.plants import *
from util import *

# define sim dt (seconds)
dt_s = 1/1000
t_end_s = 5
n_steps = round(t_end_s/dt_s)

# define bicopter
crabcopter = bicopter('crabcopter.yaml',dt_s)

# define some actuator command lookups
thr_px_cmd_lookup_frac = lookup_1D([-dt_s,t_end_s],[0,1])
thr_nx_cmd_lookup_frac = lookup_1D([-dt_s,t_end_s],[0,0.9])
phi_px_cmd_lookup_deg = lookup_1D([-dt_s,t_end_s],[-10,10])
phi_nx_cmd_lookup_deg = lookup_1D([-dt_s,t_end_s],[10,-10])

# define log array
log_array = np.zeros((n_steps,18)) # [t_s wx wy wz phi_px phi_nx phidot_px phidot_nx w_px w_nx wdot_px wdot_nx]
log_col_names = ["t_s",
                 "wx_radps","wy_radps","wz_radps",
                 "phi_px_rad","phi_nx_rad","phidot_px_radps","phidot_nx_radps",
                 "w_px_radps,w_nx_radps,wdot_px_radps2,wdot_nx_radps",
                 "F_px_x_N","F_px_y_N","F_px_z_N",
                 "F_nx_x_N","F_nx_y_N","F_nx_z_N"]
log_col_names = ','.join(log_col_names)

# perform simulation
t_s = 0

for i_step in range(n_steps):

    # build actuator command from lookups
    thr_px_cmd_frac = thr_px_cmd_lookup_frac.index(t_s)
    thr_nx_cmd_frac = thr_nx_cmd_lookup_frac.index(t_s)
    phi_px_cmd_deg = phi_px_cmd_lookup_deg.index(t_s)
    phi_nx_cmd_deg = phi_nx_cmd_lookup_deg.index(t_s)
    act_cmd = [thr_px_cmd_frac,thr_nx_cmd_frac,phi_px_cmd_deg,phi_nx_cmd_deg]

    # step plant & log output
    log_array[i_step,:] = np.hstack((np.array([[t_s]]),crabcopter.step(act_cmd)))

    # update time
    t_s += dt_s

np.savetxt('log.csv', log_array, delimiter=',', fmt='%.3f', header=log_col_names, comments='')