from plants import *
from gnc import *
from util import *

# define sim dt (seconds)
dt_s = 1/4000
t_end_s = 1
n_steps = round(t_end_s/dt_s)
t_s = np.linspace(0,t_end_s,n_steps)

# define bicopter
crabcopter = bicopter('crabcopter.yaml',dt_s)

# define gnc
crabbrain = gnc('crabcopter_gnc.yaml',dt_s)

# define some controller command lookups
wx_cmd_lookup_radps = lookup_1D([-dt_s,t_end_s],np.deg2rad([0,30]))
wy_cmd_lookup_radps = lookup_1D([-dt_s,t_end_s],np.deg2rad([0,-10]))
wz_cmd_lookup_radps = lookup_1D([-dt_s,t_end_s],np.deg2rad([0,15]))
thr_lookup_frac = lookup_1D([-dt_s,t_end_s],[0.4,0.4])

# define plant log array
plant_log_array = np.zeros((n_steps,17))
plant_log_col_names = ["wx_radps","wy_radps","wz_radps",
                       "phi_px_rad","phi_nx_rad","phidot_px_radps","phidot_nx_radps",
                       "w_px_radps,w_nx_radps,wdot_px_radps2,wdot_nx_radps",
                       "F_px_x_N","F_px_y_N","F_px_z_N",
                       "F_nx_x_N","F_nx_y_N","F_nx_z_N"]
plant_log_col_names = ','.join(plant_log_col_names)

# define GNC log array
gnc_log_array = np.zeros((n_steps,10))
gnc_log_col_names = ["wx_cmd_radps","wy_cmd_radps","wz_cmd_radps",
                     "Mx_cmd_Nm","My_cmd_Nm","Mz_cmd_Nm",
                     "fpx_frac","fnx_frac","phi_px_cmd_rad","phi_nx_cmd_rad"]
gnc_log_col_names = ','.join(gnc_log_col_names)

# perform simulation
for i_step in range(n_steps):

    t_cur_s = t_s[i_step]

    # get controller commands
    wx_cmd_radps = wx_cmd_lookup_radps.index(t_cur_s)
    wy_cmd_radps = wy_cmd_lookup_radps.index(t_cur_s)
    wz_cmd_radps = wz_cmd_lookup_radps.index(t_cur_s)
    w_des_radps = np.array([wx_cmd_radps,wy_cmd_radps,wz_cmd_radps])
    thr_frac = thr_lookup_frac.index(t_cur_s)

    # step GNC & log outputs
    gnc_state = crabbrain.step(w_des_radps,crabcopter.body.W_B_wrt_I_radps,thr_frac)
    gnc_log_array[i_step,:] = np.hstack((w_des_radps, gnc_state))
    act_cmd = gnc_state[3:7]

    # step plant & log output
    plant_log_array[i_step,:] = crabcopter.step(act_cmd)

# build composite log
log_array = np.hstack((t_s[:,np.newaxis],plant_log_array,gnc_log_array))
log_col_names = "t_s," + plant_log_col_names + "," + gnc_log_col_names
np.savetxt('log.csv', log_array, delimiter=',', fmt='%.3f', header=log_col_names, comments='')