from plants import *
from gnc import *
from util import *
from sensor import ICM42688

# define sim dt (seconds)
dt_s = 1/1000
t_end_s = 20
n_steps = round(t_end_s/dt_s)
t_s = np.linspace(0,t_end_s,n_steps)

# define bicopter
crabcopter = bicopter('crabcopter.yaml',dt_s)

# define gnc
crabbrain = gnc('sboc.yaml')

# start the rotors off at a nonzero speed
crabcopter.px_motor_model.x =   crabcopter.w_rapds_2_f_N.index_y(5)
crabcopter.nx_motor_model.x = - crabcopter.w_rapds_2_f_N.index_y(5)

# define gyro sensor
gyro = ICM42688(1000,200)

# define some controller command lookups
t_lookup_s = np.linspace(-dt_s,t_end_s,9)
wx_cmd_lookup_radps = lookup_1D(t_lookup_s,np.deg2rad([0,0,0,0,0,0,0,0,0]))
wy_cmd_lookup_radps = lookup_1D(t_lookup_s,np.deg2rad([0,0,0,200,200,200,0,0,0]))
wz_cmd_lookup_radps = lookup_1D(t_lookup_s,np.deg2rad([0,0,0,0,0,0,0,0,0]))

# define plant log array
plant_log_array = np.zeros((n_steps,20))
plant_log_col_names = ["wx_radps","wy_radps","wz_radps",
                       "phi_px_rad","phi_nx_rad","phidot_px_radps","phidot_nx_radps",
                       "w_px_radps","w_nx_radps","wdot_px_radps2","wdot_nx_radps2",
                       "F_px_x_N","F_px_y_N","F_px_z_N",
                       "F_nx_x_N","F_nx_y_N","F_nx_z_N",
                       "M_net_x_Nm","M_net_y_Nm","M_net_z_Nm"]
plant_log_col_names = ','.join(plant_log_col_names)

# define GNC log array
gnc_log_array = np.zeros((n_steps,10))
gnc_log_col_names = ["wx_cmd_radps","wy_cmd_radps","wz_cmd_radps",
                     "wx_meas_radps","wy_meas_radps","wz_meas_radps",
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

    # get gyro measurement of body rate
    # w_meas_radps = gyro.get_measurement(crabcopter.body.W_B_wrt_I_radps)

    # get actuator state from plant
    act_est = np.array([crabcopter.px_servo_model.x,
                        crabcopter.nx_servo_model.x,
                        crabcopter.px_motor_model.x,
                        crabcopter.nx_motor_model.x])

    # step GNC & log outputs
    act_cmd = crabbrain.step(w_des_radps,crabcopter.body.W_B_wrt_I_radps,act_est,0.25)
    gnc_log_array[i_step,:] = np.hstack((w_des_radps, crabcopter.body.W_B_wrt_I_radps.flatten(), act_cmd))

    # step plant & log output
    plant_log_array[i_step,:] = crabcopter.step(act_cmd)

# build composite log
log_array = np.hstack((t_s[:,np.newaxis],plant_log_array,gnc_log_array))
log_col_names = "t_s," + plant_log_col_names + "," + gnc_log_col_names
np.savetxt('log.csv', log_array, delimiter=',', fmt='%.3f', header=log_col_names, comments='')