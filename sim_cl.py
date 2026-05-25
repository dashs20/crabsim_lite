from plants import *
from gnc import *
from util import *
from sensor import ICM42688
from csv_to_pdf_smart import generate_smart_pdf_plots

# load guidance table from CSV
guidance_data = np.loadtxt('guidance.csv', delimiter=',', skiprows=1)
t_lookup_s = guidance_data[:,0]
wx_cmd_lookup_radps = lookup_1D(t_lookup_s, guidance_data[:,1])
wy_cmd_lookup_radps = lookup_1D(t_lookup_s, guidance_data[:,2])
wz_cmd_lookup_radps = lookup_1D(t_lookup_s, guidance_data[:,3])
thr_frac_lookup = lookup_1D(t_lookup_s, guidance_data[:,4])

# define sim dt (seconds)
dt_s = 1/2000
t_end_s = t_lookup_s[-1]
n_steps = round(t_end_s/dt_s)
t_s = np.linspace(0,t_end_s,n_steps)

# define bicopter
crabcopter = bicopter('crabcopter.yaml',dt_s)

# define gnc
crabbrain = gnc('sboc.yaml')

# start the rotors off at a nonzero speed based on the initial throttle fraction
initial_thrust_N = thr_frac_lookup.index(0.0) * crabcopter.max_motor_f_N
crabcopter.px_motor_model.x =   crabcopter.w_rapds_2_f_N.index_y(initial_thrust_N)
crabcopter.nx_motor_model.x = - crabcopter.w_rapds_2_f_N.index_y(initial_thrust_N)

# define gyro sensor
gyro = ICM42688(1000,200)

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
gnc_log_array = np.zeros((n_steps,11))
gnc_log_col_names = ["wx_cmd_radps","wy_cmd_radps","wz_cmd_radps",
                     "wx_meas_radps","wy_meas_radps","wz_meas_radps",
                     "fpx_frac","fnx_frac","phi_px_cmd_rad","phi_nx_cmd_rad",
                     "thr_cmd_frac"]
gnc_log_col_names = ','.join(gnc_log_col_names)

# perform simulation
for i_step in range(n_steps):

    t_cur_s = t_s[i_step]

    # get controller commands
    wx_cmd_radps = wx_cmd_lookup_radps.index(t_cur_s)
    wy_cmd_radps = wy_cmd_lookup_radps.index(t_cur_s)
    wz_cmd_radps = wz_cmd_lookup_radps.index(t_cur_s)
    thr_frac_cmd = thr_frac_lookup.index(t_cur_s)
    w_des_radps = np.array([wx_cmd_radps,wy_cmd_radps,wz_cmd_radps])

    # get gyro measurement of body rate
    w_meas_radps = gyro.get_measurement(crabcopter.body.W_B_wrt_I_radps)

    # get actuator state from plant
    act_est = np.array([crabcopter.px_servo_model.x,
                        crabcopter.nx_servo_model.x,
                        crabcopter.px_motor_model.x,
                        crabcopter.nx_motor_model.x])

    # step GNC & log outputs
    act_cmd = crabbrain.step(w_des_radps,crabcopter.body.W_B_wrt_I_radps,act_est,thr_frac_cmd)
    gnc_log_array[i_step,:] = np.hstack((w_des_radps, w_meas_radps.flatten(), act_cmd, thr_frac_cmd))

    # step plant & log output
    plant_log_array[i_step,:] = crabcopter.step(act_cmd)

# build composite log
log_array = np.hstack((t_s[:,np.newaxis],plant_log_array,gnc_log_array))
log_col_names = "t_s," + plant_log_col_names + "," + gnc_log_col_names
np.savetxt('log.csv', log_array, delimiter=',', fmt='%.3f', header=log_col_names, comments='')
generate_smart_pdf_plots("log.csv")