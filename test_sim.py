from plants import *
from util import *

# define sim dt (seconds)
dt_s = 1/1000
t_end_s = 5
n_steps = round(t_end_s/dt_s)

# define rotor
rotor_length_m = mm2m(180) # rotor length, x axis
rotor_width_m = mm2m(10) # rotor width, y axis
rotor_thickness_m = mm2m(1) # rotor height, z axis
rotor_mass_kg = g2kg(4.2) # rotor mass
rotor_I_kgm2 = rec_prism_inertia(rotor_length_m,rotor_width_m,rotor_thickness_m,rotor_mass_kg)
rotor_j_kgm2 = rotor_I_kgm2[2,2] # Rotor axial inertia

# define body
body_length_m = mm2m(200)
body_width_m = mm2m(40)
body_height_m = mm2m(80)
body_mass_kg = g2kg(800)
body_I_kgm2 = rec_prism_inertia(body_length_m,body_width_m,body_height_m,body_mass_kg) # body inertia tensor

# define positions of thrusters in body coordinates
r_cg2px_m = np.array([mm2m(100),0,mm2m(-50)]).reshape(-1, 1) # reshape to column
r_cg2nx_m = np.array([mm2m(-100),0,mm2m(-50)]).reshape(-1, 1) # reshape to column

# build rotor wheels
rotor_wheel_px = rotor_as_wheel(rotor_j_kgm2)
rotor_wheel_nx = rotor_as_wheel(rotor_j_kgm2)

# define rotor thruster rate to force lookup
raw_data = data = np.loadtxt('motor_lookup.csv', delimiter=',', skiprows=1)
w_rapds_2_f_N = lookup_1D(raw_data[:,1],raw_data[:,0])

# build rotors as thrusters
rotor_thruster_px = rotor_as_thruster(w_rapds_2_f_N,r_cg2px_m)
rotor_thruster_nx = rotor_as_thruster(w_rapds_2_f_N,r_cg2nx_m)

# build rigid body
body = simple_rigid_body(body_I_kgm2,dt_s)

# define some lookups for the actuators
w_rotor_px_lookup_radps = lookup_1D([-dt_s,t_end_s],[0,20000])
w_rotor_nx_lookup_rapds = lookup_1D([-dt_s,t_end_s],[0,-20000])
phi_tilter_px_lookup_rad = lookup_1D([-dt_s,t_end_s],[np.deg2rad(-5),np.deg2rad(5)])
phi_tilter_nx_lookup_rad = lookup_1D([-dt_s,t_end_s],[np.deg2rad(5),np.deg2rad(-5)])

# define log array
log_array = np.zeros((n_steps,12)) # [t_s wx wy wz phi_px phi_nx phidot_px phidot_nx w_px w_nx wdot_px wdot_nx]
log_col_names = ["t_s",
                 "wx_radps","wy_radps","wz_radps",
                 "phi_px_rad","phi_nx_rad","phidot_px_radps","phidot_nx_radps",
                 "w_px_radps,w_nx_radps,wdot_px_radps2,wdot_nx_radps"]
log_col_names = ','.join(log_col_names)

# perform simulation
t_s = 0
for i_step in range(n_steps):

    # pull body rate
    W_B_wrt_I_radps = body.W_B_wrt_I_radps

    # get actuator states
    w_rotor_px_radps = w_rotor_px_lookup_radps.index(t_s)
    w_rotor_nx_radps = w_rotor_nx_lookup_rapds.index(t_s)
    phi_tilter_px_rad = phi_tilter_px_lookup_rad.index(t_s)
    phi_tilter_nx_rad = phi_tilter_nx_lookup_rad.index(t_s)

    # get actuator rates of change
    wdot_rotor_px_radps2 = (w_rotor_px_radps - w_rotor_px_lookup_radps.index(t_s-dt_s))/dt_s
    wdot_rotor_nx_radps2 = (w_rotor_nx_radps - w_rotor_nx_lookup_rapds.index(t_s-dt_s))/dt_s
    phidot_tilter_px_radps = (phi_tilter_px_rad - phi_tilter_px_lookup_rad.index(t_s-dt_s))/dt_s
    phidot_tilter_nx_radps = (phi_tilter_nx_rad - phi_tilter_nx_lookup_rad.index(t_s-dt_s))/dt_s

    # get wheel moments on rigid body
    M_px_wheel_Nm = rotor_wheel_px.get_M(w_rotor_px_radps,wdot_rotor_px_radps2,phi_tilter_px_rad,phidot_tilter_px_radps,W_B_wrt_I_radps)
    M_nx_wheel_Nm = rotor_wheel_nx.get_M(w_rotor_nx_radps,wdot_rotor_nx_radps2,phi_tilter_nx_rad,phidot_tilter_nx_radps,W_B_wrt_I_radps)

    # get thrust moments on rigid body
    M_px_thrust_Nm = rotor_thruster_px.get_M(phi_tilter_px_rad,w_rotor_px_radps)
    M_nx_thrust_Nm = rotor_thruster_px.get_M(phi_tilter_nx_rad,w_rotor_nx_radps)

    # get net moment on rigid body
    M_net_Nm = M_px_wheel_Nm + M_nx_wheel_Nm + M_px_thrust_Nm + M_nx_thrust_Nm

    # update rigid body state
    body.step(M_net_Nm)

    # stack everything
    vertical_state = np.vstack((t_s,
                                W_B_wrt_I_radps,
                                phi_tilter_px_rad,
                                phi_tilter_nx_rad,
                                phidot_tilter_px_radps,
                                phidot_tilter_nx_radps,
                                w_rotor_px_radps,
                                w_rotor_nx_radps,
                                wdot_rotor_px_radps2,
                                wdot_rotor_nx_radps2))
    
    log_array[i_step,:] = np.transpose(vertical_state)

    # update time
    t_s += dt_s

np.savetxt('log.csv', log_array, delimiter=',', fmt='%.3f', header=log_col_names, comments='')