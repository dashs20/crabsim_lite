import numpy as np
from util.integrator import rk4
from util import *
import yaml
from models.tf import tf1

def compute_T_hat(phi_rad):
    return np.array([[0],[np.sin(phi_rad)],[-np.cos(phi_rad)]])

def compute_Tdot_hat(phi_rad,phidot_rad):
    return np.array([[0],[phidot_rad*np.cos(phi_rad)],[phidot_rad*np.sin(phi_rad)]])

class rotor_as_wheel:
    def __init__(self,j_kgm2):
        self.j_kgm2 = j_kgm2 # axial moment of inertia of the rotor (1,1)

    def get_M(self,
        w_rotor_radps, # speed of rotor (1,1)
        wdot_rotor_radps2, # acceleration of rotor (1,1)
        phi_rotor_rad, # angle of rotor with respect to -body z (1,1)
        phidot_rotor_radps, # rate change of angle of rotor (1,1)
        W_B_wrt_I_radps): # angular rate of body (3,1)
        
        T_hat = compute_T_hat(phi_rotor_rad)
        Tdot_hat = compute_Tdot_hat(phi_rotor_rad,phidot_rotor_radps)

        return -self.j_kgm2 * (Tdot_hat * w_rotor_radps + T_hat * wdot_rotor_radps2 + np.cross(W_B_wrt_I_radps,T_hat * w_rotor_radps,axis=0))
    
class rotor_as_thruster:
    def __init__(self,w_radps_2_f_N,r_cg2r_m):
        self.w_radps_2_f_N = w_radps_2_f_N # lookup table mapping rotor speed to rotor thrust (1,1) -> (1,1)
        self.r_cg2r_m = r_cg2r_m # vector going from the body CG to the point the rotor applies force (3,1)

    def get_M(self,
        phi_rotor_rad, # angle of rotor with respect to -body z (1,1)
        w_rotor_radps): # speed of rotor (1,1)
          
        T_hat = compute_T_hat(phi_rotor_rad)
        f_N = self.w_radps_2_f_N.index(np.abs(w_rotor_radps)) # thrust maps to |rotor speed|
        F_N = f_N * T_hat
        return np.cross(self.r_cg2r_m,F_N,axis=0), F_N
    
class simple_rigid_body:
    def __init__(self,I_kgm2,dt_s):
        self.I_kgm2 = I_kgm2 # Inertia tensor for rigid body (3,3)
        self.I_inv = np.linalg.inv(self.I_kgm2)
        self.W_B_wrt_I_radps = np.zeros((3,1)) # angular rate of body (3,1)
        self.dt_s = dt_s # timestep for numerical integration (1,1)
        self.M_net_Nm = np.zeros((3,1)) # moment on body

    def step(self,M_net_Nm):
        
        def dstate(W_B_wrt_I_radps):
            M_net_Nm = self.M_net_Nm
            return self.I_inv @ (M_net_Nm - np.cross(W_B_wrt_I_radps,self.I_kgm2 @ W_B_wrt_I_radps,axis=0))
        
        self.M_net_Nm = M_net_Nm
        self.W_B_wrt_I_radps = rk4(dstate,self.W_B_wrt_I_radps,self.dt_s)

class bicopter:
    def __init__(self,yaml_path,dt_s,config_dict=None):

        # load config
        if config_dict is not None:
            data = config_dict
        else:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

        # save timestep size
        self.dt_s = dt_s

        # body inertia tensor
        self.body_I_kgm2 = rec_prism_inertia(mm2m(data['rotor_length_mm']),
                                             mm2m(data['body_width_mm']),
                                             mm2m(data['body_height_mm']),
                                             g2kg(data['body_mass_g']))
        
        # define positions of thrusters in body coordinates
        self.r_cg2px_m = np.array(mm2m(np.array(data['r_cg2px_mm']))).reshape(-1, 1)
        self.r_cg2nx_m = np.array(mm2m(np.array(data['r_cg2nx_mm']))).reshape(-1, 1)

        # Rotor axial inertia
        rotor_I_kgm2 = rec_prism_inertia(mm2m(data['rotor_length_mm']),
                                         mm2m(data['rotor_width_mm']),
                                         mm2m(data['rotor_thickness_mm']),
                                         g2kg(data['rotor_mass_g']))
        self.rotor_j_kgm2 = rotor_I_kgm2[2,2]

        # build rotor wheels
        self.rotor_wheel_px = rotor_as_wheel(self.rotor_j_kgm2)
        self.rotor_wheel_nx = rotor_as_wheel(self.rotor_j_kgm2)

        # define rotor thruster rate to force lookup
        raw_data = np.loadtxt(data['motor_lookup'], delimiter=',', skiprows=1)
        self.w_rapds_2_f_N = lookup_1D(raw_data[:,1],raw_data[:,0])
        self.max_motor_f_N = np.max(raw_data[:,0])

        # build rotors as thrusters
        self.rotor_thruster_px = rotor_as_thruster(self.w_rapds_2_f_N,self.r_cg2px_m)
        self.rotor_thruster_nx = rotor_as_thruster(self.w_rapds_2_f_N,self.r_cg2nx_m)

        # build rigid body
        self.body = simple_rigid_body(self.body_I_kgm2,self.dt_s)

        # build actuator models
        self.px_motor_model = tf1(ms2s(data['motor_time_constant_ms']),1,self.dt_s)
        self.nx_motor_model = tf1(ms2s(data['motor_time_constant_ms']),1,self.dt_s)
        self.px_servo_model = tf1(ms2s(data['servo_time_constant_ms']),1,self.dt_s)
        self.nx_servo_model = tf1(ms2s(data['servo_time_constant_ms']),1,self.dt_s)
        
        self.thrust_enabled = True

    def step(self,act_cmd):

        # unpack command vector
        thr_px_cmd_frac = act_cmd[0]
        thr_nx_cmd_frac = act_cmd[1]
        phi_px_cmd_rad = act_cmd[2]
        phi_nx_cmd_rad = act_cmd[3]

        # convert throttle fractions into thrusts
        thr_px_cmd_N = thr_px_cmd_frac * self.max_motor_f_N
        thr_nx_cmd_N = thr_nx_cmd_frac * self.max_motor_f_N

        # convert motor thrusts into omegas
        thr_px_cmd_radps = self.w_rapds_2_f_N.index_y(thr_px_cmd_N)
        thr_nx_cmd_radps = self.w_rapds_2_f_N.index_y(thr_nx_cmd_N) * -1 # spin nx rotor backwards

        # step actuator models; obtain actuator states and dstates
        w_rotor_px_radps, wdot_rotor_px_radps2 = self.px_motor_model.step(thr_px_cmd_radps)
        w_rotor_nx_radps, wdot_rotor_nx_radps2 = self.nx_motor_model.step(thr_nx_cmd_radps)
        phi_tilter_px_rad, phidot_tilter_px_radps = self.px_servo_model.step(phi_px_cmd_rad)
        phi_tilter_nx_rad, phidot_tilter_nx_radps = self.nx_servo_model.step(phi_nx_cmd_rad)

        # get wheel moments on rigid body
        M_px_wheel_Nm = self.rotor_wheel_px.get_M(w_rotor_px_radps,wdot_rotor_px_radps2,phi_tilter_px_rad,phidot_tilter_px_radps,self.body.W_B_wrt_I_radps)
        M_nx_wheel_Nm = self.rotor_wheel_nx.get_M(w_rotor_nx_radps,wdot_rotor_nx_radps2,phi_tilter_nx_rad,phidot_tilter_nx_radps,self.body.W_B_wrt_I_radps)

        # get thrust moments on rigid body
        M_px_thrust_Nm, F_px_N = self.rotor_thruster_px.get_M(phi_tilter_px_rad,w_rotor_px_radps)
        M_nx_thrust_Nm, F_nx_N = self.rotor_thruster_nx.get_M(phi_tilter_nx_rad,w_rotor_nx_radps)

        # get net moment on rigid body
        if self.thrust_enabled:
            M_net_Nm = M_px_wheel_Nm + M_nx_wheel_Nm + M_px_thrust_Nm + M_nx_thrust_Nm
        else:
            M_net_Nm = M_px_wheel_Nm + M_nx_wheel_Nm

        # update rigid body state
        self.body.step(M_net_Nm)

        # return full vehicle state
        state = np.vstack((self.body.W_B_wrt_I_radps,
                           phi_tilter_px_rad,
                           phi_tilter_nx_rad,
                           phidot_tilter_px_radps,
                           phidot_tilter_nx_radps,
                           w_rotor_px_radps,
                           w_rotor_nx_radps,
                           wdot_rotor_px_radps2,
                           wdot_rotor_nx_radps2,
                           F_px_N,
                           F_nx_N,
                           M_net_Nm))
        
        return np.transpose(state)