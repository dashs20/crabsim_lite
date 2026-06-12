import numpy as np
import os
from util import *
from scipy.linalg import solve_discrete_are, inv
from config.config_util import ConfigStruct
from gnc.gnc import compute_T_hat

class lqr_live:
    def __init__(self,Q,R):
        self.Q = Q
        self.R = R
        self.R_inv = inv(self.R)

    def get_u(self,A_live,B_live,x_est,x_des):

        # solve for P via DARE
        x_err = x_des - x_est
        P = solve_discrete_are(A_live,B_live,self.Q,self.R)

        # solve for gain matrix K
        K = self.R_inv @ np.transpose(B_live) @ P

        # solve for actuator effort U
        return -K @ x_err
    
class AB_live:
    def __init__(self, gnc_config: ConfigStruct):
        self.gnc_config = gnc_config
        plant = self.gnc_config.plant
        
        # upon instantiation, derive plant parameters required for A and B
        self.j_rotor_kgm2 = rec_prism_inertia(
            mm2m(plant.rotor_length_mm),
            mm2m(plant.rotor_width_mm),
            mm2m(plant.rotor_thickness_mm),
            g2kg(plant.rotor_mass_g)
        )[2, 2] # only axial inertia required
        self.I_body_kgm2 = rec_prism_inertia(
            mm2m(plant.body_length_mm),
            mm2m(plant.body_width_mm),
            mm2m(plant.body_height_mm),
            g2kg(plant.body_mass_g)
        )
        self.r_px_m = mm2m(np.array(plant.r_cg2px_mm))
        self.r_nx_m = mm2m(np.array(plant.r_cg2nx_mm))
        self.tau_s_s = ms2s(plant.servo_time_constant_ms)
        self.tau_m_s = ms2s(plant.motor_time_constant_ms)

        self.I_body_kgm2_inv = np.linalg.inv(self.I_body_kgm2)

        # build lookup for rotor
        motor_csv_path = os.path.join('models/plant_sys_id', plant.motor_lookup)
        motor_csv_data = np.genfromtxt(motor_csv_path, delimiter=',', names=True)
        self.omega_r_2_f_N = lookup_1D(motor_csv_data['w_radps'], motor_csv_data['f_N'])

    def A_live(self,x):

        # unpack state vector
        omega_b_radps = x[0:3]
        phi_rad = x[2:4]
        omega_r_radps = x[4:6]

        # get T_hats
        T_hat_px = compute_T_hat(phi_rad[0])
        T_hat_nx = compute_T_hat(phi_rad[1])

        # A[1:3,1:3]
        A_13_13 = self.I_body_kgm2_inv @ np.array([[0,0,omega_b_radps[1] * (self.I_body_kgm2[2,2] - self.I_body_kgm2[1,1])],
                                                   [omega_b_radps[2] * (self.I_body_kgm2[0,0] - self.I_body_kgm2[2,2]),0,0],
                                                   [0,omega_b_radps[0] * (self.I_body_kgm2[1,1] - self.I_body_kgm2[0,0]),0]])

        # A[1:3,4:5]
        A_13_45 = self.I_body_kgm2_inv @ np.array([[(-self.r_px_m[2]*np.sin(phi_rad[0]))/phi_rad[0] if phi_rad[0] != 0 else -self.r_px_m[2],0],
                                                   [0,0],
                                                   [0,0]])


