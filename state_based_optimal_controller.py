import numpy as np
import dill
import yaml
from util import *
from scipy.linalg import solve_continuous_are, inv

class sboc:
    def __init__(self,sboc_yaml,sdre_pickle_path):

        # load A(x) and B(x)
        with open(sdre_pickle_path, 'rb') as f:
            plant = dill.load(f)

        self.A_func = plant['A_func']
        self.B_func = plant['B_func']

        # Load vehicle configuration data
        with open(sboc_yaml, 'r') as file:
            data = yaml.safe_load(file)

        # (Ixx, Iyy, Izz, tau_s, tau_r, k_s, k_r, j_rotor, r_pxx, r_pxz, r_nxx, r_nxz)
        
        # compute body inertia
        body_I_kgm2 = rec_prism_inertia(mm2m(data['rotor_length_mm']),
                                        mm2m(data['body_width_mm']),
                                        mm2m(data['body_height_mm']),
                                        g2kg(data['body_mass_g']))
        Ixx = body_I_kgm2[0,0]
        Iyy = body_I_kgm2[1,1]
        Izz = body_I_kgm2[2,2]

        # pull actuator data
        tau_s = ms2s(data['motor_time_constant_ms'])
        tau_r = ms2s(data['servo_time_constant_ms'])
        k = 1

        # compute rotor inertia
        rotor_I_kgm2 = rec_prism_inertia(mm2m(data['rotor_length_mm']),
                                         mm2m(data['rotor_width_mm']),
                                         mm2m(data['rotor_thickness_mm']),
                                         g2kg(data['rotor_mass_g']))
        j_rotor = rotor_I_kgm2[2,2]

        # pull vehicle geometry
        r_pxx = mm2m(data['r_cg2px_mm'][0])
        r_pxz = mm2m(data['r_cg2px_mm'][2])
        r_nxx = mm2m(data['r_cg2nx_mm'][0])
        r_nxz = mm2m(data['r_cg2nx_mm'][2])

        # build params tuple
        self.params = (Ixx, Iyy, Izz, tau_s, tau_r, 1, 1, j_rotor, r_pxx, r_pxz, r_nxx, r_nxz)

        # build lookup mapping thrust to omega
        raw_data = np.loadtxt(data['motor_lookup'], delimiter=',', skiprows=1)
        self.w_rapds_2_f_N = lookup_1D(raw_data[:,1],raw_data[:,0])

        # build Q and R matrices for CARE
        max_omega_b_error_radps = np.array(data['Q_max_acceptable_error_radps'])
        Qii_omega_b = 1/max_omega_b_error_radps**2
        self.Q = np.diag(np.hstack(Qii_omega_b,np.zeros(4)))

        max_servo_angles_rad = np.deg2rad(data['Servo_max_angles_deg'])
        Rii_phi = np.ones(2) * 1/max_servo_angles_rad ** 2

        max_motor_omega = max(raw_data[:,0])
        Rii_omega_r = np.ones(2) * 1/max_motor_omega ** 2

        # phi_px_cmd, phi_nx_cmd, omega_rpx_cmd, omega_rnx_cmd
        self.R = np.diag(np.hstack(Rii_phi,Rii_omega_r))
        self.R_inv = np.inv(self.R)

    def get_u(self,full_state_est,w_des_radps):

        # (omega_bx, omega_by, omega_bz, phi_px, phi_nx, omega_rpx, omega_rnx)
        # build desired state vector & compute error
        full_state_des = np.vstack((w_des_radps,np.zeros(4)))
        full_state_error = full_state_est - full_state_des

        # obtain thrusts from LUTs
        f_px = self.w_rapds_2_f_N.index(np.abs(full_state_est[5]))
        f_nx = self.w_rapds_2_f_N.index(np.abs(full_state_est[6]))

        eval_args = tuple(full_state_est) + (f_px, f_nx) + self.params

        # Compute A and B
        A = self.A_func(*eval_args)
        B = self.B_func(*eval_args)

        # solve continuous ARE
        P = solve_continuous_are(A, B, self.Q, self.R)
    
        # compute gain
        K =  self.R_inv @ B.T @ P
        
        # compute U
        return -K @ full_state_error