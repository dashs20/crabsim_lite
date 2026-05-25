import numpy as np
import dill
import yaml
from util import *
from scipy.linalg import solve_continuous_are, inv

class sboc:
    def __init__(self,sboc_yaml):

        # Load vehicle configuration data
        with open(sboc_yaml, 'r') as file:
            data = yaml.safe_load(file)

        # load A(x) and B(x)
        with open(data['pickle_path'], 'rb') as f:
            plant = dill.load(f)

        self.A_func = plant['A_func']
        self.B_func = plant['B_func']
        self.M_net_func = plant['M_net_func']

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
        tau_s = ms2s(data['servo_time_constant_ms'])
        tau_r = ms2s(data['motor_time_constant_ms'])
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
        self.f_max_N = max(raw_data[:,0])

        # build Q and R matrices for CARE
        max_omega_b_error_radps = np.array(data['Q_max_acceptable_error_radps'])
        Qii_omega_b = 1/max_omega_b_error_radps**2
        self.Q = np.diag(np.hstack((Qii_omega_b,np.zeros(4))))

        self.max_servo_angles_rad = np.deg2rad(data['Servo_max_angles_deg'])
        Rii_phi = np.ones(2) * 1/self.max_servo_angles_rad ** 2

        self.max_motor_omega = max(raw_data[:,1])
        Rii_omega_r = np.ones(2) * 1/self.max_motor_omega ** 2

        # phi_px_cmd, phi_nx_cmd, omega_rpx_cmd, omega_rnx_cmd
        self.R = np.diag(np.hstack((Rii_phi,Rii_omega_r)))
        self.R_inv = np.linalg.inv(self.R)

    def get_u(self,full_state_est,w_des_radps,thr_frac):

        # (omega_bx, omega_by, omega_bz, phi_px, phi_nx, omega_rpx, omega_rnx)
        # build desired state vector & compute error

        f_des_N = thr_frac * self.f_max_N
        omega_px_des_radps =  self.w_rapds_2_f_N.index_y(f_des_N)
        omega_nx_des_radps = -self.w_rapds_2_f_N.index_y(f_des_N)

        desired_actuator_state = np.array([0,0,omega_px_des_radps,omega_nx_des_radps])

        full_state_des = np.hstack((w_des_radps.flatten(),desired_actuator_state))
        full_state_error = full_state_est - full_state_des

        # obtain thrusts from LUTs
        f_px = self.w_rapds_2_f_N.index(np.abs(full_state_est[5]))
        f_nx = self.w_rapds_2_f_N.index(np.abs(full_state_est[6]))

        # Safe division for thrusts
        w_px = full_state_est[5]
        w_nx = full_state_est[6]
        f_px_div_w = f_px / w_px if abs(w_px) > 1e-6 else 0.0
        f_nx_div_w = f_nx / w_nx if abs(w_nx) > 1e-6 else 0.0

        # Safe trigonometric limits
        phi_px = full_state_est[3]
        phi_nx = full_state_est[4]
        
        sinc_px = np.sin(phi_px) / phi_px if abs(phi_px) > 1e-6 else 1.0
        sinc_nx = np.sin(phi_nx) / phi_nx if abs(phi_nx) > 1e-6 else 1.0
        
        cosc_px = (np.cos(phi_px) - 1.0) / phi_px if abs(phi_px) > 1e-6 else 0.0
        cosc_nx = (np.cos(phi_nx) - 1.0) / phi_nx if abs(phi_nx) > 1e-6 else 0.0

        eval_args = tuple(full_state_est) + (f_px_div_w, f_nx_div_w, sinc_px, sinc_nx, cosc_px, cosc_nx) + self.params

        # Compute A and B
        A = self.A_func(*eval_args)
        B = self.B_func(*eval_args)

        # Scale Q and R to improve the condition number of the Hamiltonian matrix
        scale = 1e9

        # solve continuous ARE
        try:
            P = solve_continuous_are(A, B, self.Q * scale, self.R * scale)
        
            # compute gain
            # K = R_inv @ B.T @ P. But R_inv here must be the inverse of the SCALED R!
            # Alternatively, since P is scaled by 'scale', and we use unscaled R_inv:
            # K = self.R_inv @ B.T @ (P / scale)
            K = self.R_inv @ B.T @ (P / scale)
            
            # compute U
            # [phi_px, phi_nx, omega_rpx, omega_rnx]
            u = -K @ full_state_error

            # clip if it's too high
            phi_cmd_clipped = np.clip(u[0:2],-self.max_servo_angles_rad,self.max_servo_angles_rad)
            omega_cmd_clipped = np.clip(u[2:4],-self.max_motor_omega,self.max_motor_omega)
            u = np.hstack((phi_cmd_clipped,omega_cmd_clipped))
        except:
            u = np.zeros(4)

        return u