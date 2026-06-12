import numpy as np
import yaml
from util import *
from gnc.state_based_optimal_controller import sboc
from models.tf import tf1
from gnc.nav import nav

def compute_T_hat(phi_rad):
    return np.array([[0],[np.sin(phi_rad)],[-np.cos(phi_rad)]])

class error_integrator:
    def __init__(self, ki, i_lims, dt_s):
        self.ki = np.array(ki)
        self.i_lims = np.array(i_lims) # expects shape (2,) for global or (3, 2) for axial
        self.dt_s = dt_s
        self.i_err = np.zeros(3)
        
    def step(self, err):
        self.i_err += err * self.dt_s
        if self.i_lims.ndim == 1:
            self.i_err = np.clip(self.i_err, self.i_lims[0], self.i_lims[1])
        else:
            for i in range(3):
                self.i_err[i] = np.clip(self.i_err[i], self.i_lims[i,0], self.i_lims[i,1])
        return self.ki * self.i_err

class gnc:
    def __init__(self,ctrl_yaml_path,nav_yaml_path,dt_s):

        # build state-based optimal controller
        self.ctl = sboc(ctrl_yaml_path)

        # obtain max force from motor lookup
        self.f_max_N = np.max(self.ctl.w_rapds_2_f_N.y)

        # read actuator time constants for Kalman filtering - Use YAML values EXACTLY
        with open(ctrl_yaml_path, 'r') as file:
            ctrl_data = yaml.safe_load(file)
        servo_tau_s = ms2s(ctrl_data['servo_time_constant_ms'])
        motor_tau_s = ms2s(ctrl_data['motor_time_constant_ms'])

        # build error integrator
        ki = ctrl_data['integral_gain']
        i_limit_raw = ctrl_data['integral_limit']
        i_lims = np.array([-1,1]) * abs(i_limit_raw)            
        self.integrator = error_integrator(ki, i_lims, dt_s)

        # build navigation object
        self.nav = nav(nav_yaml_path, dt_s, servo_tau_s, motor_tau_s)

        # store previous command for Kalman filter predict step and A/B formulation
        self.prev_u_nav = np.zeros(4)
        
    def step(self,w_des_radps,w_est_radps,act_est,thr_frac):

        # estimate state using nav object
        w_est_filtered, act_est_filtered = self.nav.estimate(w_est_radps, act_est, self.prev_u_nav)
        w_est_flat = w_est_filtered.flatten()

        # integrate error
        err = w_des_radps - w_est_flat
        w_des_modified = w_des_radps + self.integrator.step(err)

        # build full state estimate
        full_state_est = np.hstack((w_est_flat, act_est_filtered))

        # obtain actuator effort from SDRE
        u = self.ctl.get_u(full_state_est,w_des_modified,thr_frac)

        # update previous command for next step
        self.prev_u_nav = np.array([u[0], u[1], np.abs(u[2]), np.abs(u[3])])

        # convert motor omegas to throttle fractions
        f_px = self.ctl.w_rapds_2_f_N.index(np.abs(u[2]))
        f_nx = self.ctl.w_rapds_2_f_N.index(np.abs(u[3]))
        thr_px_frac = f_px / self.f_max_N
        thr_nx_frac = f_nx / self.f_max_N

        # return actuator effort
        return np.array([thr_px_frac, thr_nx_frac, u[0], u[1]])


