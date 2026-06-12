import numpy as np
from util import *
from config.config_util import load_gnc_config

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
    def __init__(self,gnc_config: gnc_config):
        self.gnc_config = gnc_config
        
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


