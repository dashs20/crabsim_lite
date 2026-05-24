import numpy as np
import yaml
from util import *

def compute_T_hat(phi_rad):
    return np.array([[0],[np.sin(phi_rad)],[-np.cos(phi_rad)]])

class allocator:
    def __init__(self,r_cg2px_m,r_cg2nx_m,motor_data,min_thr_frac,phi_max_rad):
        # extract components of geometry
        rpx_x = r_cg2px_m[0]
        rpx_z = r_cg2px_m[2]

        rnx_x = r_cg2nx_m[0]
        rnx_z = r_cg2nx_m[2]

        # build actuator to moment matrix
        self.act_2_M = np.array([[-rpx_z,0,-rnx_z,0],
                                 [0,-rpx_x,0,-rnx_x],
                                 [rpx_x,0,rnx_x,0],
                                 [0,1,0,1]])
        
        # take inverse (only need to do this once! More performant)
        self.M_2_act = np.linalg.inv(self.act_2_M)

        # force to rotor rate lookup
        raw_data = np.loadtxt(motor_data, delimiter=',', skiprows=1)
        self.f_max_N = np.max(raw_data[:,0]) # max thrust of a single motor
        self.min_thr_frac = min_thr_frac # minimum throttle the vehicle is allowed to operate at

        # max allowable servo angle
        self.phi_max_rad = phi_max_rad

    def allocate(self,M_desired_Nm,thr_frac):

        # clip throttle
        thr_frac = max(thr_frac,self.min_thr_frac)

        # build desired actuator "ask"
        Fz_desired_N = thr_frac * -2 * self.f_max_N # 2 thrusters, -z is up
        ask = np.hstack((M_desired_Nm,
                         Fz_desired_N))

        # obtain actuator forces
        f = self.M_2_act @ np.transpose(ask)

        # contents of f:
        # [0]: fpx_y
        # [1]: fpx_z
        # [2]: fnx_y
        # [3]: fnx_z

        # obtain servo angles and clip if too large
        phi_px_rad = np.clip(np.atan2(f[0],-f[1]),-self.phi_max_rad,self.phi_max_rad)
        phi_nx_rad = np.clip(np.atan2(f[2],-f[3]),-self.phi_max_rad,self.phi_max_rad)

        # obtain throttle fractions for motors and clip if too large
        fpx_frac = np.clip(np.linalg.norm(f[0:2])/self.f_max_N,0,1)
        fnx_frac = np.clip(np.linalg.norm(f[2:4])/self.f_max_N,0,1)

        return [fpx_frac,fnx_frac,phi_px_rad,phi_nx_rad]
    
    # This function generates lookup functions for max moment authority as a function of throttle.
    # For certain estimators that leverage control input as part of the estimation, controller-plant
    # parity is crucial. Without these tables, the controller can freely command moments the plant
    # cannot achieve.
    def compute_max_authority(self):

        # local helper functions
        def compute_F_N(f_N,phi_rad):
            T_hat = compute_T_hat(phi_rad)
            return T_hat * f_N
        
        def get_max_x(self,thr_frac):
            Fpx = compute_F_N(self.f_max_N*thr_frac,self.phi_max_rad)
            Fnx = Fpx

            Mx_max_Nm = self.act_2_M @ np.array([Fpx[1],
                                                 Fpx[2],
                                                 Fnx[1],
                                                 Fnx[2]])
            return Mx_max_Nm[0,0]
        
        def get_max_y(self,thr_frac):
            Fpx = compute_F_N(self.f_max_N*thr_frac,0)
            Fnx = np.zeros((3,1))

            My_max_Nm = self.act_2_M @ np.array([Fpx[1],
                                                 Fpx[2],
                                                 Fnx[1],
                                                 Fnx[2]])
            return My_max_Nm[1,0]
        
        def get_max_z(self,thr_frac):
            Fpx = compute_F_N(self.f_max_N*thr_frac,self.phi_max_rad)
            Fnx = compute_F_N(self.f_max_N*thr_frac,-self.phi_max_rad)

            Mz_max_Nm = self.act_2_M @ np.array([Fpx[1],
                                                 Fpx[2],
                                                 Fnx[1],
                                                 Fnx[2]])
            return Mz_max_Nm[2,0]
        
        # perform a sweep to generate max moments as a function of throttle
        n_checks = 100
        thr_fracs = np.linspace(0,1,n_checks)

        Mx_max_Nm = np.zeros(np.size(thr_fracs))
        My_max_Nm = np.zeros(np.size(thr_fracs))
        Mz_max_Nm = np.zeros(np.size(thr_fracs))

        for i in range(n_checks):
            Mx_max_Nm[i] = get_max_x(self,thr_fracs[i])
            My_max_Nm[i] = get_max_y(self,thr_fracs[i])
            Mz_max_Nm[i] = get_max_z(self,thr_fracs[i])

        # build lookups
        thr_frac_2_max_Mx = lookup_1D(thr_fracs,Mx_max_Nm)
        thr_frac_2_max_My = lookup_1D(thr_fracs,My_max_Nm)
        thr_frac_2_max_Mz = lookup_1D(thr_fracs,Mz_max_Nm)

        return [thr_frac_2_max_Mx, thr_frac_2_max_My, thr_frac_2_max_Mz]

class PID:
    def __init__(self,gains,i_lims,dt_s):
        self.dt_s = dt_s
        self.kp = gains[0]
        self.ki = gains[1]
        self.kd = gains[2]
        self.i_lims = i_lims
        self.i_err = 0
        self.prev_err = 0

    def get_cmd(self,des_state,cur_state):
        # compute error
        err =  des_state - cur_state

        # compute the derivative of the error
        d_err = (err - self.prev_err) / self.dt_s

        # update the integral of the error
        self.i_err += self.dt_s * err
        self.i_err = np.clip(self.i_err, self.i_lims[0], self.i_lims[1])

        # save previous error
        self.prev_err = err

        # compute output signal
        return self.kp * err + self.ki * self.i_err + self.kd * d_err

class smart_PID:
    def __init__(self,yaml_path,M_max_lookups_Nm,dt_s):
        # open yaml
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        self.dt_s = dt_s
        self.PID_x = PID(data['x_gains'],data['x_i_lim'] * np.array([-1,1]),self.dt_s)
        self.PID_y = PID(data['y_gains'],data['y_i_lim'] * np.array([-1,1]),self.dt_s)
        self.PID_z = PID(data['z_gains'],data['z_i_lim'] * np.array([-1,1]),self.dt_s)

        self.Mx_max_lookup_Nm = M_max_lookups_Nm[0]
        self.My_max_lookup_Nm = M_max_lookups_Nm[1]
        self.Mz_max_lookup_Nm = M_max_lookups_Nm[2]

    def get_M_cmd(self,w_des_radps,w_est_radps,thr_frac):
        Mx_max = self.Mx_max_lookup_Nm.index(thr_frac)
        My_max = self.My_max_lookup_Nm.index(thr_frac)
        Mz_max = self.Mz_max_lookup_Nm.index(thr_frac)

        Mx_cmd = np.clip(self.PID_x.get_cmd(w_des_radps[0],w_est_radps[0,0]),-Mx_max,Mx_max)
        My_cmd = np.clip(self.PID_y.get_cmd(w_des_radps[1],w_est_radps[1,0]),-My_max,My_max)
        Mz_cmd = np.clip(self.PID_z.get_cmd(w_des_radps[2],w_est_radps[2,0]),-Mz_max,Mz_max)

        return np.array([Mx_cmd,My_cmd,Mz_cmd])

class gnc:
    def __init__(self,yaml_path,dt_s):

        self.dt_s = dt_s

        # open yaml
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # build allocator
        self.allocator = allocator(mm2m(np.array(data['r_cg2px_mm'])),
                                   mm2m(np.array(data['r_cg2nx_mm'])),
                                   data['motor_lookup'],
                                   data['min_thr_frac'],
                                   np.deg2rad(data['phi_max_deg']))
        
        # build controller
        self.controller = smart_PID(data['controller_yaml'],
                                    self.allocator.compute_max_authority(),
                                    self.dt_s)
        
    def step(self,w_des_radps,w_est_radps,thr_frac):

        # run core control algorithm
        M_cmd_Nm = self.controller.get_M_cmd(w_des_radps,w_est_radps,thr_frac)
        act_cmd = self.allocator.allocate(M_cmd_Nm,thr_frac)

        # return software state
        state = np.hstack((M_cmd_Nm,
                           act_cmd))
        
        return state


