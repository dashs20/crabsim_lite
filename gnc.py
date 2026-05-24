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
        act_2_M = np.array([[-rpx_z,0,-rnx_z,0],
                            [0,-rpx_x,0,-rnx_x],
                            [rpx_x,0,rnx_x,0],
                            [0,1,0,1]])
        
        # take inverse (only need to do this once! More performant)
        self.M_2_act = np.linalg.inv(act_2_M)

        # force to rotor rate lookup
        raw_data = np.loadtxt(motor_data, delimiter=',', skiprows=1)
        self.f_max_N = np.max(raw_data[:,0]) # max thrust of a single motor
        self.min_thr_frac = min_thr_frac # minimum throttle the vehicle is allowed to operate at

        # max allowable servo angle
        self.phi_max_rad = phi_max_rad

    def allocate(self,M_desired_Nm,thr_frac):

        # clip throttle
        thr_frac = np.min(thr_frac,self.min_thr_frac)

        # build desired actuator "ask"
        Fz_desired_N = thr_frac * -2 * self.f_max_N # 2 thrusters, -z is up
        ask = np.vstack((M_desired_Nm,
                         Fz_desired_N))

        # obtain actuator forces
        f = self.M_2_act @ ask

        # contents of f:
        # [0]: fpx_y
        # [1]: fpx_z
        # [2]: fnx_y
        # [3]: fnx_z

        # obtain servo angles and clip if too large
        phi_px_rad = np.clip(np.atan2(f[1],-f[0]),-self.phi_max_rad,self.phi_max_rad)
        phi_nx_rad = np.clip(np.atan2(f[2],-f[1]),-self.phi_max_rad,self.phi_max_rad)

        # obtain throttle fractions for motors and clip if too large
        fpx_frac = np.clip(np.linalg.norm(f[0:3])/self.f_max_N,0,1)
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

            Mx_max_Nm = self.M_2_act @ np.array([[Fpx[1]],
                                                 [Fpx[2]],
                                                 [Fnx[1]],
                                                 [Fnx[2]]])
        
        def get_max_y(self,thr_frac):
            Fpx = compute_F_N(self.f_max_N*thr_frac,0)
            Fnx = np.zeros((3,1))

            My_max_Nm = self.M_2_act @ np.array([[Fpx[1]],
                                                 [Fpx[2]],
                                                 [Fnx[1]],
                                                 [Fnx[2]]])
        
        def get_max_z(self,thr_frac):
            Fpx = compute_F_N(self.f_max_N*thr_frac,self.phi_max_rad)
            Fnx = compute_F_N(self.f_max_N*thr_frac,-self.phi_max_rad)

            Mz_max_Nm = self.M_2_act @ np.array([[Fpx[1]],
                                                [Fpx[2]],
                                                [Fnx[1]],
                                                [Fnx[2]]])
        
        # perform a sweep to generate max moments as a function of throttle
        n_checks = 100
        thr_fracs = np.linspace(0,1,n_checks)

        Mx_max_Nm = np.zeros(np.size(thr_fracs))
        My_max_Nm = np.zeros(np.size(thr_fracs))
        Mz_max_Nm = np.zeros(np.size(thr_fracs))

        for i in range(n_checks):
            Mx_max_Nm[i] = get_max_x(thr_fracs[i])
            My_max_Nm[i] = get_max_y(thr_fracs[i])
            Mz_max_Nm[i] = get_max_z(thr_fracs[i])

        # build lookups
        thr_frac_2_max_Mx = lookup_1D(thr_fracs,Mx_max_Nm)
        thr_frac_2_max_My = lookup_1D(thr_fracs,My_max_Nm)
        thr_frac_2_max_Mz = lookup_1D(thr_fracs,Mz_max_Nm)

        return thr_frac_2_max_Mx, thr_frac_2_max_My, thr_frac_2_max_Mz

class GNC:
    def __init__(self,yaml_path):
        # open yaml
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # build allocator
        self.allocator = allocator(mm2m(data['r_cg2px_mm']),
                                   mm2m(data['r_cg2nx_mm']),
                                   data['motor_lookup'],
                                   data['min_thr_frac'],
                                   np.deg2rad(data['phi_max_deg']))