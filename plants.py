import numpy as np
from integrator import rk4

def compute_T_hat(phi_rad):
    return np.array([[0],[np.sin(phi_rad)],[-np.cos(phi_rad)]])

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
        Tdot_hat = compute_T_hat(phidot_rotor_radps)

        return -self.j_kgm2 * (Tdot_hat * w_rotor_radps + T_hat * wdot_rotor_radps2 + np.cross(T_hat * w_rotor_radps,W_B_wrt_I_radps,axis=0))
    
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
        return np.cross(self.r_cg2r_m,F_N,axis=0)
    
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