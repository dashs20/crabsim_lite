from models.plants import *
from util.util import *
import matplotlib.pyplot as plt
import numpy as np
import random

# define sim dt (seconds)
dt_s = 1/10000
t_end_s = 1
n_steps = round(t_end_s/dt_s)

def run_sim(thrust_on):
    crabcopter = bicopter('crabcopter.yaml', dt_s)
    
    # Enable or disable the thrust moments natively
    crabcopter.thrust_enabled = thrust_on

    t_history = np.zeros(n_steps)
    L_total_mag_history = np.zeros(n_steps)
    phi_px_history = np.zeros(n_steps)
    phi_nx_history = np.zeros(n_steps)
    w_px_history = np.zeros(n_steps)
    w_nx_history = np.zeros(n_steps)
    wx_history = np.zeros(n_steps)
    wy_history = np.zeros(n_steps)
    wz_history = np.zeros(n_steps)
    
    # Set seed to ensure the commands are identical between both runs
    random.seed(42)
    np.random.seed(42)
    
    t_s = 0
    for i_step in range(n_steps):

        # command sine waves to actuators
        # servos: +/- 45 deg
        phi_px_cmd_deg = 45 * np.sin(2 * np.pi * 0.5 * t_s)
        phi_nx_cmd_deg = 45 * np.sin(2 * np.pi * 0.7 * t_s + 1.0)
        
        # throttle fractions: 0.2 to 0.8
        thr_px_cmd_frac = 0.005 + 0.003 * np.sin(2 * np.pi * 0.4 * t_s + 0.5)
        thr_nx_cmd_frac = 0.005 + 0.003 * np.sin(2 * np.pi * 0.6 * t_s + 1.5)
        
        act_cmd = [thr_px_cmd_frac, thr_nx_cmd_frac, np.radians(phi_px_cmd_deg), np.radians(phi_nx_cmd_deg)]

        # step plant
        state = crabcopter.step(act_cmd)
        
        # Extract state from row vector
        W_B_wrt_I_radps = state[0, 0:3].reshape(3,1)
        phi_tilter_px_rad = state[0, 3]
        phi_tilter_nx_rad = state[0, 4]
        w_rotor_px_radps = state[0, 7]
        w_rotor_nx_radps = state[0, 8]
        
        # Calculate Body Angular Momentum
        L_body = crabcopter.body_I_kgm2 @ W_B_wrt_I_radps
        
        # Calculate Rotor Angular Momentum
        T_hat_px = compute_T_hat(phi_tilter_px_rad)
        L_rotor_px = crabcopter.rotor_j_kgm2 * w_rotor_px_radps * T_hat_px
        
        T_hat_nx = compute_T_hat(phi_tilter_nx_rad)
        L_rotor_nx = crabcopter.rotor_j_kgm2 * w_rotor_nx_radps * T_hat_nx
        
        # Total Angular Momentum
        L_total = L_body + L_rotor_px + L_rotor_nx
        
        t_history[i_step] = t_s
        L_total_mag_history[i_step] = np.linalg.norm(L_total)
        phi_px_history[i_step] = phi_px_cmd_deg
        phi_nx_history[i_step] = phi_nx_cmd_deg
        w_px_history[i_step] = thr_px_cmd_frac
        w_nx_history[i_step] = thr_nx_cmd_frac
        wx_history[i_step] = W_B_wrt_I_radps[0, 0]
        wy_history[i_step] = W_B_wrt_I_radps[1, 0]
        wz_history[i_step] = W_B_wrt_I_radps[2, 0]

        # update time
        t_s += dt_s
        
    return t_history, L_total_mag_history, phi_px_history, phi_nx_history, w_px_history, w_nx_history, wx_history, wy_history, wz_history

# Run the simulations
print("Running with thrust ON...")
t_history, L_mag_on, phi_px, phi_nx, thr_px, thr_nx, wx_on, wy_on, wz_on = run_sim(thrust_on=True)

print("Running with thrust OFF...")
_, L_mag_off, _, _, _, _, wx_off, wy_off, wz_off = run_sim(thrust_on=False)

# Plotting
plt.figure(figsize=(10, 6))

# Plot 1: Angular Momentum
plt.plot(t_history, L_mag_on, label='Thrust ON')
plt.plot(t_history, L_mag_off, label='Thrust OFF')
plt.title('Magnitude of Total Angular Momentum During Tumble')
plt.xlabel('Time (s)')
plt.ylabel('L Mag (kg*m^2/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('tumble_angular_momentum.png')
print("Simulations complete. Plot saved as 'tumble_angular_momentum.png'.")
