import numpy as np
import yaml

class lpf:
    def __init__(self, fc_hz, dt_s):
        self.fc_hz = fc_hz
        self.dt_s = dt_s
        self.alpha = 2 * np.pi * self.fc_hz * self.dt_s / (1 + 2 * np.pi * self.fc_hz * self.dt_s)
        self.y_prev = None

    def step(self, u):
        if self.y_prev is None:
            self.y_prev = u
        else:
            self.y_prev = self.alpha * u + (1 - self.alpha) * self.y_prev
        return self.y_prev

class kalman_tf1:
    def __init__(self, tau, dt_s, Q, R):
        self.tau = tau
        self.dt_s = dt_s
        self.Q = Q
        self.R = R
        
        # Euler discretization of 1/(tau*s + 1)
        self.Ad = 1.0 - (self.dt_s / self.tau)
        self.Bd = self.dt_s / self.tau
        
        self.x = None
        self.P = 1.0

    def filter(self, u, z):
        if self.x is None:
            self.x = z
            return self.x

        # Predict
        x_pred = self.Ad * self.x + self.Bd * u
        P_pred = self.Ad * self.P * self.Ad + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        
        return self.x

class nav:
    def __init__(self, yaml_path, dt_s, servo_tau_s, motor_tau_s):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # Use YAML values directly and exactly
        fc_body = data['filter_cutoff_hz_body_rates']
        
        # LPF for body rates
        self.lpf_body_rates = lpf(fc_body, dt_s)
        
        # Kalman Filters for actuators
        q_servo = data['kf_q_servo']
        r_servo = data['kf_r_servo']
        q_motor = data['kf_q_motor']
        r_motor = data['kf_r_motor']
        
        self.kf_phi_px = kalman_tf1(servo_tau_s, dt_s, q_servo, r_servo)
        self.kf_phi_nx = kalman_tf1(servo_tau_s, dt_s, q_servo, r_servo)
        self.kf_w_px = kalman_tf1(motor_tau_s, dt_s, q_motor, r_motor)
        self.kf_w_nx = kalman_tf1(motor_tau_s, dt_s, q_motor, r_motor)

        self.w_est_filtered = None
        self.act_est_filtered = None

    def estimate(self, w_meas_radps, act_meas, prev_act_cmd):
        self.w_est_filtered = self.lpf_body_rates.step(w_meas_radps)
        
        # act_meas = [phi_px, phi_nx, w_px, w_nx]
        # prev_act_cmd = [phi_px_cmd, phi_nx_cmd, w_px_cmd, w_nx_cmd]
        
        phi_px_filt = self.kf_phi_px.filter(prev_act_cmd[0], act_meas[0])
        phi_nx_filt = self.kf_phi_nx.filter(prev_act_cmd[1], act_meas[1])
        w_px_filt = self.kf_w_px.filter(prev_act_cmd[2], act_meas[2])
        w_nx_filt = self.kf_w_nx.filter(prev_act_cmd[3], act_meas[3])
        
        self.act_est_filtered = np.array([phi_px_filt, phi_nx_filt, w_px_filt, w_nx_filt])
        
        return self.w_est_filtered, self.act_est_filtered
