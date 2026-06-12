import numpy as np
from util import rad2deg, deg2rad

class ICM42688:
    def __init__(self, fsr, bandwidth_hz, dt_s):
        self.fsr = fsr
        self.bandwidth_hz = bandwidth_hz
        self.dt_s = dt_s
        
        # Hardware constant from datasheet: 2.8 mdps/sqrt(Hz)
        self.noise_density = 0.0028 
        
        # Standard deviation to add per step to match target noise density
        # sigma = Density * sqrt(fs / 2) = Density * sqrt(1 / (2 * dt))
        self.sigma = self.noise_density * np.sqrt(1.0 / (2.0 * self.dt_s))
        
        # Internal low-pass filter to match sensor bandwidth
        # We'll use a simple first-order discrete LPF
        self.alpha = 2 * np.pi * self.bandwidth_hz * self.dt_s / (1 + 2 * np.pi * self.bandwidth_hz * self.dt_s)
        self.w_meas_prev = None
        
        self.lsb_per_dps = 32768 / self.fsr

    def get_measurement(self, w_true_radps: float | np.ndarray) -> float | np.ndarray:

        # Convert input rad/s to degrees/s since sensor parameters are in dps
        w_true_dps = rad2deg(w_true_radps)

        # 1. Generate wideband Gaussian noise with correct spectral density
        noise = np.random.normal(0, self.sigma, size=w_true_radps.shape)
            
        # 2. Inject noise into the true signal
        w_raw_dps = w_true_dps + noise
        
        # 3. Apply internal sensor bandwidth filter
        if self.w_meas_prev is None:
            self.w_meas_prev = w_raw_dps
        else:
            self.w_meas_prev = self.alpha * w_raw_dps + (1 - self.alpha) * self.w_meas_prev
        
        # 4. Apply hardware saturation (clipping to physical FSR limits)
        measured_rate_dps = np.clip(self.w_meas_prev, -self.fsr, self.fsr)
        
        # 5. Simulate 16-bit ADC quantization
        measured_rate_dps = np.round(measured_rate_dps * self.lsb_per_dps) / self.lsb_per_dps
            
        # Convert back to rad/s for output
        measured_rate = deg2rad(measured_rate_dps)
            
        return measured_rate