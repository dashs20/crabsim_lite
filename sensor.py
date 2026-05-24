import numpy as np

class ICM42688:
    def __init__(self, fsr, bandwidth_hz):
        self.fsr = fsr
        self.bandwidth_hz = bandwidth_hz
        
        # Hardware constant from datasheet: 2.8 mdps/sqrt(Hz)
        self.noise_density = 0.0028 
        
        # Pre-compute the 1-sigma noise and ADC scale factor during initialization
        self.sigma = self.noise_density * np.sqrt(self.bandwidth_hz)
        self.lsb_per_dps = 32768 / self.fsr

    def get_measurement(self, w_true_radps: float | np.ndarray) -> float | np.ndarray:

        # Convert input rad/s to degrees/s since sensor parameters are in dps
        w_true_dps = w_true_radps * (180.0 / np.pi)

        # 1. Generate wideband Gaussian noise
        noise = np.random.normal(0, self.sigma, size=w_true_radps.shape)
            
        # 2. Inject noise into the true signal
        w_meas_dps = w_true_dps + noise
        
        # 3. Apply hardware saturation (clipping to physical FSR limits)
        measured_rate_dps = np.clip(w_meas_dps, -self.fsr, self.fsr)
        
        # 4. Simulate 16-bit ADC quantization
        measured_rate_dps = np.round(measured_rate_dps * self.lsb_per_dps) / self.lsb_per_dps
            
        # Convert back to rad/s for output
        measured_rate = measured_rate_dps * (np.pi / 180.0)
            
        return measured_rate