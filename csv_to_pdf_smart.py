import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_smart_pdf_plots(input_csv):
    base_path, _ = os.path.splitext(input_csv)
    output_pdf = f"{base_path}_smart.pdf"

    try:
        data = np.genfromtxt(input_csv, delimiter=',', names=True, deletechars="")
        if data.dtype.names is None:
            raise ValueError("No headers found in CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    t = data['t_s']

    def rad2deg(val):
        return val * 180 / np.pi

    with PdfPages(output_pdf) as pdf:
        # 1. Angular Rates (Commanded vs. Actual) - in deg/s
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        rates = [('x (Roll)', 'wx_cmd_radps', 'wx_radps'), 
                 ('y (Pitch)', 'wy_cmd_radps', 'wy_radps'), 
                 ('z (Yaw)', 'wz_cmd_radps', 'wz_radps')]
        
        for i, (axis, cmd, act) in enumerate(rates):
            axes[i].plot(t, rad2deg(data[cmd]), '--', label=f'Cmd {axis}')
            axes[i].plot(t, rad2deg(data[act]), label=f'Act {axis}')
            axes[i].set_ylabel(f'w_{axis} [deg/s]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        
        axes[0].set_title('Body Angular Rates: Commanded vs. Actual')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Servos: Angles and Rates on one page
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Angles (deg)
        axes[0].plot(t, rad2deg(data['phi_px_cmd_rad']), '--', label='PX Cmd')
        axes[0].plot(t, rad2deg(data['phi_px_rad']), label='PX Act')
        axes[0].plot(t, rad2deg(data['phi_nx_cmd_rad']), '--', label='NX Cmd')
        axes[0].plot(t, rad2deg(data['phi_nx_rad']), label='NX Act')
        axes[0].set_ylabel('Servo Angle [deg]')
        axes[0].set_title('Servo Dynamics (Angles & Rates)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Rates (deg/s)
        axes[1].plot(t, rad2deg(data['phidot_px_radps']), label='PX Rate')
        axes[1].plot(t, rad2deg(data['phidot_nx_radps']), label='NX Rate')
        axes[1].set_ylabel('Servo Rate [deg/s]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        axes[1].set_xlabel('Time [s]')
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Motors: Speeds and Accelerations on one page
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Speeds (rad/s)
        axes[0].plot(t, data['w_px_radps'], label='PX Speed')
        axes[0].plot(t, np.abs(data['w_nx_radps']), label='NX Speed (abs)')
        axes[0].set_ylabel('Motor Speed [rad/s]')
        axes[0].set_title('Motor Dynamics (Speeds & Accels)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Accelerations (rad/s^2)
        axes[1].plot(t, data['wdot_px_radps2'], label='PX Accel')
        axes[1].plot(t, np.abs(data['wdot_nx_radps2']), label='NX Accel (abs)')
        axes[1].set_ylabel('Motor Accel [rad/s^2]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        axes[1].set_xlabel('Time [s]')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Control Efforts (Throttle & Moments)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Throttle
        axes[0].plot(t, data['fpx_frac'], label='PX Throttle')
        axes[0].plot(t, data['fnx_frac'], label='NX Throttle')
        axes[0].set_ylabel('Throttle Fraction')
        axes[0].set_title('Control Efforts')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Moments
        axes[1].plot(t, data['Mx_cmd_Nm'], label='Mx Cmd')
        axes[1].plot(t, data['My_cmd_Nm'], label='My Cmd')
        axes[1].plot(t, data['Mz_cmd_Nm'], label='Mz Cmd')
        axes[1].set_ylabel('Moment [Nm]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        axes[1].set_xlabel('Time [s]')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated {output_pdf}")

if __name__ == "__main__":
    generate_smart_pdf_plots("log.csv")
