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

    with PdfPages(output_pdf) as pdf:
        # 1. Angular Rates (Commanded vs. Actual)
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        rates = [('x', 'wx_cmd_radps', 'wx_radps'), 
                 ('y', 'wy_cmd_radps', 'wy_radps'), 
                 ('z', 'wz_cmd_radps', 'wz_radps')]
        
        for i, (axis, cmd, act) in enumerate(rates):
            axes[i].plot(t, data[cmd], '--', label=f'Cmd {axis}')
            axes[i].plot(t, data[act], label=f'Act {axis}')
            axes[i].set_ylabel(f'w_{axis} [rad/s]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        
        axes[0].set_title('Angular Rates: Commanded vs. Actual')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Servos (Angles and Rates)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Angles
        axes[0].plot(t, data['phi_px_cmd_rad'], '--', label='Phi PX Cmd')
        axes[0].plot(t, data['phi_px_rad'], label='Phi PX Act')
        axes[0].plot(t, data['phi_nx_cmd_rad'], '--', label='Phi NX Cmd')
        axes[0].plot(t, data['phi_nx_rad'], label='Phi NX Act')
        axes[0].set_ylabel('Servo Angle [rad]')
        axes[0].set_title('Servo Dynamics')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Rates
        axes[1].plot(t, data['phidot_px_radps'], label='PhiDot PX')
        axes[1].plot(t, data['phidot_nx_radps'], label='PhiDot NX')
        axes[1].set_ylabel('Servo Rate [rad/s]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        axes[1].set_xlabel('Time [s]')
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Motors (Throttle and Speeds)
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Fractions
        axes[0].plot(t, data['fpx_frac'], label='PX Throttle')
        axes[0].plot(t, data['fnx_frac'], label='NX Throttle')
        axes[0].set_ylabel('Throttle Fraction')
        axes[0].set_title('Motor Performance')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # Speeds (Note: header in sim_cl had a typo "w_px_radps,w_nx_radps,wdot_px_radps2,wdot_nx_radps" as one string)
        # Looking at log.csv headers from previous check: 
        # t_s,wx_radps,wy_radps,wz_radps,phi_px_rad,phi_nx_rad,phidot_px_radps,phidot_nx_radps,
        # w_px_radps,w_nx_radps,wdot_px_radps2,wdot_nx_radps,F_px_x_N,F_px_y_N,F_px_z_N,F_nx_x_N,F_nx_y_N,F_nx_z_N,
        # wx_cmd_radps,wy_cmd_radps,wz_cmd_radps,Mx_cmd_Nm,My_cmd_Nm,Mz_cmd_Nm,fpx_frac,fnx_frac,phi_px_cmd_rad,phi_nx_cmd_rad
        
        axes[1].plot(t, data['w_px_radps'], label='Omega PX')
        axes[1].plot(t, data['w_nx_radps'], label='Omega NX')
        axes[1].set_ylabel('Motor Speed [rad/s]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        axes[1].set_xlabel('Time [s]')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Commanded Moments
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 5))
        axes.plot(t, data['Mx_cmd_Nm'], label='Mx Cmd')
        axes.plot(t, data['My_cmd_Nm'], label='My Cmd')
        axes.plot(t, data['Mz_cmd_Nm'], label='Mz Cmd')
        axes.set_ylabel('Moment [Nm]')
        axes.set_title('Commanded Control Moments')
        axes.set_xlabel('Time [s]')
        axes.legend(loc='upper right')
        axes.grid(True)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated {output_pdf}")

if __name__ == "__main__":
    generate_smart_pdf_plots("log.csv")
