import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from util import rad2deg, deg2rad

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
        # 1. Angular Rates (Commanded vs. Measured vs. Filtered) - in deg/s
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        rates = [('x (Roll)', 'wx_cmd_radps', 'wx_meas_radps', 'wx_filt_radps'), 
                 ('y (Pitch)', 'wy_cmd_radps', 'wy_meas_radps', 'wy_filt_radps'), 
                 ('z (Yaw)', 'wz_cmd_radps', 'wz_meas_radps', 'wz_filt_radps')]
        
        for i, (axis, cmd, meas, filt) in enumerate(rates):
            axes[i].plot(t, rad2deg(data[cmd]), '--', color='lightgray', label=f'Cmd {axis}', zorder=0)
            axes[i].plot(t, rad2deg(data[meas]), color='red', alpha=0.5, label=f'Meas {axis}', zorder=1)
            if filt in data.dtype.names:
                axes[i].plot(t, rad2deg(data[filt]), color='blue', label=f'Filt {axis}', zorder=2)
            axes[i].set_ylabel(f'w_{axis} [deg/s]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        
        axes[0].set_title('Body Angular Rates: Cmd vs. Meas vs. Filt')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Servos: Raw vs. Filtered Estimates
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        servos = [('PX Servo', 'phi_px_est_rad', 'phi_px_filt_rad', 'phi_px_cmd_rad'),
                  ('NX Servo', 'phi_nx_est_rad', 'phi_nx_filt_rad', 'phi_nx_cmd_rad')]
        
        for i, (name, raw, filt, cmd) in enumerate(servos):
            if raw in data.dtype.names:
                axes[i].plot(t, rad2deg(data[cmd]), '--', color='lightgray', label=f'Cmd {name}', zorder=0)
                axes[i].plot(t, rad2deg(data[raw]), color='red', alpha=0.5, label=f'Raw Est {name}', zorder=1)
                axes[i].plot(t, rad2deg(data[filt]), color='blue', label=f'Filt Est {name}', zorder=2)
                axes[i].set_ylabel(f'{name} [deg]')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)

        axes[0].set_title('Servo Estimates: Raw vs. Filtered')
        axes[1].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Motors: Raw vs. Filtered Estimates
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        motors = [('PX Motor', 'w_px_est_radps', 'w_px_filt_radps'),
                  ('NX Motor', 'w_nx_est_radps', 'w_nx_filt_radps')]
        
        for i, (name, raw, filt) in enumerate(motors):
            if raw in data.dtype.names:
                axes[i].plot(t, data[raw], color='red', alpha=0.5, label=f'Raw Est {name}', zorder=1)
                axes[i].plot(t, data[filt], color='blue', label=f'Filt Est {name}', zorder=2)
                axes[i].set_ylabel(f'{name} [rad/s]')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)

        axes[0].set_title('Motor Estimates: Raw vs. Filtered')
        axes[1].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Servos: Angles and Rates on one page
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        # Angles (deg)
        axes[0].plot(t, rad2deg(data['phi_px_cmd_rad']), '--', color='lightgray', label='PX Cmd', zorder=0)
        axes[0].plot(t, rad2deg(data['phi_nx_cmd_rad']), '--', color='lightgray', label='NX Cmd', zorder=0)
        axes[0].plot(t, rad2deg(data['phi_px_rad']), label='PX Act', zorder=1)
        axes[0].plot(t, rad2deg(data['phi_nx_rad']), label='NX Act', zorder=2)
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

        # 5. Motors: Speeds and Accelerations on one page
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

        # 6. Control Efforts (Throttle)
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
        # Throttle
        if 'thr_cmd_frac' in data.dtype.names:
            ax.plot(t, data['thr_cmd_frac'], '--', color='lightgray', label='Cmd Throttle', zorder=0)
        ax.plot(t, data['fpx_frac'], label='PX Throttle', zorder=1)
        ax.plot(t, data['fnx_frac'], label='NX Throttle', zorder=2)
        ax.set_ylabel('Throttle Fraction')
        ax.set_title('Control Efforts (Throttle)')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_xlabel('Time [s]')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 7. True vs. Measured Rates
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        rates_meas = [('x (Roll)', 'wx_radps', 'wx_meas_radps'), 
                      ('y (Pitch)', 'wy_radps', 'wy_meas_radps'), 
                      ('z (Yaw)', 'wz_radps', 'wz_meas_radps')]
        
        for i, (axis, true, meas) in enumerate(rates_meas):
            axes[i].plot(t, rad2deg(data[true]), label=f'True {axis}')
            axes[i].plot(t, rad2deg(data[meas]), label=f'Meas {axis}', alpha=0.7)
            axes[i].set_ylabel(f'w_{axis} [deg/s]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
        
        axes[0].set_title('Body Angular Rates: True vs. Measured')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 8. Net Plant Moments
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        moments = [('x (Roll)', 'M_net_x_Nm'), 
                   ('y (Pitch)', 'M_net_y_Nm'), 
                   ('z (Yaw)', 'M_net_z_Nm')]
        
        for i, (axis, moment_col) in enumerate(moments):
            if moment_col in data.dtype.names:
                axes[i].plot(t, data[moment_col], label=f'Net M_{axis}')
                axes[i].set_ylabel(f'M_{axis} [Nm]')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)
        
        axes[0].set_title('Net Plant Moments')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 9. GNC Integrator State
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        integrators = [('x (Roll)', 'i_err_x'), 
                       ('y (Pitch)', 'i_err_y'), 
                       ('z (Yaw)', 'i_err_z')]
        
        for i, (axis, col) in enumerate(integrators):
            if col in data.dtype.names:
                axes[i].plot(t, data[col], label=f'Int Err {axis}', color='blue')
                axes[i].set_ylabel(f'Accum Error {axis}')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)
        
        axes[0].set_title('GNC Integrator State (Accumulated Error)')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated {output_pdf}")

if __name__ == "__main__":
    generate_smart_pdf_plots("log.csv")
