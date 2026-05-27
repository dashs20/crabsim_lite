import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from util import rad2deg, deg2rad

def generate_mc_report():
    output_pdf = "mc_report.pdf"
    log_files = glob.glob('mc_logs/log_*.csv')
    
    if not log_files:
        print("No log files found in mc_logs/")
        return

    # Load all data
    datasets = []
    for f in log_files:
        try:
            data = np.genfromtxt(f, delimiter=',', names=True, deletechars="")
            datasets.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not datasets:
        print("No valid datasets loaded.")
        return

    with PdfPages(output_pdf) as pdf:
        
        # Create a colormap for the runs
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(datasets))]

        # --- Time-Series Plots (Overlay) ---
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        rates = [('x (Roll)', 'wx_cmd_radps', 'wx_filt_radps'), 
                 ('y (Pitch)', 'wy_cmd_radps', 'wy_filt_radps'), 
                 ('z (Yaw)', 'wz_cmd_radps', 'wz_filt_radps')]
        
        for i, (axis, cmd, filt) in enumerate(rates):
            # Plot all runs with low alpha and varied colors
            axes[i].set_rasterization_zorder(1)
            
            # Plot command from the first dataset first (bottom layer)
            axes[i].plot(datasets[0]['t_s'], rad2deg(datasets[0][cmd]), '--', color='lightgray', label=f'Cmd {axis}', zorder=0)

            for idx, data in enumerate(datasets):
                # Use filtered if it exists, otherwise fallback to true/act
                col = filt if filt in data.dtype.names else filt.replace('_filt', '')
                axes[i].plot(data['t_s'], rad2deg(data[col]), color=colors[idx], alpha=0.3, zorder=1)
            
            axes[i].set_ylabel(f'w_{axis} [deg/s]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
            
        axes[0].set_title('Monte Carlo: Body Angular Rates Envelope (Filtered)')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # --- Servo Control Effort Overlay ---
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
        servos = [('PX Servo', 'phi_px_cmd_rad', 'phi_px_filt_rad'),
                  ('NX Servo', 'phi_nx_cmd_rad', 'phi_nx_filt_rad')]
        
        for i, (name, cmd, filt) in enumerate(servos):
            axes[i].set_rasterization_zorder(1)
            
            axes[i].plot(datasets[0]['t_s'], rad2deg(datasets[0][cmd]), '--', color='lightgray', label=f'Cmd {name}', zorder=0)

            for idx, data in enumerate(datasets):
                col = filt if filt in data.dtype.names else filt.replace('_filt', '')
                axes[i].plot(data['t_s'], rad2deg(data[col]), color=colors[idx], alpha=0.3, zorder=1)
            
            axes[i].set_ylabel(f'{name} [deg]')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)

        axes[0].set_title('Monte Carlo: Servo Control Effort Envelope (Filtered)')
        axes[1].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Control Efforts (Throttle) Overlay
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
        ax.set_rasterization_zorder(1)
        
        if 'thr_cmd_frac' in datasets[0].dtype.names:
            ax.plot(datasets[0]['t_s'], datasets[0]['thr_cmd_frac'], '--', color='lightgray', label='Cmd Throttle', zorder=0)

        for idx, data in enumerate(datasets):
            ax.plot(data['t_s'], data['fpx_frac'], color=colors[idx], alpha=0.2, zorder=1)
            ax.plot(data['t_s'], data['fnx_frac'], color=colors[idx], alpha=0.2, linestyle=':', zorder=1)
            
        # Add dummy lines for legend
        ax.plot([], [], color='gray', label='PX Throttle Runs', zorder=2)
        ax.plot([], [], color='gray', linestyle=':', label='NX Throttle Runs', zorder=2)
        
        ax.set_ylabel('Throttle Fraction')
        ax.set_title('Monte Carlo: Control Efforts (Throttle) Envelope')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_xlabel('Time [s]')

        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # --- Integrator State Overlay ---
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
        integrators = [('x (Roll)', 'i_err_x'), 
                       ('y (Pitch)', 'i_err_y'), 
                       ('z (Yaw)', 'i_err_z')]
        
        for i, (axis, col) in enumerate(integrators):
            axes[i].set_rasterization_zorder(1)
            if col in datasets[0].dtype.names:
                for idx, data in enumerate(datasets):
                    axes[i].plot(data['t_s'], data[col], color=colors[idx], alpha=0.3, zorder=1)
                
                axes[i].set_ylabel(f'Int Error {axis}')
                axes[i].grid(True)
            
        axes[0].set_title('Monte Carlo: GNC Integrator State (Accumulated Error)')
        axes[2].set_xlabel('Time [s]')
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # --- Histogram Plots (Parameter Spread) ---
        if os.path.exists('mc_logs/mc_params.yaml') and os.path.exists('dispersions.yaml'):
            with open('mc_logs/mc_params.yaml', 'r') as file:
                all_params = yaml.safe_load(file)
            with open('dispersions.yaml', 'r') as file:
                dispersions = yaml.safe_load(file)
            with open('crabcopter.yaml', 'r') as file:
                base_config = yaml.safe_load(file)

            # Gather dispersed keys
            plot_keys = list(dispersions.keys())
            
            # Create subplots dynamically
            num_plots = 0
            for key in plot_keys:
                if isinstance(dispersions[key], list):
                    num_plots += len(dispersions[key])
                else:
                    num_plots += 1
            
            # Create a large figure for all histograms
            cols = 2
            rows = int(np.ceil(num_plots / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3 * rows))
            axes = axes.flatten()
            
            ax_idx = 0
            for key in plot_keys:
                base_val = base_config.get(key, 0)
                disp_val = dispersions[key]
                
                # Extract values across all runs
                vals_list = [p[key] for p in all_params if key in p]
                
                if not vals_list:
                    continue
                    
                if isinstance(disp_val, list):
                    # It's a list like r_cg2px_mm: [100, 0, -50]
                    for i in range(len(disp_val)):
                        element_vals = [v[i] for v in vals_list]
                        nom = base_val[i] if isinstance(base_val, list) else 0
                        d = disp_val[i]
                        
                        ax = axes[ax_idx]
                        ax.hist(element_vals, bins=max(5, len(element_vals)//2), color='skyblue', edgecolor='black')
                        ax.axvline(nom, color='black', linestyle='--', label='Nominal')
                        ax.axvline(nom - d, color='red', linestyle=':', label='-Disp')
                        ax.axvline(nom + d, color='red', linestyle=':', label='+Disp')
                        
                        ax.set_title(f"{key}[{i}]")
                        ax.legend()
                        ax_idx += 1
                else:
                    # Scalar
                    ax = axes[ax_idx]
                    ax.hist(vals_list, bins=max(5, len(vals_list)//2), color='skyblue', edgecolor='black')
                    ax.axvline(base_val, color='black', linestyle='--', label='Nominal')
                    ax.axvline(base_val - disp_val, color='red', linestyle=':', label='-Disp')
                    ax.axvline(base_val + disp_val, color='red', linestyle=':', label='+Disp')
                    
                    ax.set_title(f"{key}")
                    ax.legend()
                    ax_idx += 1
                    
            # Hide empty subplots
            for i in range(ax_idx, len(axes)):
                fig.delaxes(axes[i])
                
            fig.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    print(f"Generated {output_pdf}")

if __name__ == "__main__":
    generate_mc_report()