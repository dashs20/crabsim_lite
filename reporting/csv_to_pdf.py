import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_pdf_plots(input_csv):
    # Automatically change the file extension from .csv to .pdf
    base_path, _ = os.path.splitext(input_csv)
    output_pdf = f"{base_path}.pdf"

    # --- 1. Load Data ---
    try:
        # Try loading with headers
        data = np.genfromtxt(input_csv, delimiter=',', names=True, deletechars="")
        if data.dtype.names is None:
            raise ValueError("No headers found")
        
        column_names = data.dtype.names
        x_name = column_names[0]
        x_data = data[x_name]
        y_names = column_names[1:]
        y_data_list = [data[name] for name in y_names]
    except Exception:
        # Fallback: load as raw matrix if headers fail
        data = np.genfromtxt(input_csv, delimiter=',')
        if len(data.shape) < 2 or data.shape[1] < 2:
            print("Error: CSV must have at least 2 columns and multiple rows.")
            return
        
        x_data = data[:, 0]
        y_data_list = [data[:, i] for i in range(1, data.shape[1])]
        column_names = ["Column 0"] + [f"Column {i}" for i in range(1, data.shape[1])]
        x_name = column_names[0]
        y_names = column_names[1:]

    # --- 2. Plotting Configuration ---
    num_plots = len(y_data_list)
    plots_per_page = 3

    # --- 3. Generate PDF ---
    with PdfPages(output_pdf) as pdf:
        for i in range(0, num_plots, plots_per_page):
            # Create a 3x1 grid of subplots for each page
            fig, axes = plt.subplots(plots_per_page, 1, figsize=(8.5, 11))
            
            # Ensure axes is always a flat list even if plots_per_page is 1
            axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]

            for j in range(plots_per_page):
                idx = i + j
                ax = axes_list[j]
                
                if idx < num_plots:
                    ax.plot(x_data, y_data_list[idx])
                    ax.set_title(y_names[idx])
                    ax.set_xlabel(x_name)
                    ax.grid(True, linestyle='--', alpha=0.7)
                else:
                    # Hide empty subplots on the final page
                    ax.axis('off')
            
            fig.tight_layout(pad=3.0)
            pdf.savefig(fig)
            plt.close(fig)

    total_pages = (num_plots + plots_per_page - 1) // plots_per_page
    print(f"Generated {output_pdf} with {num_plots} plots across {total_pages} pages.")

generate_pdf_plots("log.csv")