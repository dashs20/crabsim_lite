import yaml
import numpy as np
import multiprocessing
import os
import copy
import glob
from sim_cl import run_sim
from mc_report import generate_mc_report

def worker(run_id):
    # Load base config and dispersions
    with open('crabcopter.yaml', 'r') as file:
        base_config = yaml.safe_load(file)
    with open('dispersions.yaml', 'r') as file:
        dispersions = yaml.safe_load(file)
        
    # Apply dispersions
    config = copy.deepcopy(base_config)
    for key, disp in dispersions.items():
        if key in config:
            if isinstance(disp, list):
                # Disperse each element in the list
                for i in range(len(disp)):
                    config[key][i] += np.random.uniform(-disp[i], disp[i])
            else:
                config[key] += np.random.uniform(-disp, disp)
                
    # Run sim
    log_filename = f"mc_logs/log_{run_id}.csv"
    print(f"Starting run {run_id}...")
    run_sim(plant_config_dict=config, log_filename=log_filename, generate_pdf=False)
    print(f"Finished run {run_id}.")
    return config

if __name__ == '__main__':
    os.makedirs('mc_logs', exist_ok=True)
    # Clear existing logs
    for f in glob.glob('mc_logs/log_*.csv'):
        os.remove(f)

    n_runs = 200
    
    # Run in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(worker, range(n_runs))
        
    # Save the parameters used for all runs
    with open('mc_logs/mc_params.yaml', 'w') as file:
        yaml.dump(results, file)
        
    print("Monte Carlo simulation complete!")

    generate_mc_report()

