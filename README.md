# CrabSim Lite

![CrabSim Lite Logo](crablite.png)

CrabSim Lite is a lightweight, Python-based dynamics simulation environment designed for modeling rigid bodies equipped with rotors, specifically focusing on bicopter configurations. It provides a modular approach to simulating complex interactions between body motion, rotor reaction torques, tilting actuator dynamics, and thrust-generated moments.

## Purpose

The primary goal of CrabSim Lite is to capture the major dynamics of an RC, ~1-2kg bicopter (crabcopter) to facilitate experimentation with high-performance GNC (Guidance, Navigation, and Control) algorithms, including rate control and control allocation.

## Features

- **Rigid Body Dynamics:** Simulate 6-DOF (currently focused on 3-DOF angular) dynamics of a rigid body.
- **Advanced Rotor Modeling:** Supports modeling rotors as both reaction wheels (for torque) and thrusters (for force), including gyroscopic effects.
- **GNC System:** Integrated Guidance, Navigation, and Control system including:
    - **PID Control:** Configurable PID controllers for rate management.
    - **Control Allocation:** Sophisticated motor/servo mixing logic to map desired moments to actuator commands.
- **Numerical Integration:** Uses a robust Runge-Kutta 4th order (RK4) integrator for accurate state updates.
- **YAML Configuration:** Easily configure vehicle parameters and GNC settings using YAML files.
- **Data Logging & Visualization:** Support for logging simulation data to CSV and generating detailed PDF reports (Standard and "Smart" versions).

## Project Structure

- `plants.py`: Contains physical models for rigid bodies, rotors, and the `bicopter` assembly.
- `gnc.py`: Implements the GNC system, including the `gnc` class, `allocator`, and `pid` controllers.
- `integrator.py`: Implements the RK4 integration algorithm.
- `util.py`: Utility functions for inertia calculations, unit conversions, and lookup tables.
- `tf.py`: Coordinate transformations and transfer function utilities.
- `sim_cl.py`: Closed-loop simulation script demonstrating autonomous control.
- `sim_ol.py`: Open-loop simulation script for manual actuator commanding.
- `csv_to_pdf_smart.py`: Advanced post-processing script to convert simulation logs into detailed PDF plots (Commanded vs. Actual).
- `verify_allocator.py`: Script to verify the control allocation logic against expected physical outputs.
- `sympy_alloc.py`: Symbolic derivation of the allocation matrix using SymPy.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a Closed-Loop Simulation:**
   Execute the `sim_cl.py` script to run a simulation with the GNC system active.
   ```bash
   python sim_cl.py
   ```
   This will generate a `log.csv` file.

3. **Visualize Results:**
   Use the `csv_to_pdf_smart.py` script to generate a visual report comparing commanded and actual states.
   ```bash
   python csv_to_pdf_smart.py
   ```
   The output will be saved as `log_smart.pdf`.

## Configuration

The simulation and GNC systems are configured via YAML files:
- `crabcopter.yaml`: Defines physical properties of the vehicle (mass, dimensions, motor lookup).
- `crabcopter_gnc.yaml`: Defines GNC settings (PID gains, allocator constraints).
- `smart_PID.yaml`: Specialized PID configuration.

## Core Components

### `bicopter` (`plants.py`)
A high-level class that assembles the rigid body and two tilting rotor assemblies (PX and NX). It handles the propagation of state based on actuator commands.

### `gnc` (`gnc.py`)
Manages the control loop, taking desired setpoints and current states to produce actuator commands via PID controllers and a control allocator.

### `allocator` (`gnc.py`)
Maps desired 3-axis moments and throttle requests to specific motor speeds and servo angles. It uses an inverted actuator-to-moment matrix derived in `sympy_alloc.py`.

---
*Developed for high-fidelity, lightweight dynamics research.*
