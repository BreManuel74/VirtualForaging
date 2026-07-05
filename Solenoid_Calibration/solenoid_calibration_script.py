"""
Solenoid Calibration Script
Original author: Rafiq Huda
Updated by: Grayson Sipe & Brenna Manuel
============================
Calibrates a water-delivery solenoid by measuring the volume dispensed
over a range of opening durations (dt values) and fitting a linear model:

    dw (μL) = b0 * dt (ms) + b1

Workflow
--------
1. Connect to an Arduino that controls the solenoid.
2. Optionally load a previous calibration session from CSV.
3. Fire the solenoid a fixed number of times per dt value.
4. Enter the cumulative water volume collected after each dt sweep.
5. Fit a linear regression, display an R² goodness-of-fit, and plot results.
6. Optionally append the new batch to the calibration CSV.

Usage
-----
    python solenoid_calibration_script.py

Configuration constants (inside main())
----------------------------------------
arduino_port  : str   – COM port of the Arduino (default "COM7").
baud_rate     : int   – Serial baud rate (default 115200).
ndt           : int   – Number of dt test points (default 4).
num_reps      : int   – Solenoid pulses per dt point (default 30).
r2_threshold  : float – Minimum acceptable R² (default 0.9).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import time
import serial

def connect_to_arduino(port, baud_rate):
    """
    Establish a serial connection to the Arduino.

    Parameters
    ----------
    port : str
        The COM port the Arduino is connected to (e.g. "COM7").
    baud_rate : int
        The serial communication baud rate (e.g. 115200).

    Returns
    -------
    serial.Serial or None
        An open Serial object on success, or None if the connection fails.
    """
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on port {port} at {baud_rate} baud.")
        return arduino
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino: {e}")
        return None

def load_previous_calibration(file_path):
    """
    Load previous calibration data from a CSV file into a DataFrame.

    Parameters
    ----------
    file_path : str
        Absolute path to the calibration CSV file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the previous calibration data, or None if the
        file does not exist or cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Starting with new calibration data.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Previous calibration data loaded successfully.")
        #print("Columns in the loaded DataFrame:", df.columns)  # Debugging: Print column names
        #print("First few rows of the DataFrame:\n", df.head())  # Debugging: Print first few rows
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def save_calibration_data(file_path, calibration_data, batch_id):
    """
    Append calibration data to a CSV file, creating it if it does not exist.

    A ``batch_id`` column is added to the DataFrame before saving so that
    multiple calibration sessions can be distinguished in the same file.

    Parameters
    ----------
    file_path : str
        Absolute path to the destination CSV file.
    calibration_data : pandas.DataFrame
        DataFrame containing columns: dt, w, dw, b0, b1.
    batch_id : int
        Integer identifier for the current calibration session.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        # Add the batch_id column to the calibration data
        calibration_data['batch_id'] = batch_id

        # Check if the file already exists
        if os.path.exists(file_path):
            # Append to the existing file without writing the header again
            calibration_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Write a new file with the header
            calibration_data.to_csv(file_path, index=False)
        print(f"Calibration data saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def get_solenoid_dt_values(ndt, previous_dt_values=None):
    """
    Retrieve solenoid opening-time test points (dt values) in milliseconds.

    If ``previous_dt_values`` is supplied the user is not prompted and those
    values are reused.  Otherwise the user is interactively asked to enter
    ``ndt`` integer values.

    Parameters
    ----------
    ndt : int
        Number of dt test points to collect.
    previous_dt_values : list of int, optional
        Pre-existing dt values to reuse instead of prompting the user.

    Returns
    -------
    list of int
        Solenoid opening times in milliseconds.
    """
    if previous_dt_values is not None:
        print("Using previous solenoid dt values:", previous_dt_values)
        return previous_dt_values
    else:
        dt_values = []
        for i in range(ndt):
            while True:
                try:
                    dt = int(input(f"Enter solenoid opening time (ms) dt({i + 1}): "))
                    dt_values.append(dt)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        print("New solenoid dt values entered:", dt_values)
        return dt_values

def collect_water_volumes(arduino, dt_values, num_reps, initial_volume):
    """
    Drive the solenoid at each dt value and record the dispensed water volumes.

    For every dt in ``dt_values`` the solenoid is pulsed ``num_reps`` times via
    the Arduino, then the user is prompted to enter the cumulative volume in the
    collection vessel.  The net volume (cumulative minus ``initial_volume``) is
    stored.

    Parameters
    ----------
    arduino : serial.Serial
        Open serial connection to the Arduino.
    dt_values : list of int
        Solenoid opening times to test, in milliseconds.
    num_reps : int
        Number of solenoid pulses per dt test point.
    initial_volume : float
        Volume of water already in the collection vessel before testing (mL).

    Returns
    -------
    list of float
        Net water volumes dispensed (mL) for each dt value.
    """
    water_volumes = []
    for j, dt in enumerate(dt_values):
        print(f"Testing solenoid opening time: {dt} ms")
        
        # Clear input buffer
        while arduino.in_waiting > 4:
            arduino.readline()
        
        for _ in range(num_reps):
            signal = int(f"2{dt}")
            print(signal)
            arduino.write(str(signal).encode() + b"\n")
            time.sleep(0.5)
        
        while True:
            try:
                cumulative_volume = float(input(f"Enter cumulative water amount (mL) for dt({j + 1}): "))
                water_volumes.append(cumulative_volume - initial_volume)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return water_volumes

def main():
    """
    Run the interactive solenoid calibration routine.

    Steps
    -----
    1. Connect to the Arduino over the configured serial port.
    2. Optionally load an existing calibration CSV to compare against.
    3. Collect or reuse solenoid opening-time (dt) test points.
    4. Pulse the solenoid and record dispensed volumes for each dt.
    5. Fit a linear model (dw = b0*dt + b1) and compute R².
    6. Plot current data alongside the previous fit (if available).
    7. Prompt the user to save results; append to the calibration CSV if confirmed.
    """
    print("Welcome to Solenoid Calibration!")
    
    # Configuration
    arduino_port = "COM7"  # Replace with your Arduino's port
    baud_rate = 115200
    r2_threshold = 0.9
    ndt = 4  # Number of solenoid opening times to test
    num_reps = 30
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "solenoid_calibration_results.csv"
    file_path = os.path.join(script_dir, file_name)
    
    # Connect to Arduino
    arduino = connect_to_arduino(arduino_port, baud_rate)
    if not arduino:
        return
    
    # Load previous calibration data
    open_previous = input("Open previous calibration results file? ([0]/[1]): ").strip()
    previous_data = load_previous_calibration(file_path) if open_previous == "1" else None
    
    # Determine the next batch_id
    if previous_data is not None:
        batch_id = previous_data['batch_id'].max() + 1
    else:
        batch_id = 1
    
    # Get solenoid dt values
    use_previous = input("Use previous solenoid dt values? ([0]/[1]): ").strip()
    dt_values = get_solenoid_dt_values(ndt, previous_data['dt'].tolist() if previous_data is not None and use_previous == "1" else None)
    
    # Get initial water volume
    while True:
        try:
            initial_volume = float(input("Enter the starting volume of water (in mL): "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    
    # Collect water volumes
    water_volumes = collect_water_volumes(arduino, dt_values, num_reps, initial_volume)
    print("Collected water volumes:", water_volumes)
    
    # Perform calibration calculations
    dw_values = np.array(water_volumes) / num_reps * 1000  # Convert to μL
    calibration_fit = np.polyfit(dt_values, dw_values, 1)  # Linear fit: dw = b0 * dt + b1
    
    # Create a pandas DataFrame for calibration data
    calibration_data = pd.DataFrame({
        'dt': dt_values,
        'w': water_volumes,
        'dw': dw_values,
        'b0': [calibration_fit[0]] * len(dt_values),  # Slope
        'b1': [calibration_fit[1]] * len(dt_values)   # Intercept
    })
    
    # Plot results
    plt.figure()
    plt.plot(dt_values, dw_values, '*', label='Data')
    x = np.array([min(dt_values), max(dt_values)])  # Use dt_values for the x-axis range
    plt.plot(x, calibration_fit[0] * x + calibration_fit[1], '--', label=f'dw = {calibration_fit[0]:.2f}*dt + {calibration_fit[1]:.2f}')
    
    if previous_data is not None:
        prev_dw = previous_data['dw']
        prev_fit = [previous_data['b0'].iloc[0], previous_data['b1'].iloc[0]]
        plt.plot(prev_fit[0] * prev_dw + prev_fit[1], prev_dw, 'b--', label='Previous fit')
    
    plt.xlabel('Solenoid ON time (ms)')
    plt.ylabel('Water amount (μL)')
    plt.legend(loc='lower right')
    plt.show(block=False)
    
    # Compute and display R²
    r2 = round(np.corrcoef(dw_values, dt_values)[0, 1] ** 2, 2)
    print(f'R² value: {r2}')
    if r2 < r2_threshold:
        print('RECALIBRATE, DO NOT SAVE!!!')
    
    # Save calibration data
    save = input('Save? (0/[1]): ').strip()
    if not save or save == "1":
        save_calibration_data(file_path, calibration_data, batch_id)

if __name__ == "__main__":
    main()