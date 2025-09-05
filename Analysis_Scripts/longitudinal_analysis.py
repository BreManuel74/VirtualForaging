import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import colorsys

def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.rand() * 0.3  # Random between 0.7-1.0
        value = 0.7 + np.random.rand() * 0.3       # Random between 0.7-1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def analyze_mouse_data(data_files, markers):
    speed_fig = plt.figure(figsize=(12, 6))
    sensitivity_fig = plt.figure(figsize=(12, 6))
    colors = generate_colors(len(data_files))  # Generate colors based on number of mice
    
    all_results = []
    
    for idx, data_file in enumerate(data_files):
        # Read the combined data file
        df = pd.read_csv(data_file, index_col='timestamp')
        
        print(f"Reading data from: {data_file}")
        
        # Initialize lists to store results
        dates = []
        speeds = []
        hits = []  # List for reward events
        misses_list = []  # List for misses (texture changes minus hits)
        sensitivities = []  # List for sensitivity values
        
        # Process each date's data
        for timestamp, row in df.iterrows():
            try:
                # Read the treadmill data from the file path
                treadmill_data = pd.read_csv(row['treadmill'])
                
                # Calculate average speed for this date
                avg_speed = treadmill_data['speed'].mean()
                
                # Read trial log data for texture history and reward events
                trial_log = pd.read_csv(row['trial_log'])
                texture_count = len(trial_log['texture_history'].dropna())  # Count non-null texture entries
                reward_count = len(trial_log['reward_event'].dropna())  # Count non-null reward events
                
                # Calculate misses (texture changes minus hits)
                misses = texture_count - reward_count
                
                # Calculate sensitivity (hits / total trials)
                # Convert to float to ensure proper division
                sensitivity = float(reward_count) / float(texture_count) if texture_count > 0 else 0.0
                
                # Convert Unix timestamp to datetime and store results
                date = datetime.fromtimestamp(int(timestamp))
                dates.append(date)
                speeds.append(avg_speed)
                hits.append(reward_count)
                misses_list.append(misses)
                sensitivities.append(sensitivity)
                
                #print(f"Processed date {date.strftime('%Y-%m-%d')}: Average speed = {avg_speed:.2f}, Hits = {reward_count}, Misses = {misses}")
                
            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                print("Raw treadmill data:")
                print(row['treadmill'][:500])  # Print first 500 chars
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'average_speed': speeds,
            'hits': hits,
            'misses': misses_list,
            'sensitivity': sensitivities
        })
        
        # Sort and remove duplicates
        results_df = results_df.drop_duplicates(subset=['date'])
        results_df = results_df.sort_values('date')
        
        # Remove the first date as requested for hits, misses, and sensitivity analysis
        results_df.loc[1:, 'hits'] = results_df.loc[1:, 'hits']  # Keep only hits after first date
        results_df.loc[1:, 'misses'] = results_df.loc[1:, 'misses']  # Keep only misses after first date
        results_df.loc[1:, 'sensitivity'] = results_df.loc[1:, 'sensitivity']  # Keep only sensitivity after first date
        
        # Get mouse name
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Store results for this mouse
        all_results.append({
            'mouse': mouse_name,
            'dates': dates,
            'speeds': speeds,
            'df': results_df
        })
        
        # Plot this mouse's data with sequential day numbers and specified marker
        day_numbers = np.arange(1, len(results_df) + 1)
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Plot speed data
        plt.figure(speed_fig.number)
        plt.plot(day_numbers, results_df['average_speed'], 
            f'{markers[mouse_name]}-', color=colors[idx], markersize=8, label=mouse_name)
        
        # Plot sensitivity data
        plt.figure(sensitivity_fig.number)
        plt.plot(day_numbers, results_df['sensitivity'], 
            f'{markers[mouse_name]}-', color=colors[idx], markersize=8, label=mouse_name)
    
    # Configure speed plot
    plt.figure(speed_fig.number)
    plt.title('Average Speed Over Time')
    plt.xlabel('Day')
    plt.ylabel('Average Speed')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=-10)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Configure sensitivity plot
    plt.figure(sensitivity_fig.number)
    plt.title('Sensitivity Over Time')
    plt.xlabel('Day')
    plt.ylabel('Sensitivity (Hits / Total Trials)')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.05, 1.05)  # Sensitivity is between 0 and 1
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    return speed_fig, sensitivity_fig, all_results

if __name__ == "__main__":
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select multiple data files
    file_paths = filedialog.askopenfilenames(
        title='Select mouse data files',
        filetypes=[('CSV files', '*.csv')],
        initialdir=os.getcwd()  # Start in current directory
    )
    
    if file_paths:
        # Create a dictionary to store marker choices
        markers = {}
        
        # Ask for marker type for each mouse
        for file_path in file_paths:
            mouse_name = os.path.basename(file_path).split("_")[0]
            while True:
                choice = input(f"Enter marker type for {mouse_name} (s for square, o for circle): ").lower().strip()
                if choice in ['s', 'o']:
                    markers[mouse_name] = choice
                    break
                else:
                    print("Invalid choice. Please enter 's' for square or 'o' for circle.")
        
        # Run the analysis with marker choices
        speed_fig, sensitivity_fig, results = analyze_mouse_data(file_paths, markers)

        # Configure speed figure
        plt.figure(speed_fig.number)
        if len(file_paths) > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.85)
        else:
            plt.legend()
        plt.tight_layout()

        # Configure sensitivity figure
        plt.figure(sensitivity_fig.number)
        if len(file_paths) > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.85)
        else:
            plt.legend()
        plt.tight_layout()

        # Display both plots
        speed_fig.show()
        sensitivity_fig.show()
        plt.show()

        # Ask if user wants to save the plot
        save = input("Would you like to save the plot? (yes/no): ").lower().strip()
        if save.startswith('y'):
            # Set common plot parameters
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['svg.fonttype'] = 'none'

            # Save speed plot
            speed_save_path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save speed plot as",
                initialfile=f"mouse_speed_comparison_{len(file_paths)}mice.svg"
            )
            if speed_save_path:
                speed_fig.savefig(speed_save_path, bbox_inches='tight', format='svg')
                print(f"Speed plot saved to: {speed_save_path}")
            
            # Save sensitivity plot
            sensitivity_save_path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save sensitivity plot as",
                initialfile=f"mouse_sensitivity_comparison_{len(file_paths)}mice.svg"
            )
            if sensitivity_save_path:
                sensitivity_fig.savefig(sensitivity_save_path, bbox_inches='tight', format='svg')
                print(f"Sensitivity plot saved to: {sensitivity_save_path}")
    else:
        print("No file selected. Exiting...")