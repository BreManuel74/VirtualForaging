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
    lick_fig = plt.figure(figsize=(12, 6))
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
        lick_counts = []  # List for daily lick counts
        
        # Process each date's data
        for timestamp, row in df.iterrows():
            try:
                # Read the treadmill data from the file path
                treadmill_data = pd.read_csv(row['treadmill'])
                
                # Calculate average speed for this date
                avg_speed = treadmill_data['speed'].mean()
                
                # Read capacitive data and perform z-score normalization
                capacitive_data = pd.read_csv(row['capacitive'])
                capacitive_values = capacitive_data['capacitive_value']
                
                # Calculate z-score: z = (x - μ) / σ
                z_scores = (capacitive_values - capacitive_values.mean()) / capacitive_values.std()
                
                # Binarize the data based on z-score threshold of 3
                binary_data = (z_scores > 3).astype(int)
                
                # Count lick bouts (transitions from 0 to 1)
                # Convert to numpy array for easier manipulation
                binary_array = binary_data.values
                # Find where values change (True where value changes, False where it stays the same)
                transitions = np.diff(binary_array, prepend=0)
                # Count only 0->1 transitions (positive changes)
                lick_count = np.sum(transitions == 1)
                
                # print(f"\nMouse: {os.path.basename(data_file).split('_')[0]}")
                # print(f"Date: {datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')}")
                # print(f"Total lick bouts: {lick_count}")
                
                # Read trial log data for texture history and reward events
                trial_log = pd.read_csv(row['trial_log'])
                
                # Print first few rows to verify data structure
                # if idx == 0 and timestamp == df.index[0]:  # Only for first mouse, first date
                #     print("\nSample of trial_log data for verification:")
                #     print(trial_log[['texture_history', 'reward_event']].head())
                #     print("\nUnique texture types:")
                #     print(trial_log['texture_history'].unique())
                
                # Count total trials and reward opportunities
                total_trials = len(trial_log['texture_history'].dropna())  # Total number of trials
                reward_opportunities = len(trial_log[trial_log['texture_history'] == 'assets/reward_mean100.jpg'])
                reward_count = len(trial_log['reward_event'].dropna())  # Count non-null reward events
                
                # Calculate misses (reward opportunities minus hits)
                misses = reward_opportunities - reward_count
                
                # Calculate sensitivity only if there are at least 30 trials
                if total_trials >= 30:
                    sensitivity = float(reward_count) / float(reward_opportunities) if reward_opportunities > 0 else 0.0
                else:
                    sensitivity = float('nan')  # Will not be plotted
                
                # Convert Unix timestamp to datetime and store results
                date = datetime.fromtimestamp(int(timestamp))
                
                # # Print detailed stats for verification
                # print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                # print(f"Reward opportunities (reward texture count): {reward_opportunities}")
                # print(f"Actual rewards (hits): {reward_count}")
                # print(f"Misses: {misses}")
                # print(f"Sensitivity: {sensitivity:.3f}")
                
                dates.append(date)
                speeds.append(avg_speed)
                hits.append(reward_count)
                misses_list.append(misses)
                sensitivities.append(sensitivity)
                lick_counts.append(lick_count)
                
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
            'sensitivity': sensitivities,
            'lick_count': lick_counts
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
            
        # Plot lick count data
        plt.figure(lick_fig.number)
        plt.plot(day_numbers, results_df['lick_count'], 
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
    
    # Configure lick count plot
    plt.figure(lick_fig.number)
    plt.title('Lick Counts Over Time')
    plt.xlabel('Day')
    plt.ylabel('Number of Licks')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)  # Lick counts cannot be negative
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    return speed_fig, sensitivity_fig, lick_fig, all_results

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
        speed_fig, sensitivity_fig, lick_fig, results = analyze_mouse_data(file_paths, markers)

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

        # Configure lick count figure
        plt.figure(lick_fig.number)
        if len(file_paths) > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.85)
        else:
            plt.legend()
        plt.tight_layout()

        # Display all plots
        speed_fig.show()
        sensitivity_fig.show()
        lick_fig.show()
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
                
            # Save lick count plot
            lick_save_path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save lick count plot as",
                initialfile=f"mouse_lick_count_comparison_{len(file_paths)}mice.svg"
            )
            if lick_save_path:
                lick_fig.savefig(lick_save_path, bbox_inches='tight', format='svg')
                print(f"Lick count plot saved to: {lick_save_path}")
    else:
        print("No file selected. Exiting...")