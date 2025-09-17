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
    # Create a dictionary to map mouse names to markers
    markers = {os.path.basename(file).split("_")[0]: marker for file, marker in zip(data_files, markers)}
    
    speed_fig = plt.figure(figsize=(12, 6))
    sensitivity_fig = plt.figure(figsize=(12, 6))
    lick_fig = plt.figure(figsize=(12, 6))
    reward_fig = plt.figure(figsize=(12, 6))
    avg_reward_fig = plt.figure(figsize=(12, 6))  # Average rewards figure
    sex_reward_fig = plt.figure(figsize=(12, 6))  # Sex-specific average rewards figure
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
        session_lengths = []  # List for session lengths in minutes
        
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
                
                # Calculate session length in minutes from the elapsed_time column
                session_length_minutes = capacitive_data['elapsed_time'].max() / 60.0
                
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
                session_lengths.append(session_length_minutes)
                
                #print(f"Processed date {date.strftime('%Y-%m-%d')}: Average speed = {avg_speed:.2f}, Hits = {reward_count}, Misses = {misses}, Session Length = {session_length_minutes:.1f} min")
                
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
            'lick_count': lick_counts,
            'session_length': session_lengths
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
            'hits': hits,
            'session_lengths': session_lengths,
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
            
        # Plot reward count data
        plt.figure(reward_fig.number)
        plt.plot(day_numbers, results_df['hits'], 
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
    
    # Configure reward count plot
    plt.figure(reward_fig.number)
    plt.title('Number of Rewards Over Time')
    plt.xlabel('Day')
    plt.ylabel('Number of Rewards')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)  # Reward counts cannot be negative
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Calculate average rewards/minute and SEM across mice
    # First, find the maximum number of days
    max_days = max(len(result['hits']) for result in all_results)
    
    # Initialize arrays for rewards per minute (all mice)
    all_rewards_per_min = np.zeros((len(data_files), max_days))
    all_rewards_per_min[:] = np.nan  # Fill with NaN initially
    
    # Initialize arrays for sex-specific rewards per minute
    male_rewards_per_min = []
    female_rewards_per_min = []
    
    # Fill in the rewards per minute data
    for i, result in enumerate(all_results):
        rewards = np.array(result['hits'])
        session_lengths = np.array(result['session_lengths'])
        # Calculate rewards per minute
        rewards_per_min = rewards / session_lengths
        all_rewards_per_min[i, :len(rewards_per_min)] = rewards_per_min
        
        # Separate data by sex based on marker type
        mouse_name = result['mouse']
        if markers[mouse_name] == 's':  # Male
            male_rewards_per_min.append(rewards_per_min)
        else:  # Female (marker 'o')
            female_rewards_per_min.append(rewards_per_min)
    
    # Convert lists to arrays and pad with NaN to make them rectangular
    if male_rewards_per_min:
        male_rewards_per_min = np.array([np.pad(x, (0, max_days - len(x)), 
                                               constant_values=np.nan) for x in male_rewards_per_min])
    if female_rewards_per_min:
        female_rewards_per_min = np.array([np.pad(x, (0, max_days - len(x)), 
                                                constant_values=np.nan) for x in female_rewards_per_min])
    
    # Calculate mean and SEM across mice for each day
    mean_rewards_per_min = np.nanmean(all_rewards_per_min, axis=0)
    sem_rewards_per_min = np.nanstd(all_rewards_per_min, axis=0) / np.sqrt(np.sum(~np.isnan(all_rewards_per_min), axis=0))
    
    # Plot average rewards/minute with SEM
    plt.figure(avg_reward_fig.number)
    day_numbers = np.arange(1, max_days + 1)
    plt.plot(day_numbers, mean_rewards_per_min, '-', color='black', linewidth=2, label='Mean')
    plt.fill_between(day_numbers, mean_rewards_per_min - sem_rewards_per_min, mean_rewards_per_min + sem_rewards_per_min, 
                     color='gray', alpha=0.3, label='SEM')
    
    # Configure average rewards plot
    plt.title('Average Rewards Per Minute Across Mice')
    plt.xlabel('Day')
    plt.ylabel('Rewards per Minute (Mean ± SEM)')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()
    
    # Plot sex-specific average rewards/minute with SEM
    plt.figure(sex_reward_fig.number)
    day_numbers = np.arange(1, max_days + 1)
    
    # Plot male data if available
    if len(male_rewards_per_min) > 0:
        # Check if we have any non-NaN values
        valid_male_data = np.any(~np.isnan(male_rewards_per_min))
        if valid_male_data:
            mean_male = np.nanmean(male_rewards_per_min, axis=0)
            # Only calculate SEM where we have more than one value
            n_male = np.sum(~np.isnan(male_rewards_per_min), axis=0)
            sem_male = np.where(n_male > 1, 
                              np.nanstd(male_rewards_per_min, axis=0) / np.sqrt(n_male),
                              0)
            plt.plot(day_numbers, mean_male, '-', color='green', linewidth=2, label=f'Male (n={len(male_rewards_per_min)})')
            plt.fill_between(day_numbers, mean_male - sem_male, mean_male + sem_male, 
                           color='green', alpha=0.2)

    # Plot female data if available
    if len(female_rewards_per_min) > 0:
        # Check if we have any non-NaN values
        valid_female_data = np.any(~np.isnan(female_rewards_per_min))
        if valid_female_data:
            mean_female = np.nanmean(female_rewards_per_min, axis=0)
            # Only calculate SEM where we have more than one value
            n_female = np.sum(~np.isnan(female_rewards_per_min), axis=0)
            sem_female = np.where(n_female > 1,
                                np.nanstd(female_rewards_per_min, axis=0) / np.sqrt(n_female),
                                0)
            plt.plot(day_numbers, mean_female, '-', color='purple', linewidth=2, label=f'Female (n={len(female_rewards_per_min)})')
            plt.fill_between(day_numbers, mean_female - sem_female, mean_female + sem_female, 
                           color='purple', alpha=0.2)

    # Configure sex-specific rewards plot
    plt.title('Sex-Specific Average Rewards Per Minute')
    plt.xlabel('Day')
    plt.ylabel('Rewards per Minute (Mean ± SEM)')
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()

    return speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, all_results

def main():
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
        # Ask user for marker type for each mouse
        markers = []
        for file_path in file_paths:
            mouse_name = os.path.basename(file_path).split("_")[0]
            while True:
                choice = input(f"Enter marker type for {mouse_name} (s for square, o for circle): ").lower().strip()
                if choice in ['s', 'o']:
                    markers.append(choice)
                    break
                else:
                    print("Invalid choice. Please enter 's' for square or 'o' for circle.")
            
        # Analyze data and plot results
        speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, all_results = analyze_mouse_data(file_paths, markers)

        # Configure all figures
        for fig in [speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig]:
            plt.figure(fig.number)
            if len(file_paths) > 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.85)
            else:
                plt.legend()
            plt.tight_layout()

        # Display all plots
        for fig in [speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig]:
            fig.show()
        plt.show()

        # Ask if user wants to save the plots
        save = input("Would you like to save the plots? (yes/no): ").lower().strip()
        if save.startswith('y'):
            # Set common plot parameters
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['svg.fonttype'] = 'none'

            # Plot configurations to save
            plot_configs = [
                (speed_fig, 'speed', 'Speed plot'),
                (sensitivity_fig, 'sensitivity', 'Sensitivity plot'),
                (lick_fig, 'lick_count', 'Lick count plot'),
                (reward_fig, 'reward_count', 'Reward count plot'),
                (avg_reward_fig, 'avg_reward', 'Average rewards plot'),
                (sex_reward_fig, 'sex_reward', 'Sex-specific average rewards plot')
            ]

            # Save all plots
            for fig, name, title in plot_configs:
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".svg",
                    filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                    title=f"Save {title} as",
                    initialfile=f"mouse_{name}_comparison_{len(file_paths)}mice.svg"
                )
                if save_path:
                    fig.savefig(save_path, bbox_inches='tight', format='svg')
                    print(f"{title} saved to: {save_path}")
    else:
        print("No file selected. Exiting...")
if __name__ == "__main__":
    main()