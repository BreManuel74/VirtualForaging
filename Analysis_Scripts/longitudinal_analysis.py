"""Longitudinal analysis of behavioral data across multiple mice.
Original Author: Brenna Manuel
"""

import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import colorsys
import sys
from scipy import stats

# Add Analysis_Scripts to path to import lick detection algorithm
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import lick_detection_algorithm as lda

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

def analyze_levels(data_files):
    """Analyze rewards/min for each level across all mice."""
    level_data = {}  # Dictionary to store rewards/min for each level
    
    for data_file in data_files:
        # Read the data file
        df = pd.read_csv(data_file)
        
        # Group by level and calculate rewards/min for each group
        for level in df['level'].unique():
            if level not in level_data:
                level_data[level] = []
                
            level_group = df[df['level'] == level]
            for _, row in level_group.iterrows():
                try:
                    # Read trial log data
                    trial_log = pd.read_csv(row['trial_log'])
                    # Count rewards (non-null reward events)
                    hits = len(trial_log['reward_event'].dropna())
                    # Calculate session length in minutes
                    capacitive_data = pd.read_csv(row['capacitive'])
                    session_length = capacitive_data['elapsed_time'].max() / 60.0
                    # Calculate rewards per minute
                    rewards_per_min = hits / session_length if session_length > 0 else 0
                    level_data[level].append(rewards_per_min)
                except Exception as e:
                    print(f"Error processing file for level {level}: {str(e)}")
    
    # Calculate statistics for each level
    level_stats = {}
    for level, rewards in level_data.items():
        if rewards:  # Only process if we have data
            level_stats[level] = {
                'mean': np.mean(rewards),
                'sem': np.std(rewards) / np.sqrt(len(rewards)),
                'n': len(rewards)
            }
    
    # Create bar plot
    level_fig = plt.figure(figsize=(15, 8))  # Larger figure size
    # Sort levels - numerical levels first, then alphabetically
    def sort_key(x):
        # Try to extract level number if it follows "level_X" pattern
        if x.startswith('level_'):
            try:
                return (0, int(x.split('_')[1].split('.')[0]))
            except (ValueError, IndexError):
                pass
        # Otherwise sort alphabetically (put after numbered levels)
        return (1, x)
    
    levels = sorted(level_stats.keys(), key=sort_key)
    means = [level_stats[level]['mean'] for level in levels]
    sems = [level_stats[level]['sem'] for level in levels]
    
    # Plot bars
    bars = plt.bar(range(len(levels)), means, yerr=sems, capsize=5, label='Mean rewards per minute')
    plt.xticks(range(len(levels)), levels, rotation=45, ha='right')
    
    # Configure plot
    plt.title('Average Rewards Per Minute by Level')
    plt.xlabel('Level')
    plt.ylabel('Rewards per Minute (Mean ± SEM)')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(bottom=0.2)
    
    # Add sample size annotations
    for i, level in enumerate(levels):
        n = level_stats[level]['n']
        plt.text(i, means[i], f'n={n}', ha='center', va='bottom')
    
    plt.tight_layout()
    return level_fig

def analyze_mouse_data(data_files, markers, starting_conditions, save_lick_plots=False, output_dir=None):
    # Create dictionaries to map mouse names to markers and starting conditions
    markers = {os.path.basename(file).split("_")[0]: marker for file, marker in zip(data_files, markers)}
    conditions = {os.path.basename(file).split("_")[0]: condition for file, condition in zip(data_files, starting_conditions)}
    
    # Create output directory for lick detection plots if needed
    if save_lick_plots and output_dir:
        lick_plots_dir = os.path.join(output_dir, 'lick_detection_plots')
        os.makedirs(lick_plots_dir, exist_ok=True)
        print(f"\nSaving lick detection plots to: {lick_plots_dir}")
    
    # Create color mapping based on starting conditions
    unique_conditions = list(set(starting_conditions))
    condition_colors = generate_colors(len(unique_conditions))
    condition_color_map = {condition: color for condition, color in zip(unique_conditions, condition_colors)}
    
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
                
                # Calculate average speed for this date (convert from mm/s to cm/s)
                avg_speed = treadmill_data['speed'].mean() / 10.0
                
                # Read capacitive data for lick detection
                capacitive_data = pd.read_csv(row['capacitive'])
                
                # Calculate session length in minutes from the elapsed_time column
                session_length_minutes = capacitive_data['elapsed_time'].max() / 60.0
                
                # Use new lick detection algorithm
                # Prepare data with Time_sec column
                cap_df = capacitive_data.copy()
                cap_df['Time_sec'] = cap_df['elapsed_time']
                
                # Compute KDE normalization
                kde_value = lda.compute_KDE(cap_df, 'capacitive_value')
                cap_df = lda.compute_KDE_normalizations(cap_df, 'capacitive_value', kde_value)
                
                # Detect lick events using dynamic threshold (max_deviation / 2)
                events_df, threshold_used = lda.detect_events_above_threshold(cap_df, 'capacitive_value', threshold=None)
                
                # Count total lick events
                lick_count = events_df['capacitive_value_event'].sum()
                
                # Optionally save lick detection plots
                if save_lick_plots and output_dir:
                    mouse_name = os.path.basename(data_file).split('_')[0]
                    date_str = datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
                    plot_filename = f"{mouse_name}_{date_str}_lick_detection.png"
                    plot_path = os.path.join(lick_plots_dir, plot_filename)
                    
                    # Create summary plot
                    fig = lda.plot_summary(
                        cap_df, events_df, 
                        column='capacitive_value',
                        kde_value=kde_value, 
                        threshold=threshold_used,
                        title=f"{mouse_name} - {date_str} - {lick_count} licks detected",
                        show=False
                    )
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                # print(f"\nMouse: {os.path.basename(data_file).split('_')[0]}")
                # print(f"Date: {datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')}")
                # print(f"Total licks detected: {lick_count}")
                
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
                
                # Calculate sensitivity only if there is at least one trial
                if total_trials >= 1:
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
            'starting_condition': conditions[mouse_name],
            'df': results_df
        })
        
        # Plot this mouse's data with sequential day numbers and specified marker
        day_numbers = np.arange(0, len(results_df))
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Use color based on starting condition for all plots
        condition_color = condition_color_map[conditions[mouse_name]]
        
        # Plot speed data
        plt.figure(speed_fig.number)
        plt.plot(day_numbers, results_df['average_speed'], 
            f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)
        
        # Plot sensitivity data
        plt.figure(sensitivity_fig.number)
        plt.plot(day_numbers, results_df['sensitivity'], 
            f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)
            
        # Plot lick count data
        plt.figure(lick_fig.number)
        plt.plot(day_numbers, results_df['lick_count'], 
            f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)
            
        # Plot reward count data
        plt.figure(reward_fig.number)
        plt.plot(day_numbers, results_df['hits'], 
            f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)
    
    # Configure speed plot
    plt.figure(speed_fig.number)
    plt.title('Average Speed Over Time')
    plt.xlabel('Training Day')
    plt.ylabel('Average Speed (cm/s)')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    max_day = max(len(result['df']) for result in all_results)
    ax.set_xlim(left=0, right=max_day - 0.5)  # Add padding to prevent data points from being cut off
    # Dynamic tick spacing based on data range
    if max_day <= 10:
        major_spacing = 1
        minor_spacing = 1
    elif max_day <= 20:
        major_spacing = 2
        minor_spacing = 1
    elif max_day <= 50:
        major_spacing = 5
        minor_spacing = 1
    elif max_day <= 100:
        major_spacing = 10
        minor_spacing = 2
    else:
        major_spacing = 20
        minor_spacing = 5
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Configure sensitivity plot
    plt.figure(sensitivity_fig.number)
    plt.title('Sensitivity Over Time')
    plt.xlabel('Day')
    plt.ylabel('Sensitivity (Hits / Total Trials)')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.05, 1.05)  # Sensitivity is between 0 and 1
    ax.set_xlim(left=0, right=max_day - 0.5)  # Add padding to prevent data points from being cut off
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Configure lick count plot
    plt.figure(lick_fig.number)
    plt.title('Lick Counts Over Time')
    plt.xlabel('Training Day')
    plt.ylabel('Number of Licks')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)  # Lick counts cannot be negative
    ax.set_xlim(left=0, right=max_day - 0.5)  # Add padding to prevent data points from being cut off
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Configure reward count plot
    plt.figure(reward_fig.number)
    plt.title('Total Reward Count Over Time')
    plt.xlabel('Training Day')
    plt.ylabel('Number of Rewards')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)  # Reward counts cannot be negative
    ax.set_xlim(left=0, right=max_day - 0.5)  # Add padding to prevent data points from being cut off
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Calculate average rewards/minute and SEM across mice
    # First, find the maximum number of days
    max_days = max(len(result['hits']) for result in all_results)
    
    # Dynamic tick spacing based on data range for aggregate plots
    if max_days <= 10:
        agg_major_spacing = 1
        agg_minor_spacing = 1
    elif max_days <= 20:
        agg_major_spacing = 2
        agg_minor_spacing = 1
    elif max_days <= 50:
        agg_major_spacing = 5
        agg_minor_spacing = 1
    elif max_days <= 100:
        agg_major_spacing = 10
        agg_minor_spacing = 2
    else:
        agg_major_spacing = 20
        agg_minor_spacing = 5
    
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
    day_numbers = np.arange(0, max_days)
    plt.plot(day_numbers, mean_rewards_per_min, '-', color='black', linewidth=2, label='Mean')
    plt.fill_between(day_numbers, mean_rewards_per_min - sem_rewards_per_min, mean_rewards_per_min + sem_rewards_per_min, 
                     color='gray', alpha=0.3, label='SEM')
    
    # Configure average rewards plot
    plt.title('Average Rewards Per Minute Across Mice')
    plt.xlabel('Day')
    plt.ylabel('Rewards per Minute (Mean ± SEM)')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_days - 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()
    
    # Plot sex-specific average rewards/minute with SEM
    plt.figure(sex_reward_fig.number)
    day_numbers = np.arange(0, max_days)
    
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
    plt.xlabel('Training Day')
    plt.ylabel('Average Rewards per Minute')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_days - 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(agg_major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(agg_minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()
    
    # Statistical Analysis: Sex-based comparison
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: SEX-BASED COMPARISON (Two-tailed t-test)")
    print("=" * 70)
    sex_sig_days = []  # Store significant days for plotting
    if len(male_rewards_per_min) > 0 and len(female_rewards_per_min) > 0:
        # Overall comparison using all data points (excluding NaN)
        male_all_data = male_rewards_per_min.flatten()
        male_all_data = male_all_data[~np.isnan(male_all_data)]
        female_all_data = female_rewards_per_min.flatten()
        female_all_data = female_all_data[~np.isnan(female_all_data)]
        
        if len(male_all_data) > 0 and len(female_all_data) > 0:
            t_stat, p_value = stats.ttest_ind(male_all_data, female_all_data)
            print(f"\nOverall Comparison (All Time Points):")
            print(f"  Male: n={len(male_all_data)}, Mean={np.mean(male_all_data):.3f}, SEM={np.std(male_all_data)/np.sqrt(len(male_all_data)):.3f}")
            print(f"  Female: n={len(female_all_data)}, Mean={np.mean(female_all_data):.3f}, SEM={np.std(female_all_data)/np.sqrt(len(female_all_data)):.3f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.001:
                print(f"  Significance: *** (p < 0.001)")
            elif p_value < 0.01:
                print(f"  Significance: ** (p < 0.01)")
            elif p_value < 0.05:
                print(f"  Significance: * (p < 0.05)")
            else:
                print(f"  Significance: ns (not significant)")
        
        # Day-by-day comparison
        print(f"\nDay-by-Day Comparison:")
        print(f"  (Showing all comparisons with sample sizes and p-values)")
        for day in range(max_days):
            male_day = male_rewards_per_min[:, day]
            female_day = female_rewards_per_min[:, day]
            male_day = male_day[~np.isnan(male_day)]
            female_day = female_day[~np.isnan(female_day)]
            
            n_male = len(male_day)
            n_female = len(female_day)
            mean_male = np.mean(male_day) if n_male > 0 else np.nan
            mean_female = np.mean(female_day) if n_female > 0 else np.nan
            
            if n_male > 1 and n_female > 1:
                t_stat, p_value = stats.ttest_ind(male_day, female_day)
                
                sig_marker = ""
                if p_value < 0.001:
                    sig_marker = " ***"
                    sex_sig_days.append((day, t_stat, p_value))
                elif p_value < 0.01:
                    sig_marker = " **"
                    sex_sig_days.append((day, t_stat, p_value))
                elif p_value < 0.05:
                    sig_marker = " *"
                    sex_sig_days.append((day, t_stat, p_value))
                else:
                    sig_marker = " ns"
                
                print(f"  Day {day}: n_male={n_male}, n_female={n_female}, Mean_male={mean_male:.3f}, Mean_female={mean_female:.3f}, t={t_stat:.3f}, p={p_value:.4f}{sig_marker}")
            elif n_male > 0 or n_female > 0:
                print(f"  Day {day}: n_male={n_male}, n_female={n_female} - Insufficient data (need n>1 for both groups)")
        
        if not sex_sig_days:
            print(f"  No significant differences found at individual days (p < 0.05)")
    else:
        print("  Insufficient data for sex-based comparison (need both male and female data)")
    print("=" * 70 + "\n")
    
    # Add significance markers to sex-specific rewards plot
    if sex_sig_days:
        plt.figure(sex_reward_fig.number)
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        for day, t_stat, p_value in sex_sig_days:
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            plt.text(day, y_max * 0.95, sig_marker, ha='center', va='top', fontsize=12, fontweight='bold')

    # Create a new figure for condition-based analysis
    condition_reward_fig = plt.figure(figsize=(12, 6))
    
    # Group mice by starting condition
    condition_groups = {}
    for result in all_results:
        condition = result['starting_condition']
        if condition not in condition_groups:
            condition_groups[condition] = []
        rewards = np.array(result['hits'])
        session_lengths = np.array(result['session_lengths'])
        rewards_per_min = rewards / session_lengths
        condition_groups[condition].append(rewards_per_min)
    
    # Plot each condition's data
    for condition, rewards_list in condition_groups.items():
        color = condition_color_map[condition]
        # Pad arrays to make them equal length
        max_len = max(len(r) for r in rewards_list)
        padded_rewards = np.array([np.pad(r, (0, max_len - len(r)), 
                                        constant_values=np.nan) for r in rewards_list])
        
        # Calculate mean and SEM
        mean_rewards = np.nanmean(padded_rewards, axis=0)
        n_mice = np.sum(~np.isnan(padded_rewards), axis=0)
        sem_rewards = np.where(n_mice > 1,
                             np.nanstd(padded_rewards, axis=0) / np.sqrt(n_mice),
                             0)
        
        # Plot the data
        day_numbers = np.arange(0, max_len)
        plt.plot(day_numbers, mean_rewards, '-', color=color, linewidth=2, 
                label=f'{condition} (n={len(rewards_list)})')
        plt.fill_between(day_numbers, mean_rewards - sem_rewards, mean_rewards + sem_rewards,
                        color=color, alpha=0.2)
    
    # Configure condition-based rewards plot
    plt.title('Average Rewards Per Minute by Starting Condition')
    plt.xlabel('Training Day')
    plt.ylabel('Average Rewards per Minute')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_days - 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(agg_major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(agg_minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()
    
    # Statistical Analysis: Condition-based comparison (Rewards)
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: STARTING CONDITION COMPARISON - REWARDS")
    print("(Two-tailed t-test, pairwise comparisons)")
    print("=" * 70)
    
    condition_reward_sig_days = []  # Store significant days for plotting
    if len(condition_groups) > 1:
        # Perform pairwise comparisons between all conditions
        conditions_list = list(condition_groups.keys())
        
        # Overall comparisons
        for i in range(len(conditions_list)):
            for j in range(i + 1, len(conditions_list)):
                cond1 = conditions_list[i]
                cond2 = conditions_list[j]
                
                # Get all data points for each condition
                cond1_data = []
                for rewards in condition_groups[cond1]:
                    cond1_data.extend(rewards[~np.isnan(rewards)])
                cond2_data = []
                for rewards in condition_groups[cond2]:
                    cond2_data.extend(rewards[~np.isnan(rewards)])
                
                cond1_data = np.array(cond1_data)
                cond2_data = np.array(cond2_data)
                
                if len(cond1_data) > 0 and len(cond2_data) > 0:
                    t_stat, p_value = stats.ttest_ind(cond1_data, cond2_data)
                    print(f"\nOverall: {cond1} vs {cond2}:")
                    print(f"  {cond1}: n={len(cond1_data)}, Mean={np.mean(cond1_data):.3f}, SEM={np.std(cond1_data)/np.sqrt(len(cond1_data)):.3f}")
                    print(f"  {cond2}: n={len(cond2_data)}, Mean={np.mean(cond2_data):.3f}, SEM={np.std(cond2_data)/np.sqrt(len(cond2_data)):.3f}")
                    print(f"  t-statistic: {t_stat:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    if p_value < 0.001:
                        print(f"  Significance: *** (p < 0.001)")
                    elif p_value < 0.01:
                        print(f"  Significance: ** (p < 0.01)")
                    elif p_value < 0.05:
                        print(f"  Significance: * (p < 0.05)")
                    else:
                        print(f"  Significance: ns (not significant)")
        
        # Day-by-day comparisons
        print(f"\nDay-by-Day Comparisons:")
        print(f"  (Showing all comparisons with sample sizes and p-values)")
        # First find the global maximum length across all conditions
        global_max_len = max(max(len(r) for r in rewards_list) for rewards_list in condition_groups.values())
        
        # Pad all conditions to the same global maximum length
        condition_padded = {}
        for condition, rewards_list in condition_groups.items():
            condition_padded[condition] = np.array([np.pad(r, (0, global_max_len - len(r)), 
                                                    constant_values=np.nan) for r in rewards_list])
        
        has_any_comparison = False
        for day in range(global_max_len):
            day_sig = False
            day_has_test = False
            for i in range(len(conditions_list)):
                for j in range(i + 1, len(conditions_list)):
                    cond1 = conditions_list[i]
                    cond2 = conditions_list[j]
                    
                    cond1_day = condition_padded[cond1][:, day]
                    cond2_day = condition_padded[cond2][:, day]
                    
                    cond1_day = cond1_day[~np.isnan(cond1_day)]
                    cond2_day = cond2_day[~np.isnan(cond2_day)]
                    
                    n1 = len(cond1_day)
                    n2 = len(cond2_day)
                    
                    # Calculate means for display
                    mean1 = np.mean(cond1_day) if n1 > 0 else np.nan
                    mean2 = np.mean(cond2_day) if n2 > 0 else np.nan
                    
                    if n1 > 1 and n2 > 1:
                        # Can perform t-test
                        t_stat, p_value = stats.ttest_ind(cond1_day, cond2_day)
                        day_has_test = True
                        has_any_comparison = True
                        
                        sig_marker = ""
                        if p_value < 0.001:
                            sig_marker = " ***"
                            if not day_sig:
                                condition_reward_sig_days.append(day)
                                day_sig = True
                        elif p_value < 0.01:
                            sig_marker = " **"
                            if not day_sig:
                                condition_reward_sig_days.append(day)
                                day_sig = True
                        elif p_value < 0.05:
                            sig_marker = " *"
                            if not day_sig:
                                condition_reward_sig_days.append(day)
                                day_sig = True
                        else:
                            sig_marker = " ns"
                        
                        print(f"  Day {day} ({cond1} vs {cond2}): n1={n1}, n2={n2}, Mean1={mean1:.3f}, Mean2={mean2:.3f}, t={t_stat:.3f}, p={p_value:.4f}{sig_marker}")
                    elif n1 > 0 or n2 > 0:
                        # Insufficient sample size
                        print(f"  Day {day} ({cond1} vs {cond2}): n1={n1}, n2={n2} - Insufficient data (need n>1 for both groups)")
        
        if not has_any_comparison:
            print(f"  No day-by-day comparisons possible (insufficient data)")
        elif not condition_reward_sig_days:
            print(f"\n  Summary: No significant differences found at individual days (p < 0.05)")
    else:
        print("  Only one condition found - cannot perform comparison")
    print("=" * 70 + "\n")
    
    # Add significance markers to condition-based rewards plot
    if condition_reward_sig_days:
        plt.figure(condition_reward_fig.number)
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        for day in condition_reward_sig_days:
            plt.text(day, y_max * 0.95, '*', ha='center', va='top', fontsize=12, fontweight='bold')

    # Create a new figure for condition-based speed analysis
    condition_speed_fig = plt.figure(figsize=(12, 6))
    
    # Group mice by starting condition for speed
    condition_speed_groups = {}
    for result in all_results:
        condition = result['starting_condition']
        if condition not in condition_speed_groups:
            condition_speed_groups[condition] = []
        speeds = np.array(result['speeds'])
        condition_speed_groups[condition].append(speeds)
    
    # Plot each condition's speed data
    for condition, speed_list in condition_speed_groups.items():
        color = condition_color_map[condition]
        # Pad arrays to make them equal length
        max_len = max(len(s) for s in speed_list)
        padded_speeds = np.array([np.pad(s, (0, max_len - len(s)), 
                                        constant_values=np.nan) for s in speed_list])
        
        # Calculate mean and SEM
        mean_speeds = np.nanmean(padded_speeds, axis=0)
        n_mice = np.sum(~np.isnan(padded_speeds), axis=0)
        sem_speeds = np.where(n_mice > 1,
                             np.nanstd(padded_speeds, axis=0) / np.sqrt(n_mice),
                             0)
        
        # Plot the data
        day_numbers = np.arange(0, max_len)
        plt.plot(day_numbers, mean_speeds, '-', color=color, linewidth=2, 
                label=f'{condition} (n={len(speed_list)})')
        plt.fill_between(day_numbers, mean_speeds - sem_speeds, mean_speeds + sem_speeds,
                        color=color, alpha=0.2)
    
    # Configure condition-based speed plot
    plt.title('Average Speed by Starting Condition')
    plt.xlabel('Training Day')
    plt.ylabel('Average Speed (cm/s)')
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_days - 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(agg_major_spacing))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(agg_minor_spacing))
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.legend()
    
    # Statistical Analysis: Condition-based comparison (Speed)
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: STARTING CONDITION COMPARISON - SPEED")
    print("(Two-tailed t-test, pairwise comparisons)")
    print("=" * 70)
    
    condition_speed_sig_days = []  # Store significant days for plotting
    if len(condition_speed_groups) > 1:
        # Perform pairwise comparisons between all conditions
        conditions_list = list(condition_speed_groups.keys())
        
        # Overall comparisons
        for i in range(len(conditions_list)):
            for j in range(i + 1, len(conditions_list)):
                cond1 = conditions_list[i]
                cond2 = conditions_list[j]
                
                # Get all data points for each condition
                cond1_data = []
                for speeds in condition_speed_groups[cond1]:
                    cond1_data.extend(speeds[~np.isnan(speeds)])
                cond2_data = []
                for speeds in condition_speed_groups[cond2]:
                    cond2_data.extend(speeds[~np.isnan(speeds)])
                
                cond1_data = np.array(cond1_data)
                cond2_data = np.array(cond2_data)
                
                if len(cond1_data) > 0 and len(cond2_data) > 0:
                    t_stat, p_value = stats.ttest_ind(cond1_data, cond2_data)
                    print(f"\nOverall: {cond1} vs {cond2}:")
                    print(f"  {cond1}: n={len(cond1_data)}, Mean={np.mean(cond1_data):.3f} cm/s, SEM={np.std(cond1_data)/np.sqrt(len(cond1_data)):.3f}")
                    print(f"  {cond2}: n={len(cond2_data)}, Mean={np.mean(cond2_data):.3f} cm/s, SEM={np.std(cond2_data)/np.sqrt(len(cond2_data)):.3f}")
                    print(f"  t-statistic: {t_stat:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    if p_value < 0.001:
                        print(f"  Significance: *** (p < 0.001)")
                    elif p_value < 0.01:
                        print(f"  Significance: ** (p < 0.01)")
                    elif p_value < 0.05:
                        print(f"  Significance: * (p < 0.05)")
                    else:
                        print(f"  Significance: ns (not significant)")
        
        # Day-by-day comparisons
        print(f"\nDay-by-Day Comparisons:")
        print(f"  (Showing all comparisons with sample sizes and p-values)")
        # First find the global maximum length across all conditions
        global_max_len = max(max(len(s) for s in speed_list) for speed_list in condition_speed_groups.values())
        
        # Pad all conditions to the same global maximum length
        condition_speed_padded = {}
        for condition, speed_list in condition_speed_groups.items():
            condition_speed_padded[condition] = np.array([np.pad(s, (0, global_max_len - len(s)), 
                                                         constant_values=np.nan) for s in speed_list])
        
        has_any_comparison = False
        for day in range(global_max_len):
            day_sig = False
            day_has_test = False
            for i in range(len(conditions_list)):
                for j in range(i + 1, len(conditions_list)):
                    cond1 = conditions_list[i]
                    cond2 = conditions_list[j]
                    
                    cond1_day = condition_speed_padded[cond1][:, day]
                    cond2_day = condition_speed_padded[cond2][:, day]
                    
                    cond1_day = cond1_day[~np.isnan(cond1_day)]
                    cond2_day = cond2_day[~np.isnan(cond2_day)]
                    
                    n1 = len(cond1_day)
                    n2 = len(cond2_day)
                    
                    # Calculate means for display
                    mean1 = np.mean(cond1_day) if n1 > 0 else np.nan
                    mean2 = np.mean(cond2_day) if n2 > 0 else np.nan
                    
                    if n1 > 1 and n2 > 1:
                        # Can perform t-test
                        t_stat, p_value = stats.ttest_ind(cond1_day, cond2_day)
                        day_has_test = True
                        has_any_comparison = True
                        
                        sig_marker = ""
                        if p_value < 0.001:
                            sig_marker = " ***"
                            if not day_sig:
                                condition_speed_sig_days.append(day)
                                day_sig = True
                        elif p_value < 0.01:
                            sig_marker = " **"
                            if not day_sig:
                                condition_speed_sig_days.append(day)
                                day_sig = True
                        elif p_value < 0.05:
                            sig_marker = " *"
                            if not day_sig:
                                condition_speed_sig_days.append(day)
                                day_sig = True
                        else:
                            sig_marker = " ns"
                        
                        print(f"  Day {day} ({cond1} vs {cond2}): n1={n1}, n2={n2}, Mean1={mean1:.3f} cm/s, Mean2={mean2:.3f} cm/s, t={t_stat:.3f}, p={p_value:.4f}{sig_marker}")
                    elif n1 > 0 or n2 > 0:
                        # Insufficient sample size
                        print(f"  Day {day} ({cond1} vs {cond2}): n1={n1}, n2={n2} - Insufficient data (need n>1 for both groups)")
        
        if not has_any_comparison:
            print(f"  No day-by-day comparisons possible (insufficient data)")
        elif not condition_speed_sig_days:
            print(f"\n  Summary: No significant differences found at individual days (p < 0.05)")
    else:
        print("  Only one condition found - cannot perform comparison")
    print("=" * 70 + "\n")
    
    # Add significance markers to condition-based speed plot
    if condition_speed_sig_days:
        plt.figure(condition_speed_fig.number)
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        for day in condition_speed_sig_days:
            plt.text(day, y_max * 0.95, '*', ha='center', va='top', fontsize=12, fontweight='bold')

    # Create the level-based analysis plot
    level_fig = analyze_levels(data_files)

    return speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, condition_reward_fig, condition_speed_fig, level_fig, all_results

def main():
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select the master CSV file
    master_csv_path = filedialog.askopenfilename(
        title='Select master CSV file (containing animal_id, sex, starting_condition)',
        filetypes=[('CSV files', '*.csv')],
        initialdir=os.getcwd()
    )
    
    if not master_csv_path:
        print("No master CSV file selected. Exiting...")
        return
    
    # Read the master CSV file
    try:
        master_df = pd.read_csv(master_csv_path)
        # Strip whitespace from column names and values
        master_df.columns = master_df.columns.str.strip()
        master_df['animal_id'] = master_df['animal_id'].str.strip()
        master_df['sex'] = master_df['sex'].str.strip().str.lower()
        master_df['starting_condition'] = master_df['starting_condition'].str.strip()
        
        # Create a dictionary mapping animal_id to sex and starting_condition
        animal_info = {}
        for _, row in master_df.iterrows():
            animal_info[row['animal_id']] = {
                'sex': row['sex'],
                'starting_condition': row['starting_condition']
            }
        
        print(f"Loaded master CSV with {len(animal_info)} animals")
        
    except Exception as e:
        print(f"Error reading master CSV file: {str(e)}")
        return

    # Open file dialog to select multiple data files
    file_paths = filedialog.askopenfilenames(
        title='Select mouse data files',
        filetypes=[('CSV files', '*.csv')],
        initialdir=os.getcwd()  # Start in current directory
    )
    
    if file_paths:
        # Extract markers and starting conditions from master CSV
        markers = []
        starting_conditions = []
        for file_path in file_paths:
            mouse_name = os.path.basename(file_path).split("_")[0]
            
            if mouse_name in animal_info:
                # Convert sex to marker type (male -> 's' for square, female -> 'o' for circle)
                sex = animal_info[mouse_name]['sex']
                marker = 's' if sex == 'male' else 'o'
                markers.append(marker)
                
                # Get starting condition
                starting_conditions.append(animal_info[mouse_name]['starting_condition'])
                
                print(f"{mouse_name}: sex={sex}, marker={marker}, condition={animal_info[mouse_name]['starting_condition']}")
            else:
                print(f"Warning: {mouse_name} not found in master CSV file. Skipping...")
                continue
        
        # Ask if user wants to save lick detection plots
       # print("\n" + "=" * 60)
        save_lick_plots_input = input("Generate lick detection plots for each session? (yes/no): ").lower().strip()
        save_lick_plots = save_lick_plots_input.startswith('y')
        
        output_dir = None
        if save_lick_plots:
            # Ask for output directory
            output_dir = filedialog.askdirectory(
                title='Select folder to save lick detection plots',
                initialdir=os.getcwd()
            )
            if not output_dir:
                print("No output directory selected. Lick detection plots will not be saved.")
                save_lick_plots = False
            else:
                print(f"Lick detection plots will be saved to: {output_dir}")
        #print("=" * 60 + "\n")
            
        # Analyze data and plot results
        speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, condition_reward_fig, condition_speed_fig, level_fig, all_results = analyze_mouse_data(
            file_paths, markers, starting_conditions, 
            save_lick_plots=save_lick_plots, 
            output_dir=output_dir
        )

        # Configure all figures
        for fig in [speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, condition_reward_fig, condition_speed_fig, level_fig]:
            plt.figure(fig.number)
            if len(file_paths) > 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.85)
            else:
                plt.legend()
            plt.tight_layout()

        # Display all plots
        for fig in [speed_fig, sensitivity_fig, lick_fig, reward_fig, avg_reward_fig, sex_reward_fig, condition_reward_fig, condition_speed_fig]:
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
                (sex_reward_fig, 'sex_reward', 'Sex-specific average rewards plot'),
                (condition_reward_fig, 'condition_reward', 'Condition-based average rewards plot'),
                (condition_speed_fig, 'condition_speed', 'Condition-based average speed plot'),
                (level_fig, 'level_reward', 'Level-based average rewards plot')
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