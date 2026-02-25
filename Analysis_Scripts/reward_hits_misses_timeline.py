"""
Timeline plot showing treadmill speed with reward hits and misses

This script creates a single plot showing:
- Interpolated treadmill speed across the session
- Green vertical lines for reward zone entries that resulted in reward delivery (hits)
- Purple vertical lines for reward zone entries without reward delivery (misses)

Author: Brenna Manuel
Created: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import ast

# Configure matplotlib for SVG output with editable text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def safe_literal_eval(val):
    """Safely evaluate string representations of lists"""
    try:
        if isinstance(val, list):
            return val
        if pd.isna(val) or val == '':
            return []
        if isinstance(val, (int, float)):
            return [val]
        if isinstance(val, str) and not (val.strip().startswith("[") and val.strip().endswith("]")):
            try:
                return [float(val)]
            except Exception:
                return [val]
        return ast.literal_eval(val)
    except Exception:
        return []


def select_data_folder():
    """Open a file dialog to select the data folder"""
    root = tk.Tk()
    root.withdraw()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_dir = os.path.dirname(script_dir)
    
    folder_path = filedialog.askdirectory(
        title="Select folder containing behavioral data files",
        initialdir=initial_dir
    )
    
    return folder_path if folder_path else None


def load_data_files(folder_path):
    """Load trial log, capacitive, and treadmill CSV files"""
    trial_log_files = [f for f in os.listdir(folder_path) if 'trial_log.csv' in f]
    treadmill_files = [f for f in os.listdir(folder_path) if 'treadmill.csv' in f]
    capacitive_files = [f for f in os.listdir(folder_path) if 'capacitive.csv' in f]
    
    if not trial_log_files or not treadmill_files or not capacitive_files:
        print("Error: Missing required files (trial_log.csv, treadmill.csv, or capacitive.csv)")
        return None
    
    print(f"\nLoading files:")
    print(f"  - {trial_log_files[0]}")
    print(f"  - {treadmill_files[0]}")
    print(f"  - {capacitive_files[0]}")
    
    data = {
        'trial_log': pd.read_csv(os.path.join(folder_path, trial_log_files[0]), engine='python'),
        'treadmill': pd.read_csv(os.path.join(folder_path, treadmill_files[0]), comment='/', engine='python'),
        'capacitive': pd.read_csv(os.path.join(folder_path, capacitive_files[0]), comment='/', engine='python')
    }
    
    print("Files loaded successfully.\n")
    return data


def interpolate_treadmill_speed(treadmill_df, capacitive_df):
    """Interpolate treadmill speed to capacitive timeline and convert mm/s to cm/s"""
    return pd.Series(
        data=np.interp(
            capacitive_df['elapsed_time'],
            treadmill_df['global_time'],
            treadmill_df['speed']
        ) / 10.0,  # Convert mm/s to cm/s
        index=capacitive_df['elapsed_time']
    )


def extract_reward_zone_entries(trial_log_df):
    """Extract all reward zone entry times from texture history"""
    all_reward_zones = []
    
    for trial_idx in range(len(trial_log_df)):
        texture_hist = safe_literal_eval(trial_log_df.loc[trial_idx, 'texture_history'])
        texture_times = safe_literal_eval(trial_log_df.loc[trial_idx, 'texture_change_time'])
        
        if not texture_hist or not texture_times:
            continue
        
        for i, texture in enumerate(texture_hist):
            if texture == "assets/reward_mean100.jpg" and i < len(texture_times):
                zone_entry_time = texture_times[i]
                if pd.notna(zone_entry_time) and zone_entry_time > 0:
                    all_reward_zones.append((trial_idx, zone_entry_time))
    
    all_reward_zones.sort(key=lambda x: x[1])
    return all_reward_zones


def extract_reward_events(trial_log_df):
    """Extract all reward delivery events"""
    reward_events = []
    
    for trial_idx in range(len(trial_log_df)):
        reward_time = pd.to_numeric(trial_log_df.loc[trial_idx, 'reward_event'], errors='coerce')
        if pd.notna(reward_time) and reward_time > 0:
            reward_events.append((trial_idx, reward_time))
    
    reward_events.sort(key=lambda x: x[1])
    return reward_events


def classify_hits_and_misses(all_reward_zones, reward_events, match_window=10.0):
    """
    Classify reward zones as hits or misses
    
    Hit: Reward zone followed by a reward event within match_window seconds
    Miss: Reward zone NOT followed by a reward event
    
    Args:
        all_reward_zones: List of (trial_idx, zone_entry_time) tuples
        reward_events: List of (trial_idx, reward_event_time) tuples
        match_window: Maximum time between zone entry and reward for a match
    
    Returns:
        Tuple of (hits, misses) where each is a list of zone_entry_times
    """
    hits = []
    misses = []
    
    reward_event_times = [r_time for _, r_time in reward_events]
    
    for trial_idx, zone_entry_time in all_reward_zones:
        # Find reward events that occur after this zone entry
        matching_rewards = [
            r_time for r_time in reward_event_times
            if zone_entry_time <= r_time <= (zone_entry_time + match_window)
        ]
        
        if matching_rewards:
            # This zone led to a reward (hit)
            hits.append(zone_entry_time)
        else:
            # This zone did not lead to a reward (miss)
            misses.append(zone_entry_time)
    
    return hits, misses


def plot_speed_with_hits_misses(capacitive_df, treadmill_interp, hits, misses, output_folder):
    """Create timeline plot showing hit/miss pattern"""
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Get time range from capacitive data
    time_min = capacitive_df['elapsed_time'].min()
    time_max = capacitive_df['elapsed_time'].max()
    
    # Add vertical lines for hits (green)
    for i, hit_time in enumerate(hits):
        ax.axvline(x=hit_time, color='green', alpha=0.7, linewidth=2,
                   label='Hit (Reward Zone → Reward)' if i == 0 else None)
    
    # Add vertical lines for misses (purple)
    for i, miss_time in enumerate(misses):
        ax.axvline(x=miss_time, color='purple', alpha=0.7, linewidth=2,
                   label='Miss (Reward Zone, No Reward)' if i == 0 else None)
    
    # Format plot
    ax.set_xlabel('Elapsed Time (s)', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Reward Hits and Misses Pattern Across Session', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(0, 1)
    
    # Remove y-axis ticks and labels since we're just showing events
    ax.set_yticks([])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, "reward_hits_misses_timeline.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")
    
    plt.show()


def main():
    """Main execution function"""
    print("=" * 60)
    print("REWARD HITS AND MISSES TIMELINE")
    print("=" * 60)
    
    # Select folder
    folder_path = select_data_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    # Load data
    data = load_data_files(folder_path)
    if data is None:
        return
    
    # Create output folder
    output_folder = os.path.join(folder_path, "svg_plots")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Interpolate treadmill speed
    print("Interpolating treadmill speed to capacitive timeline...")
    treadmill_interp = interpolate_treadmill_speed(data['treadmill'], data['capacitive'])
    
    # Extract reward zones and events
    print("Extracting reward zone entries...")
    all_reward_zones = extract_reward_zone_entries(data['trial_log'])
    print(f"Found {len(all_reward_zones)} reward zone entries")
    
    print("Extracting reward events...")
    reward_events = extract_reward_events(data['trial_log'])
    print(f"Found {len(reward_events)} reward delivery events")
    
    # Classify hits and misses
    print("\nClassifying hits and misses...")
    hits, misses = classify_hits_and_misses(all_reward_zones, reward_events)
    print(f"  Hits (zones with rewards): {len(hits)}")
    print(f"  Misses (zones without rewards): {len(misses)}")
    
    # Create plot
    print("\nGenerating timeline plot...")
    plot_speed_with_hits_misses(data['capacitive'], treadmill_interp, hits, misses, output_folder)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
