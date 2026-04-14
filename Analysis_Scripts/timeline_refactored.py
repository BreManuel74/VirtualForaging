"""
Timeline analysis script for behavioral data (REFACTORED VERSION)

This script reconstructs and visualizes multiple data streams over time, including:
- Capacitive sensor data
- Treadmill speed
- Pupil diameter (if available)

Generates plots and saves them as SVG files for high-quality visualization.

Original Author: Brenna Manuel
Refactored: February 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
from tkinter import filedialog
import tkinter as tk
import matplotlib.cm as cm
from scipy.stats import ttest_rel, ttest_ind, kurtosis, skew
from sklearn.mixture import GaussianMixture

# Configure matplotlib for SVG output with editable text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'  # Save text as actual text, not paths
plt.rcParams['xtick.direction'] = 'in'  # Tick marks face inward
plt.rcParams['ytick.direction'] = 'in'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_literal_eval(val):
    """Safely evaluate string representations of lists
    
    Args:
        val: Value to evaluate
        
    Returns:
        Evaluated list or empty list if evaluation fails
    """
    try:
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, (int, float)):
            # Single numeric value, wrap in list
            return [val]
        if isinstance(val, str):
            if val.strip() == '' or val.strip() == '[]':
                return []
            # Check if it's a plain string (not a list representation)
            if not (val.strip().startswith('[') or val.strip().startswith('(')):
                # It's a plain string or number, wrap it in a list
                return [val]
            try:
                result = ast.literal_eval(val)
                if isinstance(result, list):
                    return result
                elif isinstance(result, tuple):
                    return list(result)
                elif isinstance(result, (int, float, str)):
                    # Single value that was evaluated, wrap in list
                    return [result]
                return []
            except (ValueError, SyntaxError):
                # If literal_eval fails, treat as plain string
                return [val]
        return []
    except Exception:
        return []


def pad_list(lst, length):
    """Pad a list to a specified length with None values"""
    return lst + [None] * (length - len(lst))


def save_figure(fig, name, output_folder):
    """Save a figure as an SVG file in the output folder
    
    Args:
        fig: The matplotlib figure to save
        name: Base name for the file (without extension)
        output_folder: Directory to save the figure
    """
    if not hasattr(save_figure, 'figure_count'):
        save_figure.figure_count = 1
    
    filename = f"{name}_{save_figure.figure_count}.svg"
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, format="svg", bbox_inches="tight")
    print(f"Saved figure: {filename}")
    save_figure.figure_count += 1
    return filepath


def setup_plot_style(ax):
    """Remove top and right borders from plot axes
    
    Args:
        ax: Matplotlib axis or list of axes
    """
    if isinstance(ax, list):
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# ============================================================================
# FILE LOADING FUNCTIONS
# ============================================================================

def select_data_folder():
    """Open a file dialog to select the data folder
    
    Returns:
        str: Path to the selected folder or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_dir = os.path.dirname(script_dir)
    
    folder_path = filedialog.askdirectory(
        title="Select folder containing behavioral data files",
        initialdir=initial_dir
    )
    
    return folder_path if folder_path else None


def validate_and_find_files(folder_path):
    """Find and validate required data files in the folder
    
    Args:
        folder_path: Path to the data folder
        
    Returns:
        tuple: (file_paths dict, has_pupil_data bool) or (None, False) if validation fails
    """
    trial_log_files = [f for f in os.listdir(folder_path) if 'trial_log.csv' in f]
    treadmill_files = [f for f in os.listdir(folder_path) if 'treadmill.csv' in f]
    capacitive_files = [f for f in os.listdir(folder_path) if 'capacitive.csv' in f]
    pupil_files = [f for f in os.listdir(folder_path) if 'exposure.csv' in f]
    frame_log_files = [f for f in os.listdir(folder_path) if 'frame_log.txt' in f]
    
    # Check required files
    missing_types = []
    if not trial_log_files:
        missing_types.append("trial_log.csv")
    if not treadmill_files:
        missing_types.append("treadmill.csv")
    if not capacitive_files:
        missing_types.append("capacitive.csv")
    
    if missing_types:
        print(f"Warning: Missing required file types: {missing_types}")
        print("Please ensure all three required file types are present.")
        return None, False
    
    # Check optional pupil files
    has_pupil_data = len(pupil_files) > 0 and len(frame_log_files) > 0
    
    if has_pupil_data:
        print(f"Pupil data file found: {pupil_files[0]}")
        print(f"Frame log file found: {frame_log_files[0]}")
    else:
        print("Pupil data not found. Analysis will proceed without pupil data.")
    
    file_paths = {
        'trial_log': os.path.join(folder_path, trial_log_files[0]),
        'treadmill': os.path.join(folder_path, treadmill_files[0]),
        'capacitive': os.path.join(folder_path, capacitive_files[0]),
        'pupil': os.path.join(folder_path, pupil_files[0]) if has_pupil_data else None,
        'frame_log': os.path.join(folder_path, frame_log_files[0]) if has_pupil_data else None
    }
    
    return file_paths, has_pupil_data


def load_data_files(file_paths, has_pupil_data):
    """Load all required CSV files into pandas DataFrames
    
    Args:
        file_paths: Dictionary of file paths
        has_pupil_data: Whether pupil data is available
        
    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    print(f"\nLoading files:")
    print(f"  - {os.path.basename(file_paths['trial_log'])}")
    print(f"  - {os.path.basename(file_paths['treadmill'])}")
    print(f"  - {os.path.basename(file_paths['capacitive'])}")
    
    data = {
        'trial_log': pd.read_csv(file_paths['trial_log'], engine='python'),
        'treadmill': pd.read_csv(file_paths['treadmill'], comment='/', engine='python'),
        'capacitive': pd.read_csv(file_paths['capacitive'], comment='/', engine='python')
    }
    
    if has_pupil_data:
        data['pupil'] = pd.read_csv(file_paths['pupil'], comment='/', engine='python', skiprows=3)
        data['frame_log'] = pd.read_csv(file_paths['frame_log'], sep='\t', engine='python')
    else:
        data['pupil'] = None
        data['frame_log'] = None
    
    print("Files loaded successfully.\n")
    return data


def create_output_folder(folder_path):
    """Create an output folder for SVG files
    
    Args:
        folder_path: Path to the data folder
        
    Returns:
        str: Path to the output folder
    """
    output_folder = os.path.join(folder_path, "svg_plots")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory for SVG files: {output_folder}")
    else:
        print(f"Using existing directory for SVG files: {output_folder}")
    return output_folder


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def process_texture_history(trial_log_df):
    """Process texture history data from trial log
    
    Args:
        trial_log_df: Trial log DataFrame
        
    Returns:
        dict: Dictionary containing processed texture arrays and metadata
    """
    # Check if required columns exist
    required_cols = ['texture_history', 'texture_change_time', 'texture_revert']
    missing_cols = [col for col in required_cols if col not in trial_log_df.columns]
    
    if missing_cols:
        print(f"Warning: Missing texture columns: {missing_cols}. Creating empty arrays.")
        return create_empty_texture_arrays()
    
    # Parse list columns
    texture_history = trial_log_df['texture_history'].apply(safe_literal_eval)
    texture_change_time = trial_log_df['texture_change_time'].apply(safe_literal_eval)
    revert_time = trial_log_df['texture_revert'].apply(safe_literal_eval)
    
    # Check if there are any texture changes (handle NaN and empty cases)
    try:
        max_hist_len = texture_history.apply(len).max()
        if pd.isna(max_hist_len):
            max_hist_len = 0
    except (ValueError, TypeError):
        max_hist_len = 0
    
    try:
        max_change_len = texture_change_time.apply(len).max()
        if pd.isna(max_change_len):
            max_change_len = 0
    except (ValueError, TypeError):
        max_change_len = 0
    
    try:
        max_revert_len = revert_time.apply(len).max()
        if pd.isna(max_revert_len):
            max_revert_len = 0
    except (ValueError, TypeError):
        max_revert_len = 0
    
    has_texture_data = (max_hist_len > 0 or max_change_len > 0 or max_revert_len > 0)
    
    if not has_texture_data:
        print("Warning: No texture change data found. Creating empty arrays.")
        return create_empty_texture_arrays()
    
    # Pad arrays to same length
    max_len = max(
        texture_history.apply(len).max(),
        texture_change_time.apply(len).max(),
        revert_time.apply(len).max()
    )
    
    texture_history_padded = np.array(texture_history.apply(lambda x: pad_list(x, max_len)).tolist())
    texture_change_time_padded = np.array(texture_change_time.apply(lambda x: pad_list(x, max_len)).tolist())
    revert_time_padded = np.array(revert_time.apply(lambda x: pad_list(x, max_len)).tolist())
    
    combined_array = np.stack(
        [texture_history_padded, texture_change_time_padded, revert_time_padded],
        axis=1
    )
    
    # Create boolean masks for each asset type
    is_punish = texture_history_padded[:, 0] == "assets/punish_mean100.jpg"
    is_reward = texture_history_padded[:, 0] == "assets/reward_mean100.jpg"
    
    # Select rows for each type
    punish_array = combined_array[is_punish]
    reward_array = combined_array[is_reward]
    
    # Extract timing information
    punish_texture_change_time = punish_array[:, 1, :] if punish_array.shape[0] > 0 else np.empty((0, 1))
    punish_revert_time = punish_array[:, 2, :] if punish_array.shape[0] > 0 else np.empty((0, 1))
    
    reward_texture_change_time = reward_array[:, 1, :] if reward_array.shape[0] > 0 else np.empty((0, 1))
    reward_revert_time = reward_array[:, 2, :] if reward_array.shape[0] > 0 else np.empty((0, 1))
    
    # Create versions using only first puff per zone
    if punish_array.shape[0] > 0 and punish_array.shape[2] > 0:
        punish_texture_change_time_first = punish_array[:, 1, 0]
        punish_revert_time_first = punish_array[:, 2, 0]
    else:
        punish_texture_change_time_first = np.array([])
        punish_revert_time_first = np.array([])
    
    return {
        'has_texture_data': has_texture_data,
        'texture_history_padded': texture_history_padded,
        'texture_change_time_padded': texture_change_time_padded,
        'revert_time_padded': revert_time_padded,
        'punish_array': punish_array,
        'reward_array': reward_array,
        'punish_texture_change_time': punish_texture_change_time,
        'punish_revert_time': punish_revert_time,
        'punish_texture_change_time_first': punish_texture_change_time_first,
        'punish_revert_time_first': punish_revert_time_first,
        'reward_texture_change_time': reward_texture_change_time,
        'reward_revert_time': reward_revert_time
    }


def create_empty_texture_arrays():
    """Create empty texture arrays when no texture data is present
    
    Returns:
        dict: Dictionary with empty texture arrays
    """
    return {
        'has_texture_data': False,
        'texture_history_padded': np.empty((0, 1)),
        'texture_change_time_padded': np.empty((0, 1)),
        'revert_time_padded': np.empty((0, 1)),
        'punish_array': np.empty((0, 3, 1)),
        'reward_array': np.empty((0, 3, 1)),
        'punish_texture_change_time': np.empty((0, 1)),
        'punish_revert_time': np.empty((0, 1)),
        'punish_texture_change_time_first': np.array([]),
        'punish_revert_time_first': np.array([]),
        'reward_texture_change_time': np.empty((0, 1)),
        'reward_revert_time': np.empty((0, 1))
    }


def uniformly_sample_treadmill(treadmill_df):
    """Uniformly sample treadmill data at 50 Hz
    
    Args:
        treadmill_df: Treadmill DataFrame with 'global_time' and 'speed' columns
        
    Returns:
        tuple: (uniform_time, uniform_speed_cm_s)
            - uniform_time: np.array of uniformly sampled times at 50 Hz
            - uniform_speed_cm_s: np.array of speed values in cm/s
    """
    # Define sampling rate
    sampling_rate = 50.0  # Hz
    sampling_interval = 1.0 / sampling_rate  # 0.02 seconds
    
    # Get time range from treadmill data
    time_min = treadmill_df['global_time'].min()
    time_max = treadmill_df['global_time'].max()
    
    # Create uniform time array
    uniform_time = np.arange(time_min, time_max, sampling_interval)
    
    # Interpolate speed to uniform time points and convert to cm/s
    uniform_speed = np.interp(
        uniform_time,
        treadmill_df['global_time'].values,
        treadmill_df['speed'].values
    ) / 10.0  # Convert mm/s to cm/s
    
    return uniform_time, uniform_speed


def uniformly_sample_treadmill_distance(treadmill_df):
    """Uniformly sample treadmill distance at 50 Hz
    
    Finds the first non-zero distance value and subtracts it from all distances
    to get the distance moved in meters.
    
    Args:
        treadmill_df: Treadmill DataFrame with 'global_time' and 'distance' columns
        
    Returns:
        tuple: (uniform_time, uniform_distance_m)
            - uniform_time: np.array of uniformly sampled times at 50 Hz
            - uniform_distance_m: np.array of distance moved in meters
    """
    # Define sampling rate
    sampling_rate = 50.0  # Hz
    sampling_interval = 1.0 / sampling_rate  # 0.02 seconds
    
    # Find the first non-zero distance value
    non_zero_distances = treadmill_df['distance'][treadmill_df['distance'] != 0]
    
    if len(non_zero_distances) > 0:
        start_distance = non_zero_distances.iloc[0]
    else:
        # If all distances are zero, use 0 as start distance
        start_distance = 0
    
    # Create adjusted distance values (distance moved from start)
    distance_moved = treadmill_df['distance'] - start_distance
    
    # Get time range from treadmill data
    time_min = treadmill_df['global_time'].min()
    time_max = treadmill_df['global_time'].max()
    
    # Create uniform time array
    uniform_time = np.arange(time_min, time_max, sampling_interval)
    
    # Interpolate distance to uniform time points and convert to meters
    uniform_distance = np.interp(
        uniform_time,
        treadmill_df['global_time'].values,
        distance_moved.values
    ) / 1000.0  # Convert mm to meters
    
    return uniform_time, uniform_distance


def uniformly_sample_capacitive(capacitive_df):
    """Uniformly sample capacitive data at 50 Hz
    
    Args:
        capacitive_df: Capacitive DataFrame with 'elapsed_time' and 'capacitive_value' columns
        
    Returns:
        tuple: (uniform_time, uniform_capacitive)
            - uniform_time: np.array of uniformly sampled times at 50 Hz
            - uniform_capacitive: np.array of capacitive sensor values
    """
    # Define sampling rate
    sampling_rate = 50.0  # Hz
    sampling_interval = 1.0 / sampling_rate  # 0.02 seconds
    
    # Get time range from capacitive data
    time_min = capacitive_df['elapsed_time'].min()
    time_max = capacitive_df['elapsed_time'].max()
    
    # Create uniform time array
    uniform_time = np.arange(time_min, time_max, sampling_interval)
    
    # Interpolate capacitive values to uniform time points
    uniform_capacitive = np.interp(
        uniform_time,
        capacitive_df['elapsed_time'].values,
        capacitive_df['capacitive_value'].values
    )
    
    return uniform_time, uniform_capacitive


def process_pupil_data(pupil_df, frame_log_df):
    """Process pupil diameter data and uniformly sample at 20 Hz
    
    Args:
        pupil_df: Pupil tracking DataFrame
        frame_log_df: Frame log DataFrame with timestamps
        
    Returns:
        tuple: (uniform_time, uniform_pupil_zscore) or (None, None) if processing fails
            - uniform_time: np.array of uniformly sampled times at 20 Hz
            - uniform_pupil_zscore: np.array of z-scored pupil diameter values
    """
    if pupil_df is None or frame_log_df is None:
        return None, None
    
    # Rename columns if needed (pupil CSV has generic column names after skipping header rows)
    if pupil_df.columns[0] != 'frame_number':
        pupil_df_columns = pupil_df.columns.tolist()
        pupil_df_columns[0] = 'frame_number'
        pupil_df_columns[7] = 'point_3_x'
        pupil_df_columns[8] = 'point_3_y'
        pupil_df_columns[9] = 'point_3_likelihood'
        pupil_df_columns[19] = 'point_7_x'
        pupil_df_columns[20] = 'point_7_y'
        pupil_df_columns[21] = 'point_7_likelihood'
        pupil_df.columns = pupil_df_columns
    
    # Align frame numbers and map timestamps
    frame_to_time_mapping = dict(zip(frame_log_df['frame_number'], frame_log_df['time_seconds']))
    pupil_df['aligned_frame_number'] = pupil_df['frame_number'] + 1
    pupil_df['time_seconds'] = pupil_df['aligned_frame_number'].map(frame_to_time_mapping)
    
    # Calculate pupil diameter (only for high likelihood points)
    high_likelihood_mask = (pupil_df['point_3_likelihood'] >= 0.80) & (pupil_df['point_7_likelihood'] >= 0.80)
    pupil_df['pupil_diameter'] = np.where(
        high_likelihood_mask,
        np.sqrt((pupil_df['point_7_x'] - pupil_df['point_3_x'])**2 + 
                (pupil_df['point_7_y'] - pupil_df['point_3_y'])**2),
        np.nan
    )
    
    # Get valid data
    valid_data_mask = pupil_df['time_seconds'].notna() & pupil_df['pupil_diameter'].notna()
    
    if valid_data_mask.sum() > 1:
        valid_times = pupil_df.loc[valid_data_mask, 'time_seconds'].values
        valid_diameters = pupil_df.loc[valid_data_mask, 'pupil_diameter'].values
        
        # CRITICAL: Sort pupil data by timestamp (required for np.interp)
        sort_indices = np.argsort(valid_times)
        valid_times = valid_times[sort_indices]
        valid_diameters = valid_diameters[sort_indices]
        
        # Z-score the pupil data (normalize to session mean and std)
        pupil_mean = np.nanmean(valid_diameters)
        pupil_std = np.nanstd(valid_diameters)
        valid_diameters_zscore = (valid_diameters - pupil_mean) / pupil_std
        
        # Define sampling rate for pupil (20 fps)
        sampling_rate = 20.0  # Hz
        sampling_interval = 1.0 / sampling_rate  # 0.05 seconds
        
        # Get time range
        time_min = valid_times.min()
        time_max = valid_times.max()
        
        # Create uniform time array
        uniform_time = np.arange(time_min, time_max, sampling_interval)
        
        # Interpolate to uniform time points
        uniform_pupil_zscore = np.interp(
            uniform_time,
            valid_times,
            valid_diameters_zscore
        )
        
        print(f"Pupil data processed: {valid_data_mask.sum()} valid measurements")
        print(f"Uniformly sampled at 20 Hz: {len(uniform_time)} samples")
        print(f"Pupil z-score normalization: mean={pupil_mean:.2f} pixels, std={pupil_std:.2f} pixels")
        return uniform_time, uniform_pupil_zscore
    else:
        print("Warning: Insufficient valid pupil data for interpolation")
        return None, None


# ============================================================================
# EVENT MATCHING FUNCTIONS
# ============================================================================

def match_reward_zones_to_events(trial_log_df, reward_texture_change_time):
    """Match reward zone entries to reward delivery events using temporal proximity
    
    Args:
        trial_log_df: Trial log DataFrame
        reward_texture_change_time: Array of reward zone entry times
        
    Returns:
        list: List of (trial_idx, zone_entry_time, reward_event_time) tuples
    """
    if reward_texture_change_time.size == 0:
        return []
    
    print("\n=== MATCHING REWARD ZONES TO DELIVERY EVENTS ===")
    
    # Collect all reward zone entries with verified texture history
    all_reward_zones = []
    for trial_idx in range(len(trial_log_df)):
        texture_hist = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_history'])
        texture_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_change_time'])
        
        for i, texture in enumerate(texture_hist):
            if texture == "assets/reward_mean100.jpg" and i < len(texture_times):
                zone_entry_time = texture_times[i]
                if pd.notna(zone_entry_time) and zone_entry_time > 0:
                    all_reward_zones.append((trial_idx, zone_entry_time))
    
    print(f"Found {len(all_reward_zones)} reward zone entries")
    
    # Collect all reward events
    reward_events = []
    for trial_idx in range(len(trial_log_df)):
        reward_event = trial_log_df.iloc[trial_idx]['reward_event']
        if pd.notna(reward_event) and reward_event > 0:
            reward_events.append((trial_idx, reward_event))
    
    print(f"Found {len(reward_events)} reward delivery events")
    
    # Sort by timestamp
    all_reward_zones.sort(key=lambda x: x[1])
    reward_events.sort(key=lambda x: x[1])
    
    # Match each reward event to the most recent preceding zone entry
    reward_zone_trials = []
    matched_zones = set()
    
    for reward_trial_idx, reward_event_time in reward_events:
        best_match = None
        best_time_diff = float('inf')
        
        for i, (zone_trial_idx, zone_entry_time) in enumerate(all_reward_zones):
            if i in matched_zones:
                continue
            if zone_entry_time <= reward_event_time:
                time_diff = reward_event_time - zone_entry_time
                if time_diff < best_time_diff and time_diff < 10.0:  # Within 10 seconds
                    best_match = i
                    best_time_diff = time_diff
        
        if best_match is not None:
            zone_trial_idx, zone_entry_time = all_reward_zones[best_match]
            reward_zone_trials.append((zone_trial_idx, zone_entry_time, reward_event_time))
            matched_zones.add(best_match)
    
    # Add unmatched zones with NaN reward times
    for i, (zone_trial_idx, zone_entry_time) in enumerate(all_reward_zones):
        if i not in matched_zones:
            reward_zone_trials.append((zone_trial_idx, zone_entry_time, np.nan))
    
    reward_zone_trials.sort(key=lambda x: x[1])
    
    valid_deliveries = sum(1 for _, _, r in reward_zone_trials if pd.notna(r) and r > 0)
    print(f"Successfully matched {valid_deliveries}/{len(reward_zone_trials)} reward zones to deliveries")
    print("=== END REWARD ZONE MATCHING ===\n")
    
    return reward_zone_trials


def match_puff_zones_to_events(trial_log_df, punish_texture_change_time_first):
    """Match puff zone entries to puff delivery events using temporal proximity
    
    Args:
        trial_log_df: Trial log DataFrame
        punish_texture_change_time_first: Array of first puff zone entry times
        
    Returns:
        list: List of (trial_idx, zone_entry_time, puff_event_time) tuples
    """
    if 'puff_event' not in trial_log_df.columns or len(trial_log_df) == 0:
        return []
    
    print("\n=== MATCHING PUFF ZONES TO DELIVERY EVENTS ===")
    
    # Collect all puff zone entries using FIRST puff per zone from pre-computed array
    all_puff_zones = []
    for trial_idx in range(len(trial_log_df)):
        texture_hist = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_history'])
        
        # Only include trials with punish texture
        if len(texture_hist) > 0 and texture_hist[0] == "assets/punish_mean100.jpg":
            # Use the corresponding entry from punish_texture_change_time_first
            zone_entry_time = punish_texture_change_time_first[len(all_puff_zones)]
            if not pd.isna(zone_entry_time) and zone_entry_time != '':
                try:
                    zone_entry_time = float(zone_entry_time)
                    all_puff_zones.append((trial_idx, zone_entry_time))
                except (ValueError, TypeError):
                    continue
    
    print(f"Found {len(all_puff_zones)} puff zone entries")
    
    # Collect all puff events
    puff_events = []
    for trial_idx in range(len(trial_log_df)):
        puff_event = trial_log_df.iloc[trial_idx]['puff_event']
        if pd.notna(puff_event) and puff_event > 0:
            puff_events.append((trial_idx, puff_event))
    
    print(f"Found {len(puff_events)} puff delivery events")
    
    # Debug: show first few puff events
    if len(puff_events) > 0:
        print("\nFirst 5 puff events (trial_idx, puff_event_time):")
        for i, (trial_idx, puff_time) in enumerate(puff_events[:5]):
            print(f"  {i}: Trial {trial_idx} | Puff: {puff_time:.3f}s")
    
    if len(all_puff_zones) > 0:
        print("\nFirst 5 puff zones (trial_idx, zone_entry_time):")
        for i, (trial_idx, zone_time) in enumerate(all_puff_zones[:5]):
            print(f"  {i}: Trial {trial_idx} | Zone: {zone_time:.3f}s")
    
    # Sort by timestamp
    all_puff_zones.sort(key=lambda x: x[1])
    puff_events.sort(key=lambda x: x[1])
    
    # Match each puff event to the most recent preceding zone entry
    puff_zone_trials = []
    matched_zones = set()
    
    for puff_trial_idx, puff_event_time in puff_events:
        best_match = None
        best_time_diff = float('inf')
        
        for i, (zone_trial_idx, zone_entry_time) in enumerate(all_puff_zones):
            if i in matched_zones:
                continue
            if zone_entry_time <= puff_event_time:
                time_diff = puff_event_time - zone_entry_time
                if time_diff < best_time_diff and time_diff < 15.0:  # Within 15 seconds
                    best_match = i
                    best_time_diff = time_diff
        
        if best_match is not None:
            zone_trial_idx, zone_entry_time = all_puff_zones[best_match]
            puff_zone_trials.append((zone_trial_idx, zone_entry_time, puff_event_time))
            matched_zones.add(best_match)
        else:
            # Debug: why no match?
            if len(all_puff_zones) > 0:
                # Find closest zone (even if outside criteria)
                closest_zone_idx = min(range(len(all_puff_zones)), 
                                      key=lambda i: abs(all_puff_zones[i][1] - puff_event_time))
                closest_zone_time = all_puff_zones[closest_zone_idx][1]
                time_diff = puff_event_time - closest_zone_time
                print(f"  No match for puff {puff_event_time:.3f}s (trial {puff_trial_idx})")
                print(f"    Closest zone: {closest_zone_time:.3f}s, diff: {time_diff:.3f}s")
    
    # Add unmatched zones
    for i, (zone_trial_idx, zone_entry_time) in enumerate(all_puff_zones):
        if i not in matched_zones:
            puff_zone_trials.append((zone_trial_idx, zone_entry_time, np.nan))
    
    puff_zone_trials.sort(key=lambda x: x[1])
    
    valid_deliveries = sum(1 for _, _, p in puff_zone_trials if pd.notna(p) and p > 0)
    print(f"Successfully matched {valid_deliveries}/{len(puff_zone_trials)} puff zones to deliveries")
    
    # Debug: show first few matches
    if len(puff_zone_trials) > 0:
        print("\nFirst 5 puff zone matches (trial_idx, zone_entry_time, puff_event_time):")
        for i, (trial_idx, zone_entry, puff_event) in enumerate(puff_zone_trials[:5]):
            puff_str = f"{puff_event:.3f}" if pd.notna(puff_event) and puff_event > 0 else "None"
            delay = puff_event - zone_entry if pd.notna(puff_event) and puff_event > 0 else None
            delay_str = f" (delay: {delay:.3f}s)" if delay is not None else ""
            print(f"  {i}: Trial {trial_idx} | Zone: {zone_entry:.3f}s | Puff: {puff_str}{delay_str}")
    
    print("=== END PUFF ZONE MATCHING ===\n")
    
    return puff_zone_trials


def separate_hits_and_misses(reward_zone_trials):
    """Separate reward zones into hits (rewarded) and misses (non-rewarded)
    
    Uses the exact same logic as daily_analysis.py: a zone is a "miss" if it does not
    have a corresponding reward delivery event.
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry_time, reward_event_time) tuples
        
    Returns:
        tuple: (hits_list, misses_list) where each is a list of (trial_idx, zone_entry_time) tuples
    """
    hits = []
    misses = []
    
    for trial_idx, zone_entry_time, reward_event_time in reward_zone_trials:
        # A "hit" is a zone with a valid reward delivery
        # A "miss" is a zone without a reward delivery (NaN or invalid)
        if pd.notna(reward_event_time) and reward_event_time > 0:
            hits.append((trial_idx, zone_entry_time))
        else:
            misses.append((trial_idx, zone_entry_time))
    
    print(f"\n=== REWARD ZONE CLASSIFICATION ===")
    print(f"Total reward zones: {len(reward_zone_trials)}")
    print(f"Hits (rewarded zones): {len(hits)} ({len(hits)/len(reward_zone_trials)*100:.1f}%)")
    print(f"Misses (non-rewarded zones): {len(misses)} ({len(misses)/len(reward_zone_trials)*100:.1f}%)")
    print("=== END CLASSIFICATION ===\n")
    
    return hits, misses


# ============================================================================
# WINDOW EXTRACTION FUNCTIONS
# ============================================================================

def create_aligned_windows(time_array, data_array, event_times, window_size=5):
    """Create time-aligned windows around event times
    
    Args:
        time_array: Time values
        data_array: Data values
        event_times: List/array of event times to align to
        window_size: Window size in seconds (before and after event)
        
    Returns:
        tuple: (aligned_windows array, aligned_time array)
    """
    if len(event_times) == 0:
        return None, None
    
    windows = []
    for event_time in event_times:
        mask = (time_array >= event_time - window_size) & (time_array <= event_time + window_size)
        segment = data_array[mask]
        windows.append(segment)
    
    if len(windows) == 0:
        return None, None
    
    # Pad to same length
    max_len = max(len(seg) for seg in windows)
    windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_len - len(seg)), constant_values=np.nan)
        for seg in windows
    ])
    
    # Create aligned time axis
    aligned_time = np.linspace(-window_size, window_size, max_len)
    
    return windows_padded, aligned_time


# ============================================================================
# PLOTTING FUNCTIONS: TIMELINE
# ============================================================================

def plot_main_timeline(cap_time, cap_val, speed_time, speed_val, distance_time, distance_val,
                       pupil_diameter_data, trial_log_df, texture_data, has_pupil_data, output_folder):
    """Create the main timeline plot with all data streams
    
    Args:
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        distance_time: Treadmill distance time array
        distance_val: Treadmill distance value array
        pupil_diameter_data: Tuple of (pupil_time, pupil_val) or (None, None)
        trial_log_df: Trial log DataFrame
        texture_data: Dictionary with processed texture data
        has_pupil_data: Whether pupil data is available
        output_folder: Directory to save figures
    """
    # Unpack pupil data
    pupil_time, pupil_val = pupil_diameter_data
    
    # Determine number of plots: treadmill speed, distance, capacitive, and optionally pupil
    num_plots = 4 if has_pupil_data and pupil_time is not None else 3
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 12 if num_plots == 4 else 10), sharex=True)
    
    # Ensure axs is always a list
    axs = list(axs) if num_plots > 1 else [axs]
    
    # Get event times
    reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna() if 'puff_event' in trial_log_df.columns else pd.Series([])
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna() if 'probe_time' in trial_log_df.columns else pd.Series([])
    
    # Plot treadmill speed (top subplot)
    plot_treadmill_timeline(axs[0], speed_time, speed_val, reward_times, 
                           puff_times, probe_times, texture_data, has_more_plots=True)
    
    # Plot treadmill distance (second subplot)
    plot_treadmill_distance_timeline(axs[1], distance_time, distance_val, reward_times,
                                     puff_times, probe_times, texture_data, has_more_plots=True)
    
    # Plot capacitive data (third subplot)
    plot_capacitive_timeline(axs[2], cap_time, cap_val, reward_times, puff_times, probe_times, 
                             texture_data, "Capacitive", show_xlabel=not has_pupil_data)
    
    # Plot pupil data if available (fourth subplot)
    if num_plots == 4 and pupil_time is not None:
        plot_pupil_timeline(axs[3], pupil_time, pupil_val, reward_times,
                          puff_times, probe_times, texture_data)
    
    # Set x-axis limits
    xmin = cap_time.min()
    xmax = cap_time.max()
    for ax in axs:
        ax.set_xlim([xmin, xmax])
    
    setup_plot_style(axs)
    plt.tight_layout()
    save_figure(fig, f"timeline_{'all_data' if num_plots == 4 else 'speed_distance_capacitive'}", 
                output_folder)
    plt.show()


def plot_capacitive_timeline(ax, cap_time, cap_val, reward_times, puff_times, probe_times, 
                             texture_data, label_prefix="", show_xlabel=True):
    """Plot capacitive sensor data with event markers"""
    ax.plot(cap_time, cap_val, 
            label='Capacitive Value (a.u.)', color='C0')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    if show_xlabel:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Capacitive Value (a.u.)')
    ax.set_title(f'{label_prefix} Capacitive Sensor Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def plot_treadmill_timeline(ax, speed_time, speed_val, reward_times, 
                           puff_times, probe_times, texture_data, has_more_plots=False):
    """Plot treadmill speed data with event markers"""
    ax.plot(speed_time, speed_val, 
            label='Treadmill Speed', color='purple')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    if not has_more_plots:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Treadmill Speed Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')


def plot_treadmill_distance_timeline(ax, distance_time, distance_val, reward_times, 
                                     puff_times, probe_times, texture_data, has_more_plots=False):
    """Plot treadmill distance data with event markers"""
    ax.plot(distance_time, distance_val, 
            label='Treadmill Distance', color='teal')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    if not has_more_plots:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Treadmill Distance Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def plot_pupil_timeline(ax, pupil_time, pupil_val, reward_times,
                       puff_times, probe_times, texture_data):
    """Plot pupil diameter data with event markers"""
    ax.plot(pupil_time, pupil_val,
            label='Pupil Diameter (z-scored)', color='orange')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Pupil Diameter (pixels)')
    ax.set_title('Interpolated Pupil Diameter Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def add_event_markers(ax, reward_times, puff_times, probe_times):
    """Add vertical lines for reward, puff, and probe events"""
    # Reward events
    for i, rt in enumerate(reward_times):
        ax.axvline(x=rt, color='green', linestyle='-', alpha=0.7, linewidth=2, 
                   label='Reward Event' if i == 0 else "")
    
    # Puff events
    for i, pt in enumerate(puff_times):
        ax.axvline(x=pt, color='red', linestyle='-', alpha=0.7, linewidth=2,
                   label='Puff Event' if i == 0 else "")
    
    # Probe events
    for i, pt in enumerate(probe_times):
        ax.axvline(x=pt, color='black', linestyle='-', alpha=0.7, linewidth=2,
                   label='Probe Event' if i == 0 else "")


def add_texture_intervals(ax, texture_data):
    """Add shaded regions for reward and punish zones"""
    reward_texture_change_time = texture_data['reward_texture_change_time']
    reward_revert_time = texture_data['reward_revert_time']
    punish_texture_change_time_first = texture_data['punish_texture_change_time_first']
    punish_revert_time_first = texture_data['punish_revert_time_first']
    
    # Highlight reward intervals
    if reward_texture_change_time.shape[0] > 0 and reward_texture_change_time.shape[1] > 0:
        for trial_idx in range(reward_texture_change_time.shape[0]):
            for zone_idx in range(reward_texture_change_time.shape[1]):
                start_time = reward_texture_change_time[trial_idx, zone_idx]
                end_time = reward_revert_time[trial_idx, zone_idx]
                if pd.notna(start_time) and pd.notna(end_time):
                    ax.axvspan(start_time, end_time, color='green', alpha=0.1)
    
    # Highlight punish intervals (first puff only)
    if punish_texture_change_time_first.shape[0] > 0:
        for trial_idx in range(punish_texture_change_time_first.shape[0]):
            start_time = punish_texture_change_time_first[trial_idx]
            end_time = punish_revert_time_first[trial_idx]
            if pd.notna(start_time) and pd.notna(end_time):
                ax.axvspan(start_time, end_time, color='red', alpha=0.1)


# ============================================================================
# PLOTTING FUNCTIONS: RASTER PLOTS
# ============================================================================

def plot_raster_heatmap(windows_padded, aligned_time, event_trials, title, 
                       ylabel, colormap, output_folder, filename, vmin=None, vmax=None,
                       center_time=0, event_label="Event", show_zone_entries=False, 
                       zone_entry_color='black', zone_entry_linewidth=3.0,
                       show_delivery_markers=False, center_line_color='black'):
    """Create a raster heatmap plot for aligned data
    
    Args:
        windows_padded: 2D array of aligned data windows
        aligned_time: Time axis for windows
        event_trials: List of event information tuples (trial_idx, zone_entry_time, event_time)
        title: Plot title
        ylabel: Y-axis label
        colormap: Matplotlib colormap name
        output_folder: Directory to save figure
        filename: Base filename for saving
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        center_time: Time value to mark as center (default 0)
        event_label: Label for the center time marker
        show_zone_entries: If True, draw vertical lines showing zone entry times for each trial
        zone_entry_color: Color for zone entry markers
        zone_entry_linewidth: Line width for zone entry markers
        show_delivery_markers: If True, draw green delivery markers on zone-entry plots
        center_line_color: Color for the center vertical line (default 'black')
    """
    if windows_padded is None or len(event_trials) == 0:
        print(f"Skipping {filename} - no data available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(windows_padded, aspect='auto', cmap=colormap, 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Set up time axis
    n_timepoints = windows_padded.shape[1]
    window_size = (aligned_time[-1] - aligned_time[0]) / 2
    time_labels = aligned_time
    tick_indices = np.linspace(0, n_timepoints-1, 11, dtype=int)
    tick_labels = [f'{time_labels[i]:.1f}' for i in tick_indices]
    
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(f'Time from {event_label} (s)')
    
    # Set y-axis to show actual trial numbers (1-based) with smart labeling
    actual_trial_indices = [trial_idx + 1 for trial_idx, _, _ in event_trials]
    n_trials = len(actual_trial_indices)
    
    if n_trials <= 20:
        # Show all labels for small number of trials
        ytick_positions = list(range(n_trials))
        ytick_labels = [str(trial_num) for trial_num in actual_trial_indices]
    else:
        # Show approximately 5-6 labels for larger datasets
        step = max(1, n_trials // 5)
        ytick_positions = list(range(0, n_trials, step))
        ytick_labels = [str(actual_trial_indices[pos]) for pos in ytick_positions]
    
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel('Trial Number')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(ylabel)
    
    # Add vertical line at center time (use specified color)
    center_position = int((center_time - aligned_time[0]) / (aligned_time[-1] - aligned_time[0]) * n_timepoints)
    ax.axvline(x=center_position, color=center_line_color, linestyle='--', alpha=0.8, linewidth=2)
    
    # Add delivery markers on zone entry plots (green lines showing when delivery happened)
    if show_delivery_markers:
        markers_drawn = 0
        for raster_row_idx, (trial_idx, zone_entry_time, event_time) in enumerate(event_trials):
            if pd.notna(event_time) and event_time > 0:
                # Calculate delay between zone entry and delivery
                delay = event_time - zone_entry_time
                if aligned_time[0] <= delay <= aligned_time[-1]:  # Only draw if within the time window
                    # Convert delay to pixel position
                    delivery_position = int((delay - aligned_time[0]) / (aligned_time[-1] - aligned_time[0]) * n_timepoints)
                    # Draw green line for this specific trial (row)
                    ax.plot([delivery_position, delivery_position], 
                           [raster_row_idx - 0.4, raster_row_idx + 0.4], 
                           color='green', linestyle='-', alpha=0.8, linewidth=3.0)
                    markers_drawn += 1
        print(f"  Drew {markers_drawn} delivery markers (out of {len(event_trials)} trials)")
    
    # Add individual zone entry lines if requested (for delivery-centered plots)
    if show_zone_entries:
        for raster_row_idx, (trial_idx, zone_entry_time, event_time) in enumerate(event_trials):
            # Calculate delay between event and zone entry (typically negative)
            delay = zone_entry_time - event_time
            if aligned_time[0] <= delay <= aligned_time[-1]:  # Only draw if within the time window
                # Convert delay to pixel position
                zone_entry_position = int((delay - aligned_time[0]) / (aligned_time[-1] - aligned_time[0]) * n_timepoints)
                # Draw line only for this specific trial (row)
                ax.plot([zone_entry_position, zone_entry_position], 
                       [raster_row_idx - 0.4, raster_row_idx + 0.4], 
                       color=zone_entry_color, linestyle='-', alpha=0.8, linewidth=zone_entry_linewidth)
    
    setup_plot_style(ax)
    plt.tight_layout()
    save_figure(fig, filename, output_folder)
    plt.show()


# ============================================================================
# PLOTTING FUNCTIONS: AVERAGE TRACES
# ============================================================================

def plot_average_traces_reward(reward_zone_trials, trial_log_df, cap_time, cap_val,
                               speed_time, speed_val, pupil_diameter_data, output_folder, window=5, cap_vmax=None):
    """Plot average traces (mean ± SEM) for reward zone and delivery events
    
    Args:
        reward_zone_trials: List of reward zone trial tuples
        trial_log_df: Trial log DataFrame
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        pupil_diameter_data: Tuple of (pupil_time, pupil_val) or (None, None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones for average trace analysis")
        return
    
    # Unpack pupil data
    pupil_time, pupil_val_data = pupil_diameter_data
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in reward_zone_trials]
    
    # Create speed windows aligned to zone entry
    speed_windows = []
    for rt in zone_entry_times:
        mask = (speed_time >= rt - window) & (speed_time <= rt + window)
        speed_segment = speed_val[mask]
        speed_windows.append(speed_segment)
    
    max_speed_len = max(len(seg) for seg in speed_windows)
    speed_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_speed_len - len(seg)), constant_values=np.nan)
        for seg in speed_windows
    ])
    
    aligned_time_speed = np.linspace(-window, window, max_speed_len)
    mean_speed = np.nanmean(speed_windows_padded, axis=0)
    sem_speed = np.nanstd(speed_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_windows_padded), axis=0))
    
    # Get reward event times
    reward_event_times_flat = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
    reward_event_times_flat = reward_event_times_flat[~np.isnan(reward_event_times_flat)].values
    
    if len(reward_event_times_flat) == 0:
        print("No reward events found for average trace analysis")
        return
    
    # Create capacitive windows aligned to reward events
    cap_event_windows = []
    for rt in reward_event_times_flat:
        mask = (cap_time >= rt - window) & (cap_time <= rt + window)
        cap_segment = cap_val[mask]
        cap_event_windows.append(cap_segment)
    
    max_event_len = max(len(seg) for seg in cap_event_windows)
    cap_event_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_event_len - len(seg)), constant_values=np.nan)
        for seg in cap_event_windows
    ])
    
    aligned_time_event = np.linspace(-window, window, max_event_len)
    mean_event_vals = np.nanmean(cap_event_windows_padded, axis=0)
    sem_event_vals = np.nanstd(cap_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(cap_event_windows_padded), axis=0))
    
    # Create combined subplot figure
    num_plots = 3 if pupil_time is not None else 2
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 10 if num_plots == 2 else 14), sharex=True)
    
    if num_plots == 2:
        axs = [axs[0], axs[1]]
    
    # Plot 1: Treadmill Speed aligned to reward zone entry
    n_rewards_speed = speed_windows_padded.shape[0]
    axs[0].plot(aligned_time_speed, mean_speed, color='purple', label=f'Mean Speed (n={n_rewards_speed})')
    axs[0].fill_between(aligned_time_speed, mean_speed - sem_speed, mean_speed + sem_speed, 
                        color='purple', alpha=0.2, label='SEM')
    axs[0].axvline(0, color='red', linestyle='--', label='Reward Zone Onset (t=0)')
    axs[0].set_ylabel('Treadmill Speed (cm/s)')
    axs[0].set_title('Treadmill Speed Aligned to Reward Zone Onset')
    axs[0].legend()
    axs[0].set_xlim(-5, 5)
    axs[0].set_ylim(bottom=0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Capacitive Value aligned to reward events
    n_rewards_event = cap_event_windows_padded.shape[0]
    axs[1].plot(aligned_time_event, mean_event_vals, color='C0', label=f'Mean (n={n_rewards_event})')
    axs[1].fill_between(aligned_time_event, mean_event_vals - sem_event_vals, 
                        mean_event_vals + sem_event_vals, color='C0', alpha=0.2, label='SEM')
    axs[1].axvline(0, color='red', linestyle='--', label='Reward Event (t=0)')
    axs[1].set_ylabel('Capacitive Value (a.u.)')
    axs[1].set_title('Capacitive Value Aligned to Reward Event')
    axs[1].legend()
    axs[1].set_xlim(-5, 5)
    if cap_vmax is not None:
        axs[1].set_ylim(0, cap_vmax)
    else:
        axs[1].set_ylim(bottom=0)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    # Plot 3: Pupil diameter (if available)
    if pupil_time is not None:
        # Pupil aligned to zone entry
        pupil_zone_windows = []
        for rt in zone_entry_times:
            mask = (pupil_time >= rt - window) & (pupil_time <= rt + window)
            pupil_segment = pupil_val_data[mask]
            pupil_zone_windows.append(pupil_segment)
        
        max_pupil_len = max(len(seg) for seg in pupil_zone_windows)
        pupil_zone_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_len - len(seg)), constant_values=np.nan)
            for seg in pupil_zone_windows
        ])
        
        aligned_time_pupil = np.linspace(-window, window, max_pupil_len)
        mean_pupil = np.nanmean(pupil_zone_windows_padded, axis=0)
        sem_pupil = np.nanstd(pupil_zone_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(pupil_zone_windows_padded), axis=0))
        
        # Pupil aligned to reward events
        pupil_event_windows = []
        for rt in reward_event_times_flat:
            mask = (cap_time >= rt - window) & (cap_time <= rt + window)
            pupil_segment = pupil_val_data[mask]
            pupil_event_windows.append(pupil_segment)
        
        max_pupil_event_len = max(len(seg) for seg in pupil_event_windows)
        pupil_event_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_event_len - len(seg)), constant_values=np.nan)
            for seg in pupil_event_windows
        ])
        
        aligned_time_pupil_event = np.linspace(-window, window, max_pupil_event_len)
        mean_pupil_event = np.nanmean(pupil_event_windows_padded, axis=0)
        sem_pupil_event = np.nanstd(pupil_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(pupil_event_windows_padded), axis=0))
        
        # Create separate pupil figure with 2 subplots
        fig_pupil, axs_pupil = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Pupil aligned to zone entry
        n_pupil_zone = pupil_zone_windows_padded.shape[0]
        axs_pupil[0].plot(aligned_time_pupil, mean_pupil, color='orange', label=f'Mean Pupil Diameter (n={n_pupil_zone})')
        axs_pupil[0].fill_between(aligned_time_pupil, mean_pupil - sem_pupil, mean_pupil + sem_pupil,
                                  color='orange', alpha=0.2, label='SEM')
        axs_pupil[0].axvline(0, color='red', linestyle='--', label='Reward Zone Onset (t=0)')
        axs_pupil[0].set_ylabel('Pupil Diameter (pixels)')
        axs_pupil[0].set_title('Pupil Diameter Aligned to Reward Zone Onset')
        axs_pupil[0].legend()
        axs_pupil[0].set_xlim(-5, 5)
        axs_pupil[0].set_ylim(bottom=0)
        axs_pupil[0].spines['top'].set_visible(False)
        axs_pupil[0].spines['right'].set_visible(False)
        
        # Plot 2: Pupil aligned to reward events
        n_pupil_event = pupil_event_windows_padded.shape[0]
        axs_pupil[1].plot(aligned_time_pupil_event, mean_pupil_event, color='orange', 
                         label=f'Mean Pupil Diameter (n={n_pupil_event})')
        axs_pupil[1].fill_between(aligned_time_pupil_event, mean_pupil_event - sem_pupil_event,
                                   mean_pupil_event + sem_pupil_event, color='orange', alpha=0.2, label='SEM')
        axs_pupil[1].axvline(0, color='red', linestyle='--', label='Reward Event (t=0)')
        axs_pupil[1].set_xlabel('Time (s)')
        axs_pupil[1].set_ylabel('Pupil Diameter (pixels)')
        axs_pupil[1].set_title('Pupil Diameter Aligned to Reward Events')
        axs_pupil[1].legend()
        axs_pupil[1].set_xlim(-5, 5)
        axs_pupil[1].set_ylim(bottom=0)
        axs_pupil[1].spines['top'].set_visible(False)
        axs_pupil[1].spines['right'].set_visible(False)
        
        for ax in axs_pupil:
            ax.set_xticks(np.arange(-5, 6, 1))
        
        plt.tight_layout()
        save_figure(fig_pupil, "pupil_diameter_reward_combined", output_folder)
        plt.show()
    else:
        axs[1].set_xlabel('Time from Reward Event (s)')
    
    # Set x-axis formatting for main figure
    for ax in axs:
        ax.set_xticks(np.arange(-5, 6, 1))
    
    if num_plots == 2:
        axs[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    save_figure(fig, "reward_zone_analysis_capacitive_treadmill", output_folder)
    plt.show()
    
    print(f"Average trace plots created: {n_rewards_speed} zone entries, {n_rewards_event} reward events")


def plot_average_traces_puff(puff_zone_trials, trial_log_df, cap_time, cap_val,
                            speed_time, speed_val, pupil_diameter_data, output_folder, window=5, cap_vmax=None):
    """Plot average traces (mean ± SEM) for puff zone and delivery events
    
    Args:
        puff_zone_trials: List of puff zone trial tuples
        trial_log_df: Trial log DataFrame
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        pupil_diameter_data: Tuple of (pupil_time, pupil_val) or (None, None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    if len(puff_zone_trials) == 0:
        print("No puff zones for average trace analysis")
        return
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in puff_zone_trials]
    
    # Speed aligned to puff zone entry
    speed_puff_windows = []
    for puff_time in zone_entry_times:
        mask = (speed_time >= puff_time - window) & (speed_time <= puff_time + window)
        speed_segment = speed_val[mask]
        speed_puff_windows.append(speed_segment)
    
    if not speed_puff_windows or max(len(seg) for seg in speed_puff_windows) == 0:
        print("No valid speed data for puff zone analysis")
        return
    
    max_puff_len = max(len(seg) for seg in speed_puff_windows)
    speed_puff_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_puff_len - len(seg)), constant_values=np.nan)
        for seg in speed_puff_windows
    ])
    
    aligned_time_puff = np.linspace(-window, window, max_puff_len)
    n_puff_events = speed_puff_windows_padded.shape[0]
    mean_speed_puff = np.nanmean(speed_puff_windows_padded, axis=0)
    sem_speed_puff = np.nanstd(speed_puff_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_puff_windows_padded), axis=0))
    
    # Get puff event times
    puff_event_capacitive_data = None
    puff_event_speed_data = None
    
    if 'puff_event' in trial_log_df.columns:
        puff_event_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
        puff_event_times = puff_event_times[~np.isnan(puff_event_times)].values
        
        if len(puff_event_times) > 0:
            # Capacitive aligned to puff events
            cap_puff_event_windows = []
            for puff_event_time in puff_event_times:
                mask = (cap_time >= puff_event_time - window) & (cap_time <= puff_event_time + window)
                cap_segment = cap_val[mask]
                cap_puff_event_windows.append(cap_segment)
            
            if cap_puff_event_windows and max(len(seg) for seg in cap_puff_event_windows) > 0:
                max_puff_cap_len = max(len(seg) for seg in cap_puff_event_windows)
                cap_puff_event_windows_padded = np.array([
                    np.pad(seg.astype(float), (0, max_puff_cap_len - len(seg)), constant_values=np.nan)
                    for seg in cap_puff_event_windows
                ])
                
                aligned_time_puff_cap = np.linspace(-window, window, max_puff_cap_len)
                n_puff_event_cap = cap_puff_event_windows_padded.shape[0]
                mean_cap_puff_event = np.nanmean(cap_puff_event_windows_padded, axis=0)
                sem_cap_puff_event = np.nanstd(cap_puff_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(cap_puff_event_windows_padded), axis=0))
                
                puff_event_capacitive_data = {
                    'aligned_time': aligned_time_puff_cap,
                    'mean_values': mean_cap_puff_event,
                    'sem_values': sem_cap_puff_event,
                    'n_events': n_puff_event_cap
                }
            
            # Speed aligned to puff events
            speed_puff_event_windows = []
            for puff_event_time in puff_event_times:
                mask = (speed_time >= puff_event_time - window) & (speed_time <= puff_event_time + window)
                speed_segment = speed_val[mask]
                speed_puff_event_windows.append(speed_segment)
            
            if speed_puff_event_windows and max(len(seg) for seg in speed_puff_event_windows) > 0:
                max_puff_speed_len = max(len(seg) for seg in speed_puff_event_windows)
                speed_puff_event_windows_padded = np.array([
                    np.pad(seg.astype(float), (0, max_puff_speed_len - len(seg)), constant_values=np.nan)
                    for seg in speed_puff_event_windows
                ])
                
                aligned_time_puff_speed = np.linspace(-window, window, max_puff_speed_len)
                n_puff_event_speed = speed_puff_event_windows_padded.shape[0]
                mean_speed_puff_event = np.nanmean(speed_puff_event_windows_padded, axis=0)
                sem_speed_puff_event = np.nanstd(speed_puff_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_puff_event_windows_padded), axis=0))
                
                puff_event_speed_data = {
                    'aligned_time': aligned_time_puff_speed,
                    'mean_values': mean_speed_puff_event,
                    'sem_values': sem_speed_puff_event,
                    'n_events': n_puff_event_speed
                }
    
    # Create 3-panel figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Plot 1: Treadmill Speed aligned to puff zone entry
    axs[0].plot(aligned_time_puff, mean_speed_puff, color='purple', linewidth=2, 
               label=f'Mean Speed (n={n_puff_events})')
    axs[0].fill_between(aligned_time_puff, mean_speed_puff - sem_speed_puff, 
                        mean_speed_puff + sem_speed_puff, color='purple', alpha=0.2, label='SEM')
    axs[0].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Zone Entry (t=0)')
    axs[0].set_ylabel('Treadmill Speed (cm/s)')
    axs[0].set_title(f'Average Treadmill Speed Aligned to Puff Zone Entry Times (n={n_puff_events})')
    axs[0].legend()
    axs[0].set_xlim(-5, 5)
    axs[0].set_ylim(bottom=0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Capacitive Value aligned to puff events
    if puff_event_capacitive_data is not None:
        axs[1].plot(puff_event_capacitive_data['aligned_time'], puff_event_capacitive_data['mean_values'],
                   color='C0', linewidth=2, label=f'Mean Capacitive (n={puff_event_capacitive_data["n_events"]})')
        axs[1].fill_between(puff_event_capacitive_data['aligned_time'],
                           puff_event_capacitive_data['mean_values'] - puff_event_capacitive_data['sem_values'],
                           puff_event_capacitive_data['mean_values'] + puff_event_capacitive_data['sem_values'],
                           color='C0', alpha=0.2, label='SEM')
        axs[1].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Event (t=0)')
        axs[1].set_ylabel('Capacitive Value (a.u.)')
        axs[1].set_title(f'Average Capacitive Value Aligned to Puff Events (n={puff_event_capacitive_data["n_events"]})')
        axs[1].legend()
        if cap_vmax is not None:
            axs[1].set_ylim(0, cap_vmax)
        else:
            axs[1].set_ylim(bottom=0)
    else:
        axs[1].text(0.5, 0.5, 'No puff event data available\nfor capacitive analysis',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=12)
        axs[1].set_ylabel('Capacitive Value (a.u.)')
        axs[1].set_title('Capacitive Value Aligned to Puff Events (No Data)')
    
    axs[1].set_xlim(-5, 5)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    # Plot 3: Treadmill Speed aligned to puff events
    if puff_event_speed_data is not None:
        axs[2].plot(puff_event_speed_data['aligned_time'], puff_event_speed_data['mean_values'],
                   color='purple', linewidth=2, label=f'Mean Speed (n={puff_event_speed_data["n_events"]})')
        axs[2].fill_between(puff_event_speed_data['aligned_time'],
                           puff_event_speed_data['mean_values'] - puff_event_speed_data['sem_values'],
                           puff_event_speed_data['mean_values'] + puff_event_speed_data['sem_values'],
                           color='purple', alpha=0.2, label='SEM')
        axs[2].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Event (t=0)')
        axs[2].set_ylabel('Treadmill Speed (cm/s)')
        axs[2].set_title(f'Average Treadmill Speed Aligned to Puff Events (n={puff_event_speed_data["n_events"]})')
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, 'No puff event data available\nfor treadmill speed analysis',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[2].transAxes, fontsize=12)
        axs[2].set_ylabel('Treadmill Speed (cm/s)')
        axs[2].set_title('Treadmill Speed Aligned to Puff Events (No Data)')
    
    axs[2].set_xlabel('Time from Puff Event (s)')
    axs[2].set_xlim(-5, 5)
    axs[2].set_ylim(bottom=0)
    axs[2].set_xticks(np.arange(-5, 6, 1))
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, "puff_events_analysis", output_folder)
    plt.show()
    
    # Unpack pupil data
    pupil_time, pupil_val_data = pupil_diameter_data
    
    # Create pupil plots if available
    if pupil_time is not None:
        window_pupil = 10
        
        # Pupil aligned to puff zone entry
        pupil_puff_windows = []
        for puff_time in zone_entry_times:
            mask = (pupil_time >= puff_time - window_pupil) & (pupil_time <= puff_time + window_pupil)
            pupil_segment = pupil_val_data[mask]
            pupil_puff_windows.append(pupil_segment)
        
        max_pupil_puff_len = max(len(seg) for seg in pupil_puff_windows)
        pupil_puff_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_puff_len - len(seg)), constant_values=np.nan)
            for seg in pupil_puff_windows
        ])
        
        aligned_time_pupil_puff = np.linspace(-window_pupil, window_pupil, max_pupil_puff_len)
        mean_pupil_puff = np.nanmean(pupil_puff_windows_padded, axis=0)
        sem_pupil_puff = np.nanstd(pupil_puff_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(pupil_puff_windows_padded), axis=0))
        n_puffs_pupil = pupil_puff_windows_padded.shape[0]
        
        # Pupil aligned to puff events (if available)
        pupil_puff_event_data = None
        if 'puff_event' in trial_log_df.columns:
            puff_event_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna().values
            
            if len(puff_event_times) > 0:
                pupil_puff_event_windows = []
                for puff_time in puff_event_times:
                    mask = (cap_time >= puff_time - window_pupil) & (cap_time <= puff_time + window_pupil)
                    pupil_segment = pupil_val_data[mask]
                    pupil_puff_event_windows.append(pupil_segment)
                
                if pupil_puff_event_windows and max(len(seg) for seg in pupil_puff_event_windows) > 0:
                    max_pupil_puff_event_len = max(len(seg) for seg in pupil_puff_event_windows)
                    pupil_puff_event_windows_padded = np.array([
                        np.pad(seg.astype(float), (0, max_pupil_puff_event_len - len(seg)), constant_values=np.nan)
                        for seg in pupil_puff_event_windows
                    ])
                    
                    aligned_time_pupil_puff_event = np.linspace(-window_pupil, window_pupil, max_pupil_puff_event_len)
                    n_puffs_pupil_event = pupil_puff_event_windows_padded.shape[0]
                    mean_pupil_puff_event = np.nanmean(pupil_puff_event_windows_padded, axis=0)
                    sem_pupil_puff_event = np.nanstd(pupil_puff_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(pupil_puff_event_windows_padded), axis=0))
                    
                    pupil_puff_event_data = {
                        'time': aligned_time_pupil_puff_event,
                        'mean': mean_pupil_puff_event,
                        'sem': sem_pupil_puff_event,
                        'n': n_puffs_pupil_event
                    }
        
        # Create pupil subplot figure
        num_puff_plots = 2 if pupil_puff_event_data is not None else 1
        fig_pupil, axs_pupil = plt.subplots(num_puff_plots, 1, figsize=(12, 10 if num_puff_plots == 2 else 6), sharex=True)
        
        if num_puff_plots == 1:
            axs_pupil = [axs_pupil]
        
        # Plot 1: Pupil aligned to puff zone entry
        axs_pupil[0].plot(aligned_time_pupil_puff, mean_pupil_puff, color='red', 
                         label=f'Mean Pupil Diameter (n={n_puffs_pupil})')
        axs_pupil[0].fill_between(aligned_time_pupil_puff, mean_pupil_puff - sem_pupil_puff,
                                   mean_pupil_puff + sem_pupil_puff, color='red', alpha=0.2, label='SEM')
        axs_pupil[0].axvline(0, color='red', linestyle='--', label='Puff Zone Entry (t=0)')
        axs_pupil[0].set_ylabel('Pupil Diameter (pixels)')
        axs_pupil[0].set_title('Pupil Diameter Aligned to Puff Zone Entry')
        axs_pupil[0].legend()
        axs_pupil[0].set_xlim(-10, 10)
        axs_pupil[0].set_ylim(bottom=0)
        axs_pupil[0].spines['top'].set_visible(False)
        axs_pupil[0].spines['right'].set_visible(False)
        
        # Plot 2: Pupil aligned to puff events (if available)
        if pupil_puff_event_data is not None:
            axs_pupil[1].plot(pupil_puff_event_data['time'], pupil_puff_event_data['mean'], 
                             color='red', label=f'Mean Pupil Diameter (n={pupil_puff_event_data["n"]})')
            axs_pupil[1].fill_between(pupil_puff_event_data['time'],
                                       pupil_puff_event_data['mean'] - pupil_puff_event_data['sem'],
                                       pupil_puff_event_data['mean'] + pupil_puff_event_data['sem'],
                                       color='red', alpha=0.2, label='SEM')
            axs_pupil[1].axvline(0, color='red', linestyle='--', label='Puff Event (t=0)')
            axs_pupil[1].set_xlabel('Time (s)')
            axs_pupil[1].set_ylabel('Pupil Diameter (pixels)')
            axs_pupil[1].set_title('Pupil Diameter Aligned to Puff Events')
            axs_pupil[1].legend()
            axs_pupil[1].set_xlim(-10, 10)
            axs_pupil[1].set_ylim(bottom=0)
            axs_pupil[1].spines['top'].set_visible(False)
            axs_pupil[1].spines['right'].set_visible(False)
        else:
            axs_pupil[0].set_xlabel('Time (s)')
        
        for ax in axs_pupil:
            ax.set_xticks(np.arange(-10, 11, 2))
        
        plt.tight_layout()
        save_figure(fig_pupil, f"pupil_diameter_puff_combined_{'with_events' if pupil_puff_event_data is not None else 'zone_only'}", output_folder)
        plt.show()
    
    print(f"Puff average trace plots created: {n_puff_events} zone entries")


# ============================================================================
# ANALYSIS FUNCTIONS: REWARD ZONES
# ============================================================================

def analyze_reward_zones(reward_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                         pupil_diameter_data, output_folder, window=5):
    """Analyze data aligned to reward zone entries and deliveries
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry, reward_event) tuples
        trial_log_df: Trial log DataFrame (needed to get ALL reward events)
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        pupil_diameter_data: Tuple of (pupil_time, pupil_val) or (None, None)
        output_folder: Directory to save figures
        window: Window size in seconds
        
    Returns:
        tuple: (cap_vmin, cap_vmax) - Scale for capacitive plots
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones found. Skipping reward zone analysis.")
        return (0, 5000)  # Default scale
    
    print(f"\n=== ANALYZING {len(reward_zone_trials)} REWARD ZONES ===")
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in reward_zone_trials]
    
    # Create aligned windows for zone entries
    speed_windows, aligned_time_speed = create_aligned_windows(
        speed_time, speed_val, zone_entry_times, window
    )
    
    cap_windows, aligned_time_cap = create_aligned_windows(
        cap_time, cap_val, zone_entry_times, window
    )
    
    # Calculate capacitive scale from reward EVENT average trace (mean + SEM)
    # Get all reward event times from trial log to match what's shown in average trace plots
    reward_event_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
    reward_event_times = reward_event_times[~np.isnan(reward_event_times)].values
    
    if len(reward_event_times) > 0:
        # Create capacitive windows aligned to reward events
        cap_event_windows = []
        for rt in reward_event_times:
            mask = (cap_time >= rt - window) & (cap_time <= rt + window)
            cap_segment = cap_val[mask]
            cap_event_windows.append(cap_segment)
        
        if cap_event_windows and max(len(seg) for seg in cap_event_windows) > 0:
            max_event_len = max(len(seg) for seg in cap_event_windows)
            cap_event_windows_padded = np.array([
                np.pad(seg.astype(float), (0, max_event_len - len(seg)), constant_values=np.nan)
                for seg in cap_event_windows
            ])
            
            # Calculate mean and SEM (matching what's plotted in average traces)
            mean_event_vals = np.nanmean(cap_event_windows_padded, axis=0)
            sem_event_vals = np.nanstd(cap_event_windows_padded, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(cap_event_windows_padded), axis=0))
            
            # Use the maximum of (mean + SEM) as the upper limit
            cap_vmin = 0
            cap_vmax = np.nanmax(mean_event_vals + sem_event_vals)
            
            # Ensure cap_vmax is valid
            if np.isnan(cap_vmax) or cap_vmax <= 0:
                cap_vmax = 5000
            print(f"Capacitive scale calculated from reward event average trace: vmin={cap_vmin:.2f}, vmax={cap_vmax:.2f}")
        else:
            cap_vmin, cap_vmax = 0, 5000
            print("Using default capacitive scale: vmin=0, vmax=5000")
    else:
        cap_vmin, cap_vmax = 0, 5000
        print("No reward events found. Using default capacitive scale: vmin=0, vmax=5000")
    
    # Plot raster for treadmill speed at zone entry
    if speed_windows is not None:
        plot_raster_heatmap(
            speed_windows, aligned_time_speed, reward_zone_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Reward Zone Entry (n={len(reward_zone_trials)} trials)',
            'Treadmill Speed (cm/s)', 'coolwarm', output_folder, 
            'treadmill_speed_raster_reward_zones',
            vmin=-30,
            vmax=30,
            center_time=0, event_label="Reward Zone Entry",
            show_delivery_markers=True, center_line_color='black'
        )
    
    # Plot raster for capacitive at zone entry
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, reward_zone_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Reward Zone Entry (n={len(reward_zone_trials)} trials)',
            'Capacitive Value (a.u.)', 'binary', output_folder,
            'capacitive_raster_reward_zones',
            vmin=cap_vmin,
            vmax=cap_vmax,
            center_time=0, event_label="Reward Zone Entry",
            show_delivery_markers=True, center_line_color='blue'
        )
    
    # Analyze reward deliveries - pass trial_log_df to get ALL reward events
    print(f"Analyzing reward deliveries...")
    pupil_diameter_interp = pupil_diameter_data[1] if pupil_diameter_data is not None else None
    analyze_reward_deliveries(reward_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                             pupil_diameter_interp, output_folder, window, cap_vmin, cap_vmax)
    
    return (cap_vmin, cap_vmax)


def analyze_reward_deliveries(reward_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                              pupil_diameter_interp, output_folder, window=5, cap_vmin=0, cap_vmax=5000):
    """Analyze data aligned to reward delivery times
    
    Args:
        reward_zone_trials: List of reward zone trials (trial_idx, zone_entry, reward_event)
                           Used to map reward events to their zone entries when available
        trial_log_df: Trial log DataFrame (to get ALL reward events)
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_val: Treadmill speed array
        pupil_diameter_interp: Interpolated pupil diameter
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmin: Minimum value for capacitive colormap scale
        cap_vmax: Maximum value for capacitive colormap scale
    """
    # Get ALL reward events from trial log (not just matched ones)
    all_reward_events = []
    for trial_idx in range(len(trial_log_df)):
        reward_event = trial_log_df.iloc[trial_idx]['reward_event']
        if pd.notna(reward_event) and reward_event > 0:
            all_reward_events.append((trial_idx, reward_event))
    
    if len(all_reward_events) == 0:
        print("No reward delivery events found in trial log")
        return
    
    print(f"Found {len(all_reward_events)} total reward delivery events in trial log")
    
    # Create a mapping from reward event times to zone entry times (for matched events)
    reward_to_zone_map = {}
    for trial_idx, zone_entry_time, reward_event_time in reward_zone_trials:
        if pd.notna(reward_event_time) and reward_event_time > 0:
            reward_to_zone_map[reward_event_time] = zone_entry_time
    
    # Build complete list for plotting: (trial_idx, zone_entry_time_or_nan, reward_event_time)
    all_reward_delivery_trials = []
    for trial_idx, reward_event_time in all_reward_events:
        zone_entry_time = reward_to_zone_map.get(reward_event_time, np.nan)
        all_reward_delivery_trials.append((trial_idx, zone_entry_time, reward_event_time))
    
    # Extract reward delivery times (centered at t=0)
    reward_times = [r for _, _, r in all_reward_delivery_trials]
    
    # Create aligned windows
    speed_windows, aligned_time_speed = create_aligned_windows(
        speed_time, speed_val, reward_times, window
    )
    
    cap_windows, aligned_time_cap = create_aligned_windows(
        cap_time, cap_val, reward_times, window
    )
    
    # Plot rasters centered at reward delivery
    if speed_windows is not None:
        plot_raster_heatmap(
            speed_windows, aligned_time_speed, all_reward_delivery_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Reward Delivery (n={len(all_reward_delivery_trials)} trials)',
            'Treadmill Speed (cm/s)', 'coolwarm', output_folder,
            'treadmill_speed_raster_reward_delivery_centered',
            vmin=-30,
            vmax=30,
            center_time=0, event_label="Reward Delivery",
            show_zone_entries=True, zone_entry_color='black', center_line_color='green'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, all_reward_delivery_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Reward Delivery (n={len(all_reward_delivery_trials)} trials)',
            'Capacitive Value (a.u.)', 'binary', output_folder,
            'capacitive_raster_reward_delivery_centered',
            vmin=cap_vmin,
            vmax=cap_vmax,
            center_time=0, event_label="Reward Delivery",
            show_zone_entries=True, zone_entry_color='blue', center_line_color='green'
        )


def analyze_reward_hits_vs_misses(reward_zone_trials, cap_time, cap_val, speed_time, speed_val,
                                   output_folder, window=5):
    """Analyze speed data separately for rewarded vs non-rewarded zones
    
    Uses the same logic as daily_analysis.py to identify misses (no-reward zones).
    Creates average trace alignment plots (mean ± SEM) for hits and misses.
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry_time, reward_event_time) tuples
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        output_folder: Directory to save figures
        window: Window size in seconds before/after zone entry (default 5)
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones to analyze")
        return
    
    print(f"\n{'='*70}")
    print("ANALYZING REWARD HITS VS MISSES (SPEED)")
    print(f"{'='*70}")
    
    # Separate hits and misses
    hits, misses = separate_hits_and_misses(reward_zone_trials)
    
    # Initialize variables for comparison plot
    speed_windows_hits = None
    speed_windows_misses = None
    aligned_time_hits = None
    aligned_time_misses = None
    
    # ========================================================================
    # HITS (Rewarded Zones) Analysis
    # ========================================================================
    if len(hits) > 0:
        print(f"\nAnalyzing HITS (rewarded zones): n={len(hits)}")
        
        # Extract zone entry times for hits
        hit_zone_times = [zone_time for _, zone_time in hits]
        
        # Create aligned windows for speed
        speed_windows_hits, aligned_time_hits = create_aligned_windows(
            speed_time, speed_val, hit_zone_times, window
        )
        
        # Plot average trace for hits
        if speed_windows_hits is not None:
            avg_speed_hits = np.nanmean(speed_windows_hits, axis=0)
            sem_speed_hits = np.nanstd(speed_windows_hits, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_windows_hits), axis=0))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(aligned_time_hits, avg_speed_hits, color='darkblue', linewidth=2, label=f'Mean Speed (n={len(hits)})')
            ax.fill_between(aligned_time_hits, 
                           avg_speed_hits - sem_speed_hits,
                           avg_speed_hits + sem_speed_hits,
                           alpha=0.2, color='darkblue', label='SEM')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Reward Zone Onset (t=0)')
            ax.set_xlabel('Time from Reward Zone Onset (s)')
            ax.set_ylabel('Treadmill Speed (cm/s)')
            ax.set_title(f'Treadmill Speed: REWARDED Zones (Hits, n={len(hits)})')
            ax.legend()
            ax.set_xlim(-5, 5)
            ax.set_ylim(bottom=0)
            ax.set_xticks(np.arange(-5, 6, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            save_figure(fig, 'speed_average_reward_hits', output_folder)
            plt.show()
    else:
        print("No rewarded zones (hits) found")
    
    # ========================================================================
    # MISSES (Non-Rewarded Zones) Analysis
    # ========================================================================
    if len(misses) > 0:
        print(f"\nAnalyzing MISSES (non-rewarded zones): n={len(misses)}")
        
        # Extract zone entry times for misses
        miss_zone_times = [zone_time for _, zone_time in misses]
        
        # Create aligned windows for speed
        speed_windows_misses, aligned_time_misses = create_aligned_windows(
            speed_time, speed_val, miss_zone_times, window
        )
        
        # Plot average trace for misses
        if speed_windows_misses is not None:
            avg_speed_misses = np.nanmean(speed_windows_misses, axis=0)
            sem_speed_misses = np.nanstd(speed_windows_misses, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_windows_misses), axis=0))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(aligned_time_misses, avg_speed_misses, color='deepskyblue', linewidth=2, label=f'Mean Speed (n={len(misses)})')
            ax.fill_between(aligned_time_misses, 
                           avg_speed_misses - sem_speed_misses,
                           avg_speed_misses + sem_speed_misses,
                           alpha=0.2, color='deepskyblue', label='SEM')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Reward Zone Onset (t=0)')
            ax.set_xlabel('Time from Reward Zone Onset (s)')
            ax.set_ylabel('Treadmill Speed (cm/s)')
            ax.set_title(f'Treadmill Speed: NON-REWARDED Zones (Misses, n={len(misses)})')
            ax.legend()
            ax.set_xlim(-5, 5)
            ax.set_ylim(bottom=0)
            ax.set_xticks(np.arange(-5, 6, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            save_figure(fig, 'speed_average_reward_misses', output_folder)
            plt.show()
    else:
        print("No non-rewarded zones (misses) found")
    
    # ========================================================================
    # COMPARISON: Hits vs Misses
    # ========================================================================
    if len(hits) > 0 and len(misses) > 0:
        print(f"\nCreating comparison plot: Hits vs Misses")
        
        # Need to ensure same time axis for comparison
        if speed_windows_hits is not None and speed_windows_misses is not None:
            avg_speed_hits = np.nanmean(speed_windows_hits, axis=0)
            sem_speed_hits = np.nanstd(speed_windows_hits, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_windows_hits), axis=0))
            
            avg_speed_misses = np.nanmean(speed_windows_misses, axis=0)
            sem_speed_misses = np.nanstd(speed_windows_misses, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(speed_windows_misses), axis=0))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot hits
            ax.plot(aligned_time_hits, avg_speed_hits, color='darkblue', linewidth=2, label=f'Hits (n={len(hits)})')
            ax.fill_between(aligned_time_hits, 
                           avg_speed_hits - sem_speed_hits,
                           avg_speed_hits + sem_speed_hits,
                           alpha=0.2, color='darkblue')
            
            # Plot misses
            ax.plot(aligned_time_misses, avg_speed_misses, color='deepskyblue', linewidth=2, label=f'Misses (n={len(misses)})')
            ax.fill_between(aligned_time_misses, 
                           avg_speed_misses - sem_speed_misses,
                           avg_speed_misses + sem_speed_misses,
                           alpha=0.2, color='deepskyblue')
            
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Reward Zone Onset (t=0)')
            ax.set_xlabel('Time from Reward Zone Onset (s)')
            ax.set_ylabel('Treadmill Speed (cm/s)')
            ax.set_title(f'Treadmill Speed Comparison: Rewarded (n={len(hits)}) vs Non-Rewarded (n={len(misses)}) Zones')
            ax.legend()
            ax.set_xlim(-5, 5)
            ax.set_ylim(bottom=0)
            ax.set_xticks(np.arange(-5, 6, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            save_figure(fig, 'speed_comparison_hits_vs_misses', output_folder)
            plt.show()
    
    print(f"\n{'='*70}")
    print("HITS VS MISSES ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


def test_speed_bimodality_hits_vs_misses(reward_zone_trials, cap_time, cap_val, speed_time, speed_val, output_folder):
    """Test if speed distribution is truly bimodal or if hit/miss classification creates artificial groups
    
    This function tests the hypothesis that there are two distinct behavioral strategies (fast vs slow)
    rather than the hit/miss classification artificially creating two groups. Uses multiple statistical
    approaches:
    1. Visual inspection of distribution (histogram)
    2. Bimodality coefficient (BC > 0.555 suggests bimodality)
    3. Gaussian Mixture Model comparison (1 vs 2 components using BIC)
    4. Cluster purity analysis (how well natural clusters align with hit/miss labels)
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry_time, reward_event_time) tuples
        cap_time: Capacitive time array (for time reference)
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        output_folder: Directory to save figures
    """
    if len(reward_zone_trials) == 0:
        print("No reward zone trials available for bimodality analysis.")
        return
    
    print(f"\n{'='*70}")
    print("TESTING FOR BIMODALITY IN SPEED DISTRIBUTION")
    print("Question: Are fast/slow speeds real behavioral strategies or")
    print("          artifacts of hit/miss classification?")
    print(f"{'='*70}")
    
    # Separate hits and misses
    hits, misses = separate_hits_and_misses(reward_zone_trials)
    
    # ========================================================================
    # EXTRACT AVERAGE SPEED IN 2S POST-ZONE-ENTRY FOR EACH TRIAL
    # ========================================================================
    
    all_speeds = []
    all_labels = []  # 0 = miss, 1 = hit
    
    # Process hits
    for trial_idx, zone_entry_time in hits:
        mask = (speed_time >= zone_entry_time) & (speed_time <= zone_entry_time + 2.0)
        speed_segment = speed_val[mask]
        if len(speed_segment) > 0:
            all_speeds.append(np.nanmean(speed_segment))
            all_labels.append(1)  # hit
    
    # Process misses
    for trial_idx, zone_entry_time in misses:
        mask = (speed_time >= zone_entry_time) & (speed_time <= zone_entry_time + 2.0)
        speed_segment = speed_val[mask]
        if len(speed_segment) > 0:
            all_speeds.append(np.nanmean(speed_segment))
            all_labels.append(0)  # miss
    
    all_speeds = np.array(all_speeds)
    all_labels = np.array(all_labels)
    
    if len(all_speeds) < 4:
        print("Insufficient data for bimodality analysis (need at least 4 trials)")
        return
    
    print(f"\nTotal trials analyzed: {len(all_speeds)}")
    print(f"  Hits: {np.sum(all_labels == 1)}")
    print(f"  Misses: {np.sum(all_labels == 0)}")
    print(f"\nOverall speed statistics (2s post-zone-entry):")
    print(f"  Mean: {np.mean(all_speeds):.2f} cm/s")
    print(f"  Std: {np.std(all_speeds, ddof=1):.2f} cm/s")
    print(f"  Range: {np.min(all_speeds):.2f} - {np.max(all_speeds):.2f} cm/s")
    
    # ========================================================================
    # TEST 1: BIMODALITY COEFFICIENT
    # ========================================================================
    
    # Bimodality coefficient: BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    # BC > 0.555 suggests bimodal distribution
    # BC < 0.555 suggests unimodal distribution
    
    n = len(all_speeds)
    skewness = skew(all_speeds)
    kurt = kurtosis(all_speeds, fisher=True)  # Excess kurtosis
    
    # Calculate bimodality coefficient
    bc_numerator = skewness**2 + 1
    bc_denominator = kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3))
    bimodality_coef = bc_numerator / bc_denominator
    
    print(f"\n--- TEST 1: Bimodality Coefficient ---")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis (excess): {kurt:.4f}")
    print(f"Bimodality Coefficient: {bimodality_coef:.4f}")
    if bimodality_coef > 0.555:
        print(f"  → BIMODAL: BC > 0.555 suggests TWO distinct modes")
        print(f"  → Evidence for REAL behavioral strategies (fast vs slow)")
    else:
        print(f"  → UNIMODAL: BC < 0.555 suggests ONE mode")
        print(f"  → May be artificial grouping from hit/miss classification")
    
    # ========================================================================
    # TEST 2: GAUSSIAN MIXTURE MODEL COMPARISON
    # ========================================================================
    
    print(f"\n--- TEST 2: Gaussian Mixture Model Comparison ---")
    
    # Reshape for sklearn
    X = all_speeds.reshape(-1, 1)
    
    # Fit 1-component model (unimodal)
    gmm1 = GaussianMixture(n_components=1, random_state=42, n_init=10, init_params='random')
    gmm1.fit(X)
    bic1 = gmm1.bic(X)
    aic1 = gmm1.aic(X)
    
    # Fit 2-component model (bimodal)
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10, init_params='random')
    gmm2.fit(X)
    bic2 = gmm2.bic(X)
    aic2 = gmm2.aic(X)
    
    print(f"1-component model (unimodal):")
    print(f"  BIC: {bic1:.2f}")
    print(f"  AIC: {aic1:.2f}")
    print(f"\n2-component model (bimodal):")
    print(f"  BIC: {bic2:.2f}")
    print(f"  AIC: {aic2:.2f}")
    print(f"\nΔBIC (2-comp - 1-comp): {bic2 - bic1:.2f}")
    print(f"ΔAIC (2-comp - 1-comp): {aic2 - aic1:.2f}")
    
    if bic2 < bic1:
        print(f"  → BIMODAL: 2-component model preferred (lower BIC)")
        print(f"  → Evidence for REAL behavioral strategies")
    else:
        print(f"  → UNIMODAL: 1-component model preferred (lower BIC)")
        print(f"  → May be artificial grouping from hit/miss classification")
    
    # Extract cluster assignments from 2-component model
    cluster_labels = gmm2.predict(X)
    
    # Get cluster means to determine which cluster is "fast" vs "slow"
    cluster0_mean = np.mean(all_speeds[cluster_labels == 0])
    cluster1_mean = np.mean(all_speeds[cluster_labels == 1])
    
    # Assign semantic labels
    if cluster0_mean < cluster1_mean:
        slow_cluster = 0
        fast_cluster = 1
    else:
        slow_cluster = 1
        fast_cluster = 0
    
    print(f"\n2-component model cluster means:")
    print(f"  Slow cluster: {min(cluster0_mean, cluster1_mean):.2f} cm/s")
    print(f"  Fast cluster: {max(cluster0_mean, cluster1_mean):.2f} cm/s")
    
    # Diagnostic: Show speed ranges within each cluster
    slow_cluster_speeds_diag = all_speeds[cluster_labels == slow_cluster]
    fast_cluster_speeds_diag = all_speeds[cluster_labels == fast_cluster]
    
    print(f"\nCluster speed ranges (DIAGNOSTIC):")
    print(f"  Slow cluster: {np.min(slow_cluster_speeds_diag):.2f} - {np.max(slow_cluster_speeds_diag):.2f} cm/s")
    print(f"  Fast cluster: {np.min(fast_cluster_speeds_diag):.2f} - {np.max(fast_cluster_speeds_diag):.2f} cm/s")
    print(f"  Overlap range: {max(np.min(slow_cluster_speeds_diag), np.min(fast_cluster_speeds_diag)):.2f} - "
          f"{min(np.max(slow_cluster_speeds_diag), np.max(fast_cluster_speeds_diag)):.2f} cm/s")
    
    # Show percentiles
    print(f"\nCluster percentiles:")
    print(f"  Slow cluster: 25th={np.percentile(slow_cluster_speeds_diag, 25):.2f}, "
          f"50th={np.percentile(slow_cluster_speeds_diag, 50):.2f}, "
          f"75th={np.percentile(slow_cluster_speeds_diag, 75):.2f}")
    print(f"  Fast cluster: 25th={np.percentile(fast_cluster_speeds_diag, 25):.2f}, "
          f"50th={np.percentile(fast_cluster_speeds_diag, 50):.2f}, "
          f"75th={np.percentile(fast_cluster_speeds_diag, 75):.2f}")
    
    # ========================================================================
    # TEST 3: CLUSTER PURITY ANALYSIS
    # ========================================================================
    
    print(f"\n--- TEST 3: Cluster Purity Analysis ---")
    print("If clusters align well with hit/miss labels, it suggests")
    print("the behavioral strategies (fast/slow) CAUSE the outcomes (hit/miss)")
    
    # Create confusion matrix: cluster assignment vs hit/miss
    slow_and_hit = np.sum((cluster_labels == slow_cluster) & (all_labels == 1))
    slow_and_miss = np.sum((cluster_labels == slow_cluster) & (all_labels == 0))
    fast_and_hit = np.sum((cluster_labels == fast_cluster) & (all_labels == 1))
    fast_and_miss = np.sum((cluster_labels == fast_cluster) & (all_labels == 0))
    
    total_slow = slow_and_hit + slow_and_miss
    total_fast = fast_and_hit + fast_and_miss
    
    print(f"\nCluster assignment vs Hit/Miss outcomes:")
    print(f"                    Hits    Misses   Total")
    print(f"  Slow cluster:     {slow_and_hit:4d}    {slow_and_miss:4d}     {total_slow:4d}")
    print(f"  Fast cluster:     {fast_and_hit:4d}    {fast_and_miss:4d}     {total_fast:4d}")
    
    # Calculate purity metrics
    slow_purity_hit = slow_and_hit / total_slow if total_slow > 0 else 0
    slow_purity_miss = slow_and_miss / total_slow if total_slow > 0 else 0
    fast_purity_hit = fast_and_hit / total_fast if total_fast > 0 else 0
    fast_purity_miss = fast_and_miss / total_fast if total_fast > 0 else 0
    
    print(f"\nCluster composition:")
    print(f"  Slow cluster: {slow_purity_hit*100:.1f}% hits, {slow_purity_miss*100:.1f}% misses")
    print(f"  Fast cluster: {fast_purity_hit*100:.1f}% hits, {fast_purity_miss*100:.1f}% misses")
    
    # Overall purity (maximum class in each cluster)
    overall_purity = (max(slow_and_hit, slow_and_miss) + max(fast_and_hit, fast_and_miss)) / len(all_speeds)
    print(f"\nOverall cluster purity: {overall_purity*100:.1f}%")
    
    if overall_purity > 0.75:
        print(f"  → HIGH purity: Clusters strongly align with hit/miss outcomes")
        print(f"  → Speed strategy appears to DETERMINE outcome")
    elif overall_purity > 0.60:
        print(f"  → MODERATE purity: Some alignment between clusters and outcomes")
        print(f"  → Speed strategy partially INFLUENCES outcome")
    else:
        print(f"  → LOW purity: Weak alignment between clusters and outcomes")
        print(f"  → Hit/miss classification may not reflect speed strategies")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram with overlaid distributions
    ax1 = axes[0, 0]
    bins = np.linspace(np.min(all_speeds), np.max(all_speeds), 30)
    
    # Plot histogram
    ax1.hist(all_speeds, bins=bins, alpha=0.6, color='gray', edgecolor='black', label='All trials')
    
    # Overlay fitted distributions
    x_range = np.linspace(np.min(all_speeds), np.max(all_speeds), 200).reshape(-1, 1)
    
    # 1-component model
    logprob1 = gmm1.score_samples(x_range)
    pdf1 = np.exp(logprob1) * len(all_speeds) * (bins[1] - bins[0])
    ax1.plot(x_range, pdf1, 'b--', linewidth=2, label=f'1-component (BIC={bic1:.0f})')
    
    # 2-component model
    logprob2 = gmm2.score_samples(x_range)
    pdf2 = np.exp(logprob2) * len(all_speeds) * (bins[1] - bins[0])
    ax1.plot(x_range, pdf2, 'r-', linewidth=2, label=f'2-component (BIC={bic2:.0f})')
    
    ax1.set_xlabel('Average Speed (cm/s, 2s post-zone-entry)')
    ax1.set_ylabel('Count')
    ax1.set_title('Speed Distribution: Model Comparison')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Histogram colored by hit/miss
    ax2 = axes[0, 1]
    hit_speeds = all_speeds[all_labels == 1]
    miss_speeds = all_speeds[all_labels == 0]
    
    ax2.hist(hit_speeds, bins=bins, alpha=0.6, color='darkblue', edgecolor='black', label=f'Hits (n={len(hit_speeds)})')
    ax2.hist(miss_speeds, bins=bins, alpha=0.6, color='deepskyblue', edgecolor='black', label=f'Misses (n={len(miss_speeds)})')
    ax2.set_xlabel('Average Speed (cm/s, 2s post-zone-entry)')
    ax2.set_ylabel('Count')
    ax2.set_title('Speed Distribution: Hit/Miss Classification')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Histogram colored by unsupervised clusters
    ax3 = axes[1, 0]
    slow_cluster_speeds = all_speeds[cluster_labels == slow_cluster]
    fast_cluster_speeds = all_speeds[cluster_labels == fast_cluster]
    
    ax3.hist(slow_cluster_speeds, bins=bins, alpha=0.6, color='green', edgecolor='black', 
             label=f'Slow cluster (n={len(slow_cluster_speeds)})')
    ax3.hist(fast_cluster_speeds, bins=bins, alpha=0.6, color='orange', edgecolor='black',
             label=f'Fast cluster (n={len(fast_cluster_speeds)})')
    ax3.set_xlabel('Average Speed (cm/s, 2s post-zone-entry)')
    ax3.set_ylabel('Count')
    ax3.set_title('Speed Distribution: Unsupervised Clustering (GMM)')
    ax3.legend()
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Scatter plot showing cluster vs hit/miss relationship
    ax4 = axes[1, 1]
    
    # Create 4 groups for visualization
    slow_hit_speeds = all_speeds[(cluster_labels == slow_cluster) & (all_labels == 1)]
    slow_miss_speeds = all_speeds[(cluster_labels == slow_cluster) & (all_labels == 0)]
    fast_hit_speeds = all_speeds[(cluster_labels == fast_cluster) & (all_labels == 1)]
    fast_miss_speeds = all_speeds[(cluster_labels == fast_cluster) & (all_labels == 0)]
    
    # Plot as jittered scatter
    np.random.seed(42)
    jitter = 0.1
    
    if len(slow_hit_speeds) > 0:
        ax4.scatter(np.random.normal(0, jitter, len(slow_hit_speeds)), slow_hit_speeds, 
                   color='darkgreen', s=50, alpha=0.6, label=f'Slow+Hit (n={len(slow_hit_speeds)})')
    if len(slow_miss_speeds) > 0:
        ax4.scatter(np.random.normal(1, jitter, len(slow_miss_speeds)), slow_miss_speeds,
                   color='lightgreen', s=50, alpha=0.6, label=f'Slow+Miss (n={len(slow_miss_speeds)})')
    if len(fast_hit_speeds) > 0:
        ax4.scatter(np.random.normal(2, jitter, len(fast_hit_speeds)), fast_hit_speeds,
                   color='darkorange', s=50, alpha=0.6, label=f'Fast+Hit (n={len(fast_hit_speeds)})')
    if len(fast_miss_speeds) > 0:
        ax4.scatter(np.random.normal(3, jitter, len(fast_miss_speeds)), fast_miss_speeds,
                   color='gold', s=50, alpha=0.6, label=f'Fast+Miss (n={len(fast_miss_speeds)})')
    
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_xticklabels(['Slow\n+Hit', 'Slow\n+Miss', 'Fast\n+Hit', 'Fast\n+Miss'])
    ax4.set_ylabel('Speed (cm/s)')
    ax4.set_title('Cluster-Outcome Relationship')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_figure(fig, 'speed_bimodality_analysis', output_folder)
    plt.show()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("SUMMARY: BIMODALITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Bimodality Coefficient: {bimodality_coef:.4f} ({'BIMODAL' if bimodality_coef > 0.555 else 'UNIMODAL'})")
    print(f"Best model by BIC: {'2-component (BIMODAL)' if bic2 < bic1 else '1-component (UNIMODAL)'}")
    print(f"Cluster purity: {overall_purity*100:.1f}%")
    
    # Overall interpretation
    bimodal_evidence_count = 0
    if bimodality_coef > 0.555:
        bimodal_evidence_count += 1
    if bic2 < bic1:
        bimodal_evidence_count += 1
    if overall_purity > 0.65:
        bimodal_evidence_count += 1
    
    print(f"\n--- INTERPRETATION ---")
    if bimodal_evidence_count >= 2:
        print("✓ STRONG EVIDENCE FOR REAL BEHAVIORAL STRATEGIES")
        print("  The fast/slow speed dichotomy appears to be a genuine behavioral")
        print("  phenomenon, not an artifact of hit/miss classification.")
        print("  Interpretation: Mice use distinct movement strategies (fast vs slow)")
        print("  and these strategies determine whether they get rewarded (hit vs miss).")
    elif bimodal_evidence_count == 1:
        print("~ MIXED EVIDENCE")
        print("  Some tests support bimodality, others support unimodality.")
        print("  The fast/slow distinction may be partially real but exaggerated")
        print("  by the hit/miss classification.")
    else:
        print("✗ WEAK EVIDENCE FOR DISTINCT STRATEGIES")
        print("  The fast/slow speed groups may be largely an artifact of")
        print("  hit/miss classification rather than true behavioral strategies.")
        print("  Interpretation: Speed exists on a continuum, and the hit/miss")
        print("  classification imposes an artificial dichotomy.")
    
    print(f"{'='*70}\n")


# ============================================================================
# ANALYSIS FUNCTIONS: PUFF ZONES
# ============================================================================

def analyze_puff_zones(puff_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                       pupil_diameter_data, output_folder, window=10, cap_vmin=0, cap_vmax=5000):
    """Analyze data aligned to puff zone entries and deliveries
    
    Args:
        puff_zone_trials: List of (trial_idx, zone_entry, puff_event) tuples
        trial_log_df: Trial log DataFrame (needed to get ALL puff events)
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        pupil_diameter_data: Tuple of (pupil_time, pupil_val) or (None, None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmin: Minimum value for capacitive colormap scale
        cap_vmax: Maximum value for capacitive colormap scale
    """
    if len(puff_zone_trials) == 0:
        print("No puff zones found. Skipping puff zone analysis.")
        return
    
    print(f"\n=== ANALYZING {len(puff_zone_trials)} PUFF ZONES ===")
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in puff_zone_trials]
    
    # Create aligned windows
    speed_windows, aligned_time_speed = create_aligned_windows(
        speed_time, speed_val, zone_entry_times, window
    )
    
    cap_windows, aligned_time_cap = create_aligned_windows(
        cap_time, cap_val, zone_entry_times, window
    )
    
    # Plot rasters for zone entry
    if speed_windows is not None:
        plot_raster_heatmap(
            speed_windows, aligned_time_speed, puff_zone_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Puff Zone Entry (n={len(puff_zone_trials)} trials)',
            'Treadmill Speed (cm/s)', 'coolwarm', output_folder,
            'treadmill_speed_raster_puff_zones',
            vmin=-30,
            vmax=30,
            center_time=0, event_label="Puff Zone Entry",
            show_delivery_markers=True, center_line_color='black'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, puff_zone_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Puff Zone Entry (n={len(puff_zone_trials)} trials)',
            'Capacitive Value (a.u.)', 'binary', output_folder,
            'capacitive_raster_puff_zones',
            vmin=cap_vmin,
            vmax=cap_vmax,
            center_time=0, event_label="Puff Zone Entry",
            show_delivery_markers=True, center_line_color='blue'
        )
    
    # Analyze puff deliveries
    puff_delivery_trials = [(t, z, p) for t, z, p in puff_zone_trials 
                           if pd.notna(p) and p > 0]
    
    if len(puff_delivery_trials) > 0:
        print(f"Analyzing puff deliveries...")
        # Note: Pass puff_zone_trials and trial_log_df so we can include ALL puff events
        analyze_puff_deliveries(puff_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                               output_folder, window, cap_vmin, cap_vmax)


def analyze_puff_deliveries(puff_zone_trials, trial_log_df, cap_time, cap_val, speed_time, speed_val,
                            output_folder, window=10, cap_vmin=0, cap_vmax=5000):
    """Analyze data aligned to puff delivery times
    
    Args:
        puff_zone_trials: List of puff zone trials (trial_idx, zone_entry, puff_event)
                         Used to map puff events to their zone entries when available
        trial_log_df: Trial log DataFrame (to get ALL puff events)
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_time: Treadmill speed time array
        speed_val: Treadmill speed value array
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmin: Minimum value for capacitive colormap scale
        cap_vmax: Maximum value for capacitive colormap scale
    """
    # Check if puff_event column exists
    if 'puff_event' not in trial_log_df.columns:
        print("No puff_event column in trial log")
        return
    
    # Get ALL puff events from trial log (not just matched ones)
    all_puff_events = []
    for trial_idx in range(len(trial_log_df)):
        puff_event = trial_log_df.iloc[trial_idx]['puff_event']
        if pd.notna(puff_event) and puff_event > 0:
            all_puff_events.append((trial_idx, puff_event))
    
    if len(all_puff_events) == 0:
        print("No puff delivery events found in trial log")
        return
    
    print(f"Found {len(all_puff_events)} total puff delivery events in trial log")
    
    # Create a mapping from puff event times to zone entry times (for matched events)
    puff_to_zone_map = {}
    for trial_idx, zone_entry_time, puff_event_time in puff_zone_trials:
        if pd.notna(puff_event_time) and puff_event_time > 0:
            puff_to_zone_map[puff_event_time] = zone_entry_time
    
    # Build complete list for plotting: (trial_idx, zone_entry_time_or_nan, puff_event_time)
    all_puff_delivery_trials = []
    for trial_idx, puff_event_time in all_puff_events:
        zone_entry_time = puff_to_zone_map.get(puff_event_time, np.nan)
        all_puff_delivery_trials.append((trial_idx, zone_entry_time, puff_event_time))
    
    # Extract puff delivery times
    puff_times = [p for _, _, p in all_puff_delivery_trials]
    
    # Create aligned windows
    speed_windows, aligned_time_speed = create_aligned_windows(
        speed_time, speed_val, puff_times, window
    )
    
    cap_windows, aligned_time_cap = create_aligned_windows(
        cap_time, cap_val, puff_times, window
    )
    
    # Plot rasters centered at puff delivery
    if speed_windows is not None:
        plot_raster_heatmap(
            speed_windows, aligned_time_speed, all_puff_delivery_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Puff Delivery (n={len(all_puff_delivery_trials)} trials)',
            'Treadmill Speed (cm/s)', 'coolwarm', output_folder,
            'treadmill_speed_raster_puff_delivery_centered',
            vmin=-40,
            vmax=40,
            center_time=0, event_label="Puff Delivery",
            show_zone_entries=True, zone_entry_color='black', center_line_color='green'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, all_puff_delivery_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Puff Delivery (n={len(all_puff_delivery_trials)} trials)',
            'Capacitive Value (a.u.)', 'binary', output_folder,
            'capacitive_raster_puff_delivery_centered',
            vmin=cap_vmin,
            vmax=cap_vmax,
            center_time=0, event_label="Puff Delivery",
            show_zone_entries=True, zone_entry_color='blue', center_line_color='green'
        )


# ============================================================================
# HELPER FUNCTIONS: STATISTICAL ANALYSIS
# ============================================================================

def get_significance_stars(p_value):
    """Convert p-value to significance stars
    
    Args:
        p_value: P-value from statistical test
        
    Returns:
        str: Significance stars or empty string
    """
    if pd.isna(p_value):
        return ''
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


# ============================================================================
# ANALYSIS FUNCTIONS: PROBE EVENTS
# ============================================================================

def analyze_probe_events(trial_log_df, cap_time, cap_val, speed_time, speed_val, output_folder, window=5, cap_vmax=None):
    """Analyze probe events with treadmill speed and capacitive value alignment
    
    Args:
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed data
        output_folder: Directory for saving figures
        window: Time window in seconds (default: 5)
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    # Check if probe events exist
    if 'probe_time' not in trial_log_df.columns:
        print("No 'probe_time' column found in trial_log_df. Skipping probe analysis.")
        return
    
    probe_event_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    probe_event_times = probe_event_times[~np.isnan(probe_event_times)]
    
    if len(probe_event_times) == 0:
        print("No probe events found in the data. Skipping probe analysis.")
        return
    
    print(f"\nAnalyzing {len(probe_event_times)} probe events...")
    
    # Convert probe times to numpy array
    probe_event_times = np.array(probe_event_times, dtype=float)
    
    # --- Capacitive Value aligned to probe events ---
    cap_probe_windows = []
    for pt in probe_event_times:
        mask = (cap_time >= pt - window) & (cap_time <= pt + window)
        cap_segment = cap_val[mask]
        cap_probe_windows.append(cap_segment)
    
    # Pad all segments to the same length
    max_probe_len = max(len(seg) for seg in cap_probe_windows) if cap_probe_windows else 0
    
    if max_probe_len == 0:
        print("No valid probe event data found for alignment analysis.")
        return
    
    cap_probe_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_probe_len - len(seg)), constant_values=np.nan)
        for seg in cap_probe_windows
    ])
    
    # --- Treadmill Speed aligned to probe events ---
    speed_probe_windows = []
    for pt in probe_event_times:
        mask = (speed_time >= pt - window) & (speed_time <= pt + window)
        speed_segment = speed_val[mask]
        speed_probe_windows.append(speed_segment)
    
    # Pad speed segments to the same length
    speed_probe_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_probe_len - len(seg)), constant_values=np.nan)
        for seg in speed_probe_windows
    ])
    
    # Create a common time axis centered at 0
    aligned_time_probe = np.linspace(-window, window, max_probe_len)
    
    # --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to probe events ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    n_probes = cap_probe_windows_padded.shape[0]
    
    # --- Plot 1: Treadmill Speed aligned to probe events ---
    mean_speed_probe = np.nanmean(speed_probe_windows_padded, axis=0)
    sem_speed_probe = np.nanstd(speed_probe_windows_padded, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(speed_probe_windows_padded), axis=0)
    )
    axs[0].plot(aligned_time_probe, mean_speed_probe, color='purple', label=f'Mean Speed (n={n_probes})', linewidth=1.5)
    axs[0].fill_between(
        aligned_time_probe, 
        mean_speed_probe - sem_speed_probe, 
        mean_speed_probe + sem_speed_probe,
        color='purple', alpha=0.2, label='SEM'
    )
    axs[0].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)', linewidth=1.5)
    axs[0].set_ylabel('Speed (cm/s)')
    axs[0].set_title('Treadmill Speed Aligned to Probe Events')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(-window, window)
    axs[0].set_ylim(bottom=0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # --- Plot 2: Capacitive Value aligned to probe events ---
    mean_cap_probe = np.nanmean(cap_probe_windows_padded, axis=0)
    sem_cap_probe = np.nanstd(cap_probe_windows_padded, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(cap_probe_windows_padded), axis=0)
    )
    axs[1].plot(aligned_time_probe, mean_cap_probe, color='C0', label=f'Mean Capacitive Value (n={n_probes})', linewidth=1.5)
    axs[1].fill_between(
        aligned_time_probe,
        mean_cap_probe - sem_cap_probe,
        mean_cap_probe + sem_cap_probe,
        color='C0', alpha=0.2, label='SEM'
    )
    axs[1].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)', linewidth=1.5)
    axs[1].set_xlabel('Time from Probe Event (s)')
    axs[1].set_ylabel('Capacitive Value (a.u.)')
    axs[1].set_title('Capacitive Value Aligned to Probe Events')
    axs[1].legend(loc='upper right')
    axs[1].set_xlim(-window, window)
    if cap_vmax is not None:
        axs[1].set_ylim(0, cap_vmax)
    else:
        axs[1].set_ylim(bottom=0)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xticks(np.arange(-window, window + 1, 1))
    
    plt.tight_layout()
    save_figure(fig, "probe_event_analysis", output_folder)
    plt.show()
    
    print(f"Probe event analysis complete: {n_probes} events analyzed")


def analyze_probe_fast_vs_slow(trial_log_df, cap_time, cap_val, speed_time, speed_val, output_folder, window=5):
    """Analyze probe events split by fast vs slow speed in 2s post-probe window
    
    Calculates average speed in the 2 seconds after each probe event, then splits
    probe events into the fastest 50% and slowest 50%. Creates a comparison plot
    showing mean ± SEM for both groups.
    
    Args:
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed data
        output_folder: Directory for saving figures
        window: Time window in seconds for plotting context (default: 5)
    """
    # Check if probe events exist
    if 'probe_time' not in trial_log_df.columns:
        print("No 'probe_time' column found. Skipping probe fast/slow analysis.")
        return
    
    probe_event_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    probe_event_times = np.array(probe_event_times[~np.isnan(probe_event_times)], dtype=float)
    
    if len(probe_event_times) == 0:
        print("No probe events found. Skipping probe fast/slow analysis.")
        return
    
    print(f"\n{'='*70}")
    print(f"ANALYZING PROBE EVENTS: FAST VS SLOW SPEEDS (2s post-probe)")
    print(f"{'='*70}")
    print(f"Total probe events: {len(probe_event_times)}")
    
    # ========================================================================
    # CALCULATE AVERAGE SPEED IN 2S POST-PROBE FOR EACH EVENT
    # ========================================================================
    
    probe_avg_speeds = []
    probe_times_with_data = []
    
    for pt in probe_event_times:
        # Extract 2 seconds after probe event (0 to 2s)
        mask = (speed_time >= pt) & (speed_time <= pt + 2.0)
        speed_segment = speed_val[mask]
        
        if len(speed_segment) > 0:
            avg_speed = np.nanmean(speed_segment)
            probe_avg_speeds.append(avg_speed)
            probe_times_with_data.append(pt)
    
    probe_avg_speeds = np.array(probe_avg_speeds)
    probe_times_with_data = np.array(probe_times_with_data)
    
    if len(probe_avg_speeds) < 2:
        print(f"Insufficient probe events with data ({len(probe_avg_speeds)}). Need at least 2.")
        return
    
    # ========================================================================
    # SPLIT INTO FAST (top 50%) AND SLOW (bottom 50%) TRIALS
    # ========================================================================
    
    # Sort by average speed
    sorted_indices = np.argsort(probe_avg_speeds)
    n_probes = len(probe_avg_speeds)
    n_slow = n_probes // 2
    n_fast = n_probes - n_slow  # Handles odd numbers
    
    slow_indices = sorted_indices[:n_slow]
    fast_indices = sorted_indices[n_slow:]
    
    slow_probe_times = probe_times_with_data[slow_indices]
    fast_probe_times = probe_times_with_data[fast_indices]
    
    slow_avg_speeds = probe_avg_speeds[slow_indices]
    fast_avg_speeds = probe_avg_speeds[fast_indices]
    
    print(f"\nSlow trials (bottom 50%): n={len(slow_probe_times)}")
    print(f"  Mean speed (2s post-probe): {np.mean(slow_avg_speeds):.2f} cm/s")
    print(f"  Range: {np.min(slow_avg_speeds):.2f} - {np.max(slow_avg_speeds):.2f} cm/s")
    
    print(f"\nFast trials (top 50%): n={len(fast_probe_times)}")
    print(f"  Mean speed (2s post-probe): {np.mean(fast_avg_speeds):.2f} cm/s")
    print(f"  Range: {np.min(fast_avg_speeds):.2f} - {np.max(fast_avg_speeds):.2f} cm/s")
    
    # ========================================================================
    # CREATE ALIGNED SPEED WINDOWS FOR SLOW TRIALS
    # ========================================================================
    
    speed_windows_slow = []
    for pt in slow_probe_times:
        mask = (speed_time >= pt - window) & (speed_time <= pt + window)
        speed_segment = speed_val[mask]
        speed_windows_slow.append(speed_segment)
    
    if len(speed_windows_slow) > 0 and max(len(seg) for seg in speed_windows_slow) > 0:
        max_len_slow = max(len(seg) for seg in speed_windows_slow)
        speed_windows_slow_padded = np.array([
            np.pad(seg.astype(float), (0, max_len_slow - len(seg)), constant_values=np.nan)
            for seg in speed_windows_slow
        ])
        aligned_time_slow = np.linspace(-window, window, max_len_slow)
    else:
        speed_windows_slow_padded = None
        aligned_time_slow = None
    
    # ========================================================================
    # CREATE ALIGNED SPEED WINDOWS FOR FAST TRIALS
    # ========================================================================
    
    speed_windows_fast = []
    for pt in fast_probe_times:
        mask = (speed_time >= pt - window) & (speed_time <= pt + window)
        speed_segment = speed_val[mask]
        speed_windows_fast.append(speed_segment)
    
    if len(speed_windows_fast) > 0 and max(len(seg) for seg in speed_windows_fast) > 0:
        max_len_fast = max(len(seg) for seg in speed_windows_fast)
        speed_windows_fast_padded = np.array([
            np.pad(seg.astype(float), (0, max_len_fast - len(seg)), constant_values=np.nan)
            for seg in speed_windows_fast
        ])
        aligned_time_fast = np.linspace(-window, window, max_len_fast)
    else:
        speed_windows_fast_padded = None
        aligned_time_fast = None
    
    # ========================================================================
    # CREATE COMPARISON PLOT: FAST VS SLOW
    # ========================================================================
    
    if speed_windows_slow_padded is not None and speed_windows_fast_padded is not None:
        print(f"\nCreating comparison plot: Fast vs Slow probe trials")
        
        avg_speed_slow = np.nanmean(speed_windows_slow_padded, axis=0)
        sem_speed_slow = np.nanstd(speed_windows_slow_padded, axis=0, ddof=1) / np.sqrt(
            np.sum(~np.isnan(speed_windows_slow_padded), axis=0)
        )
        
        avg_speed_fast = np.nanmean(speed_windows_fast_padded, axis=0)
        sem_speed_fast = np.nanstd(speed_windows_fast_padded, axis=0, ddof=1) / np.sqrt(
            np.sum(~np.isnan(speed_windows_fast_padded), axis=0)
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot slow trials
        ax.plot(aligned_time_slow, avg_speed_slow, color='darkblue', linewidth=2, 
                label=f'Slow Trials (n={len(slow_probe_times)})')
        ax.fill_between(aligned_time_slow, 
                       avg_speed_slow - sem_speed_slow,
                       avg_speed_slow + sem_speed_slow,
                       alpha=0.2, color='darkblue')
        
        # Plot fast trials
        ax.plot(aligned_time_fast, avg_speed_fast, color='deepskyblue', linewidth=2, 
                label=f'Fast Trials (n={len(fast_probe_times)})')
        ax.fill_between(aligned_time_fast, 
                       avg_speed_fast - sem_speed_fast,
                       avg_speed_fast + sem_speed_fast,
                       alpha=0.2, color='deepskyblue')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Probe Event (t=0)')
        ax.set_xlabel('Time from Probe Event (s)')
        ax.set_ylabel('Treadmill Speed (cm/s)')
        ax.set_title(f'Treadmill Speed: Fast (n={len(fast_probe_times)}) vs Slow (n={len(slow_probe_times)}) Probe Trials\n(Split by median speed in 2s post-probe window)')
        ax.legend()
        ax.set_xlim(-window, window)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(-window, window + 1, 1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        save_figure(fig, 'probe_speed_comparison_fast_vs_slow', output_folder)
        plt.show()
        
        print(f"Fast vs Slow comparison plot created successfully")
    else:
        print("Could not create comparison plot: insufficient data")
    
    print(f"\n{'='*70}")
    print("PROBE FAST VS SLOW ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


def match_probe_to_revert_times(trial_log_df, texture_data):
    """Match probe times with texture revert times (approximately 1 second before probe)
    
    Args:
        trial_log_df: Trial log DataFrame
        texture_data: Dictionary with processed texture data
        
    Returns:
        tuple: (probe_revert_array, all_revert_times) or (None, revert_times_array)
    """
    # Collect all texture_revert times from all trials
    all_revert_times = []
    for trial_idx in range(len(trial_log_df)):
        revert_list = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_revert'])
        for revert_time in revert_list:
            if not pd.isna(revert_time) and revert_time != '':
                try:
                    all_revert_times.append(float(revert_time))
                except (ValueError, TypeError):
                    continue
    all_revert_times = np.array(all_revert_times)
    
    # Check if probe_time column exists
    if 'probe_time' not in trial_log_df.columns:
        print("No 'probe_time' column found. Skipping probe-revert matching.")
        return None, all_revert_times
    
    # Extract probe times
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna().values
    
    if len(probe_times) == 0:
        print("No valid probe times found. Skipping probe-revert matching.")
        return None, all_revert_times
    
    if len(all_revert_times) == 0:
        print("No valid texture revert times found. Cannot match with probe times.")
        return None, all_revert_times
    
    # Match each probe_time with closest texture_revert_time ~1 second before
    probe_revert_matches = []
    tolerance = 0.5  # Allow ±0.5 seconds around the 1-second target
    
    for probe_time in probe_times:
        # Find revert times that occur before the probe time
        candidate_reverts = all_revert_times[all_revert_times < probe_time]
        
        if len(candidate_reverts) > 0:
            # Calculate time differences (probe_time - revert_time)
            time_diffs = probe_time - candidate_reverts
            
            # Find revert times approximately 1 second before (within tolerance)
            target_diff = 1.0
            valid_matches = np.abs(time_diffs - target_diff) <= tolerance
            
            if np.any(valid_matches):
                # Get the closest match to exactly 1 second before
                valid_diffs = time_diffs[valid_matches]
                valid_reverts = candidate_reverts[valid_matches]
                closest_idx = np.argmin(np.abs(valid_diffs - target_diff))
                matched_revert = valid_reverts[closest_idx]
                actual_diff = time_diffs[valid_matches][closest_idx]
            else:
                # If no matches within tolerance, find closest revert before probe
                closest_idx = np.argmin(time_diffs)
                matched_revert = candidate_reverts[closest_idx]
                actual_diff = time_diffs[closest_idx]
        else:
            matched_revert = np.nan
            actual_diff = np.nan
        
        probe_revert_matches.append([probe_time, matched_revert, actual_diff])
    
    probe_revert_array = np.array(probe_revert_matches)
    
    successful_matches = np.sum(~np.isnan(probe_revert_array[:, 1]))
    if successful_matches == 0:
        print("Warning: No successful matches between probe times and revert times.")
    
    return probe_revert_array, all_revert_times


def analyze_simulated_probe_events(trial_log_df, probe_revert_array, all_revert_times,
                                   cap_time, cap_val, speed_time, speed_val, output_folder, window=5, cap_vmax=None):
    """Analyze simulated probe events for unpaired revert times
    
    Args:
        trial_log_df: Trial log DataFrame
        probe_revert_array: Array of matched probe-revert pairs
        all_revert_times: All texture revert times
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed
        output_folder: Directory for saving figures
        window: Time window in seconds
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    # Check if probe events exist in data
    if 'probe_time' not in trial_log_df.columns:
        print("No 'probe_time' column found. Skipping simulated probe analysis.")
        return
    
    # Find unpaired revert times
    if probe_revert_array is not None and len(probe_revert_array) > 0:
        # Get matched revert times
        matched_revert_times = probe_revert_array[:, 1]
        matched_revert_times = matched_revert_times[~np.isnan(matched_revert_times)]
        
        # Find unpaired revert times
        unpaired_revert_times = []
        tolerance_match = 1e-6
        
        for revert_time in all_revert_times:
            is_matched = np.any(np.abs(matched_revert_times - revert_time) < tolerance_match)
            if not is_matched:
                unpaired_revert_times.append(revert_time)
        
        unpaired_revert_times = np.array(unpaired_revert_times)
        
        if len(unpaired_revert_times) == 0:
            print("All revert times matched with probe events. No simulated probes needed.")
            return
    elif len(all_revert_times) > 0:
        # No probe analysis performed but we have revert times
        unpaired_revert_times = all_revert_times.copy()
    else:
        print("No revert times found. Skipping simulated probe analysis.")
        return
    
    print(f"\nAnalyzing {len(unpaired_revert_times)} simulated probe events (unpaired reverts)...")
    
    # Create simulated probe times (1 second after each unpaired revert)
    simulated_probe_times = unpaired_revert_times + 1.0
    
    # --- Capacitive Value aligned to simulated probes ---
    cap_sim_windows = []
    for sim_probe_time in simulated_probe_times:
        mask = (cap_time >= sim_probe_time - window) & (cap_time <= sim_probe_time + window)
        cap_segment = cap_val[mask]
        cap_sim_windows.append(cap_segment)
    
    # Check for valid windows
    if not cap_sim_windows or max(len(seg) for seg in cap_sim_windows) == 0:
        print("Warning: No valid data windows for simulated probe analysis.")
        return
    
    # Pad segments
    max_sim_len = max(len(seg) for seg in cap_sim_windows)
    cap_sim_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
        for seg in cap_sim_windows
    ])
    
    # --- Treadmill Speed aligned to simulated probes ---
    speed_sim_windows = []
    for sim_probe_time in simulated_probe_times:
        mask = (speed_time >= sim_probe_time - window) & (speed_time <= sim_probe_time + window)
        speed_segment = speed_val[mask]
        speed_sim_windows.append(speed_segment)
    
    speed_sim_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
        for seg in speed_sim_windows
    ])
    
    # Create aligned time axis
    aligned_time_sim = np.linspace(-window, window, max_sim_len)
    
    # --- Create Combined Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    n_sim_probes = cap_sim_windows_padded.shape[0]
    
    # --- Plot 1: Treadmill Speed ---
    mean_speed_sim = np.nanmean(speed_sim_windows_padded, axis=0)
    sem_speed_sim = np.nanstd(speed_sim_windows_padded, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(speed_sim_windows_padded), axis=0)
    )
    axs[0].plot(aligned_time_sim, mean_speed_sim, color='purple', 
                label=f'Mean Speed (n={n_sim_probes})', linewidth=1.5)
    axs[0].fill_between(
        aligned_time_sim,
        mean_speed_sim - sem_speed_sim,
        mean_speed_sim + sem_speed_sim,
        color='purple', alpha=0.2, label='SEM'
    )
    axs[0].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)', linewidth=1.5)
    axs[0].set_ylabel('Speed (cm/s)')
    axs[0].set_title('Treadmill Speed Aligned to Simulated Probe Events (1s after unpaired reverts)')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(-window, window)
    axs[0].set_ylim(bottom=0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # --- Plot 2: Capacitive Value ---
    mean_cap_sim = np.nanmean(cap_sim_windows_padded, axis=0)
    sem_cap_sim = np.nanstd(cap_sim_windows_padded, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(cap_sim_windows_padded), axis=0)
    )
    axs[1].plot(aligned_time_sim, mean_cap_sim, color='C0', 
                label=f'Mean Capacitive Value (n={n_sim_probes})', linewidth=1.5)
    axs[1].fill_between(
        aligned_time_sim,
        mean_cap_sim - sem_cap_sim,
        mean_cap_sim + sem_cap_sim,
        color='C0', alpha=0.2, label='SEM'
    )
    axs[1].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)', linewidth=1.5)
    axs[1].set_xlabel('Time from Simulated Probe Event (s)')
    axs[1].set_ylabel('Capacitive Value (a.u.)')
    axs[1].set_title('Capacitive Value Aligned to Simulated Probe Events (1s after unpaired reverts)')
    axs[1].legend(loc='upper right')
    axs[1].set_xlim(-window, window)
    if cap_vmax is not None:
        axs[1].set_ylim(0, cap_vmax)
    else:
        axs[1].set_ylim(bottom=0)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_xticks(np.arange(-window, window + 1, 1))
    
    plt.tight_layout()
    save_figure(fig, "simulated_probe_events", output_folder)
    plt.show()
    
    print(f"Simulated probe analysis complete: {n_sim_probes} events analyzed")


def compare_probe_vs_simulated_probe(trial_log_df, probe_revert_array, all_revert_times,
                                     cap_time, cap_val, speed_time, speed_val, output_folder, window=5):
    """Compare probe events vs simulated probe events with t-tests
    
    Analyzes the average speed and capacitance in the 2 seconds after probe events
    compared to 2 seconds after simulated probe events (unpaired reverts + 1s).
    
    Args:
        trial_log_df: Trial log DataFrame
        probe_revert_array: Array of matched probe-revert pairs
        all_revert_times: All texture revert times
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed
        output_folder: Directory for saving figures
        window: Time window in seconds for context (display only)
    """
    # Check if probe events exist
    if 'probe_time' not in trial_log_df.columns:
        print("No 'probe_time' column found. Skipping probe comparison.")
        return
    
    probe_event_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    probe_event_times = np.array(probe_event_times[~np.isnan(probe_event_times)], dtype=float)
    
    if len(probe_event_times) == 0:
        print("No probe events found. Skipping probe comparison.")
        return
    
    # Find unpaired revert times (simulated probes)
    if probe_revert_array is not None and len(probe_revert_array) > 0:
        matched_revert_times = probe_revert_array[:, 1]
        matched_revert_times = matched_revert_times[~np.isnan(matched_revert_times)]
        
        unpaired_revert_times = []
        tolerance_match = 1e-6
        
        for revert_time in all_revert_times:
            is_matched = np.any(np.abs(matched_revert_times - revert_time) < tolerance_match)
            if not is_matched:
                unpaired_revert_times.append(revert_time)
        
        unpaired_revert_times = np.array(unpaired_revert_times)
    elif len(all_revert_times) > 0:
        unpaired_revert_times = all_revert_times.copy()
    else:
        print("No revert times found. Skipping probe comparison.")
        return
    
    if len(unpaired_revert_times) == 0:
        print("No unpaired revert times. Skipping probe comparison.")
        return
    
    # Create simulated probe times (1 second after unpaired reverts)
    simulated_probe_times = unpaired_revert_times + 1.0
    
    print(f"\n=== COMPARING PROBE VS SIMULATED PROBE EVENTS ===")
    print(f"Probe events: {len(probe_event_times)}")
    print(f"Simulated probe events: {len(simulated_probe_times)}")
    
    # ========================================================================
    # EXTRACT 2-SECOND POST-EVENT AVERAGES FOR PROBE EVENTS
    # ========================================================================
    
    probe_speed_2s = []
    probe_cap_2s = []
    
    for pt in probe_event_times:
        # Extract 2 seconds after probe event (0 to 2s)
        speed_mask = (speed_time >= pt) & (speed_time <= pt + 2.0)
        cap_mask = (cap_time >= pt) & (cap_time <= pt + 2.0)
        speed_segment = speed_val[speed_mask]
        cap_segment = cap_val[cap_mask]
        
        if len(speed_segment) > 0:
            probe_speed_2s.append(np.nanmean(speed_segment))
        else:
            probe_speed_2s.append(np.nan)
        
        if len(cap_segment) > 0:
            probe_cap_2s.append(np.nanmean(cap_segment))
        else:
            probe_cap_2s.append(np.nan)
    
    probe_speed_2s = np.array(probe_speed_2s)
    probe_cap_2s = np.array(probe_cap_2s)
    
    # Remove NaN values
    valid_probe_speed = probe_speed_2s[~np.isnan(probe_speed_2s)]
    valid_probe_cap = probe_cap_2s[~np.isnan(probe_cap_2s)]
    
    # ========================================================================
    # EXTRACT 2-SECOND POST-EVENT AVERAGES FOR SIMULATED PROBE EVENTS
    # ========================================================================
    
    sim_probe_speed_2s = []
    sim_probe_cap_2s = []
    
    for sim_pt in simulated_probe_times:
        # Extract 2 seconds after simulated probe event (0 to 2s)
        speed_mask = (speed_time >= sim_pt) & (speed_time <= sim_pt + 2.0)
        cap_mask = (cap_time >= sim_pt) & (cap_time <= sim_pt + 2.0)
        speed_segment = speed_val[speed_mask]
        cap_segment = cap_val[cap_mask]
        
        if len(speed_segment) > 0:
            sim_probe_speed_2s.append(np.nanmean(speed_segment))
        else:
            sim_probe_speed_2s.append(np.nan)
        
        if len(cap_segment) > 0:
            sim_probe_cap_2s.append(np.nanmean(cap_segment))
        else:
            sim_probe_cap_2s.append(np.nan)
    
    sim_probe_speed_2s = np.array(sim_probe_speed_2s)
    sim_probe_cap_2s = np.array(sim_probe_cap_2s)
    
    # Remove NaN values
    valid_sim_speed = sim_probe_speed_2s[~np.isnan(sim_probe_speed_2s)]
    valid_sim_cap = sim_probe_cap_2s[~np.isnan(sim_probe_cap_2s)]
    
    # ========================================================================
    # PERFORM T-TESTS
    # ========================================================================
    
    # Speed comparison
    if len(valid_probe_speed) > 0 and len(valid_sim_speed) > 0:
        # Use independent samples t-test (not paired since different numbers of events)
        if len(valid_probe_speed) >= 2 and len(valid_sim_speed) >= 2:
            speed_t_stat, speed_p_value = ttest_ind(valid_probe_speed, valid_sim_speed)
            print(f"\nSpeed (2s post-event):")
            print(f"  Probe mean: {np.mean(valid_probe_speed):.2f} cm/s (n={len(valid_probe_speed)})")
            print(f"  Simulated probe mean: {np.mean(valid_sim_speed):.2f} cm/s (n={len(valid_sim_speed)})")
            print(f"  t-statistic: {speed_t_stat:.4f}")
            print(f"  p-value: {speed_p_value:.6f}")
        else:
            speed_t_stat = np.nan
            speed_p_value = np.nan
            print(f"\nSpeed: Insufficient data for t-test")
    else:
        speed_t_stat = np.nan
        speed_p_value = np.nan
        print(f"\nSpeed: No valid data")
    
    # Capacitance comparison
    if len(valid_probe_cap) > 0 and len(valid_sim_cap) > 0:
        if len(valid_probe_cap) >= 2 and len(valid_sim_cap) >= 2:
            cap_t_stat, cap_p_value = ttest_ind(valid_probe_cap, valid_sim_cap)
            print(f"\nCapacitance (2s post-event):")
            print(f"  Probe mean: {np.mean(valid_probe_cap):.2f} a.u. (n={len(valid_probe_cap)})")
            print(f"  Simulated probe mean: {np.mean(valid_sim_cap):.2f} a.u. (n={len(valid_sim_cap)})")
            print(f"  t-statistic: {cap_t_stat:.4f}")
            print(f"  p-value: {cap_p_value:.6f}")
        else:
            cap_t_stat = np.nan
            cap_p_value = np.nan
            print(f"\nCapacitance: Insufficient data for t-test")
    else:
        cap_t_stat = np.nan
        cap_p_value = np.nan
        print(f"\nCapacitance: No valid data")
    
    print("=== END PROBE COMPARISON ===")
    
    # ========================================================================
    # CREATE BAR PLOTS WITH T-TEST RESULTS
    # ========================================================================
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Speed Comparison
    if len(valid_probe_speed) > 0 and len(valid_sim_speed) > 0:
        means_speed = [np.mean(valid_probe_speed), np.mean(valid_sim_speed)]
        sems_speed = [
            np.std(valid_probe_speed, ddof=1) / np.sqrt(len(valid_probe_speed)),
            np.std(valid_sim_speed, ddof=1) / np.sqrt(len(valid_sim_speed))
        ]
        x_pos = [0, 1]
        bars = axs[0].bar(x_pos, means_speed, yerr=sems_speed, capsize=5,
                         color=['#9467bd', '#8c564b'], alpha=0.7, edgecolor='black')
        axs[0].set_xticks(x_pos)
        axs[0].set_xticklabels(['Probe Events\n(2s post)', 'Simulated Probes\n(2s post)'])
        axs[0].set_ylabel('Average Speed (cm/s)', fontsize=12)
        axs[0].set_title('Speed: Probe vs Simulated Probe Events', fontsize=12, fontweight='bold')
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_ylim(bottom=0)
        
        # Add significance stars
        sig_stars = get_significance_stars(speed_p_value)
        if sig_stars:
            max_height = max(means_speed[0] + sems_speed[0], means_speed[1] + sems_speed[1])
            axs[0].text(0.5, max_height * 1.1, sig_stars, ha='center', va='bottom', fontsize=20, fontweight='bold')
            axs[0].plot([0, 1], [max_height * 1.05, max_height * 1.05], 'k-', linewidth=1)
        
        # Add n and p-value text
        axs[0].text(0.5, -0.15, 
                   f'n_probe = {len(valid_probe_speed)}, n_sim = {len(valid_sim_speed)}\np = {speed_p_value:.4f}' 
                   if not pd.isna(speed_p_value) else f'n_probe = {len(valid_probe_speed)}, n_sim = {len(valid_sim_speed)}',
                   ha='center', va='top', transform=axs[0].transAxes, fontsize=10)
    else:
        axs[0].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   transform=axs[0].transAxes, fontsize=12)
        axs[0].set_title('Speed: Probe vs Simulated Probe Events', fontsize=12, fontweight='bold')
    
    # Plot 2: Capacitance Comparison
    if len(valid_probe_cap) > 0 and len(valid_sim_cap) > 0:
        means_cap = [np.mean(valid_probe_cap), np.mean(valid_sim_cap)]
        sems_cap = [
            np.std(valid_probe_cap, ddof=1) / np.sqrt(len(valid_probe_cap)),
            np.std(valid_sim_cap, ddof=1) / np.sqrt(len(valid_sim_cap))
        ]
        x_pos = [0, 1]
        bars = axs[1].bar(x_pos, means_cap, yerr=sems_cap, capsize=5,
                         color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
        axs[1].set_xticks(x_pos)
        axs[1].set_xticklabels(['Probe Events\n(2s post)', 'Simulated Probes\n(2s post)'])
        axs[1].set_ylabel('Average Capacitance (a.u.)', fontsize=12)
        axs[1].set_title('Capacitance: Probe vs Simulated Probe Events', fontsize=12, fontweight='bold')
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].set_ylim(bottom=0)
        
        # Add significance stars
        sig_stars = get_significance_stars(cap_p_value)
        if sig_stars:
            max_height = max(means_cap[0] + sems_cap[0], means_cap[1] + sems_cap[1])
            axs[1].text(0.5, max_height * 1.1, sig_stars, ha='center', va='bottom', fontsize=20, fontweight='bold')
            axs[1].plot([0, 1], [max_height * 1.05, max_height * 1.05], 'k-', linewidth=1)
        
        # Add n and p-value text
        axs[1].text(0.5, -0.15,
                   f'n_probe = {len(valid_probe_cap)}, n_sim = {len(valid_sim_cap)}\np = {cap_p_value:.4f}'
                   if not pd.isna(cap_p_value) else f'n_probe = {len(valid_probe_cap)}, n_sim = {len(valid_sim_cap)}',
                   ha='center', va='top', transform=axs[1].transAxes, fontsize=10)
    else:
        axs[1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   transform=axs[1].transAxes, fontsize=12)
        axs[1].set_title('Capacitance: Probe vs Simulated Probe Events', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, "probe_vs_simulated_probe_ttest", output_folder)
    plt.show()
    
    print(f"\nProbe vs Simulated Probe comparison complete.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("BEHAVIORAL TIMELINE ANALYSIS")
    print("=" * 60)
    
    # Step 1: Select and validate folder
    folder_path = select_data_folder()
    if not folder_path:
        print("No folder selected. Exiting...")
        return
    
    file_paths, has_pupil_data = validate_and_find_files(folder_path)
    if file_paths is None:
        return
    
    # Step 2: Load data files
    data = load_data_files(file_paths, has_pupil_data)
    output_folder = create_output_folder(folder_path)
    
    # Step 3: Process texture history
    print("Processing texture history...")
    texture_data = process_texture_history(data['trial_log'])
    
    # Step 4: Uniformly sample all data streams at their native rates
    print("Uniformly sampling capacitive data at 50 Hz...")
    cap_time, cap_val = uniformly_sample_capacitive(data['capacitive'])
    
    print("Uniformly sampling treadmill speed at 50 Hz...")
    speed_time, speed_val = uniformly_sample_treadmill(data['treadmill'])
    
    print("Uniformly sampling treadmill distance at 50 Hz...")
    distance_time, distance_val = uniformly_sample_treadmill_distance(data['treadmill'])
    
    # Step 5: Process pupil data (if available) at 20 Hz
    pupil_diameter_data = (None, None)
    if has_pupil_data:
        print("Processing and uniformly sampling pupil data at 20 Hz...")
        pupil_time, pupil_val = process_pupil_data(
            data['pupil'], data['frame_log']
        )
        pupil_diameter_data = (pupil_time, pupil_val)
    
    # Step 6: Plot main timeline
    print("\nGenerating main timeline plot...")
    plot_main_timeline(
        cap_time, cap_val, speed_time, speed_val, distance_time, distance_val, 
        pupil_diameter_data, data['trial_log'], texture_data, has_pupil_data, output_folder
    )
    
    # Step 7: Match and analyze reward zones
    print("\nAnalyzing reward zones...")
    reward_zone_trials = match_reward_zones_to_events(
        data['trial_log'], texture_data['reward_texture_change_time']
    )
    
    # Initialize capacitive scale with defaults
    cap_vmin, cap_vmax = 0, 5000
    
    if len(reward_zone_trials) > 0:
        cap_vmin, cap_vmax = analyze_reward_zones(
            reward_zone_trials, data['trial_log'], cap_time, cap_val, 
            speed_time, speed_val, pupil_diameter_data, output_folder
        )
        
        # Create average trace plots for rewards
        print("\nCreating reward average trace plots...")
        plot_average_traces_reward(
            reward_zone_trials, data['trial_log'], cap_time, cap_val,
            speed_time, speed_val, pupil_diameter_data, output_folder, cap_vmax=cap_vmax
        )
        
        # Analyze hits vs misses separately (using same logic as daily_analysis.py)
        print("\nAnalyzing reward hits vs misses (speed comparison)...")
        analyze_reward_hits_vs_misses(
            reward_zone_trials, cap_time, cap_val, speed_time, speed_val, output_folder, window=5
        )
        
        # Test for bimodality in speed distribution
        print("\nTesting for bimodality in hit/miss speed distribution...")
        test_speed_bimodality_hits_vs_misses(
            reward_zone_trials, cap_time, cap_val, speed_time, speed_val, output_folder
        )
    
    # Step 8: Match and analyze puff zones (using capacitive scale from reward analysis)
    print("\nAnalyzing puff zones...")
    puff_zone_trials = match_puff_zones_to_events(
        data['trial_log'], texture_data['punish_texture_change_time_first']
    )
    
    if len(puff_zone_trials) > 0:
        analyze_puff_zones(
            puff_zone_trials, data['trial_log'], cap_time, cap_val, speed_time, speed_val,
            pupil_diameter_data, output_folder, window=10, cap_vmin=cap_vmin, cap_vmax=cap_vmax
        )
        
        # Create average trace plots for puffs
        print("\nCreating puff average trace plots...")
        plot_average_traces_puff(
            puff_zone_trials, data['trial_log'], cap_time, cap_val,
            speed_time, speed_val, pupil_diameter_data, output_folder, cap_vmax=cap_vmax
        )
    
    # Step 9: Analyze probe events (using capacitive scale from reward analysis)
    print("\nAnalyzing probe events...")
    analyze_probe_events(
        data['trial_log'], cap_time, cap_val, speed_time, speed_val, output_folder, cap_vmax=cap_vmax
    )
    
    # Step 9b: Analyze probe events split by fast vs slow speeds
    print("\nAnalyzing probe events: fast vs slow trials...")
    analyze_probe_fast_vs_slow(
        data['trial_log'], cap_time, cap_val, speed_time, speed_val, output_folder
    )
    
    # Step 10: Match probe times to revert times and analyze simulated probes
    print("\nMatching probe events to texture revert times...")
    probe_revert_array, all_revert_times = match_probe_to_revert_times(
        data['trial_log'], texture_data
    )
    
    if probe_revert_array is not None or len(all_revert_times) > 0:
        print("\nAnalyzing simulated probe events...")
        analyze_simulated_probe_events(
            data['trial_log'], probe_revert_array, all_revert_times,
            cap_time, cap_val, speed_time, speed_val, output_folder, cap_vmax=cap_vmax
        )
        
        # Step 11: Compare probe vs simulated probe events with t-tests
        print("\nComparing probe vs simulated probe events...")
        compare_probe_vs_simulated_probe(
            data['trial_log'], probe_revert_array, all_revert_times,
            cap_time, cap_val, speed_time, speed_val, output_folder
        )
    
    # Step 12: Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if hasattr(save_figure, 'figure_count'):
        print(f"Total figures saved: {save_figure.figure_count}")
        print(f"Output directory: {output_folder}")
    print("\nAll plots displayed and saved as SVG files.")


if __name__ == "__main__":
    main()
