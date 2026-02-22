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
    # Check if required columns exist (new column names)
    required_cols = ['texture_history', 'stay_texture_change_time', 'stay_texture_revert_time', 
                     'go_texture_change_time', 'go_texture_revert_time']
    missing_cols = [col for col in required_cols if col not in trial_log_df.columns]
    
    if missing_cols:
        print(f"Warning: Missing texture columns: {missing_cols}. Creating empty arrays.")
        return create_empty_texture_arrays(trial_log_df)
    
    # Parse texture history column
    texture_history = trial_log_df['texture_history'].apply(safe_literal_eval)
    
    # Parse stay zone (reward) columns
    stay_texture_change_time = trial_log_df['stay_texture_change_time'].apply(safe_literal_eval)
    stay_texture_revert_time = trial_log_df['stay_texture_revert_time'].apply(safe_literal_eval)
    
    # Parse go zone (punish) columns
    go_texture_change_time = trial_log_df['go_texture_change_time'].apply(safe_literal_eval)
    go_texture_revert_time = trial_log_df['go_texture_revert_time'].apply(safe_literal_eval)
    
    # Check if there are any texture changes (handle NaN and empty cases)
    try:
        max_hist_len = texture_history.apply(len).max()
        if pd.isna(max_hist_len):
            max_hist_len = 0
    except (ValueError, TypeError):
        max_hist_len = 0
    
    try:
        max_stay_change_len = stay_texture_change_time.apply(len).max()
        if pd.isna(max_stay_change_len):
            max_stay_change_len = 0
    except (ValueError, TypeError):
        max_stay_change_len = 0
    
    try:
        max_stay_revert_len = stay_texture_revert_time.apply(len).max()
        if pd.isna(max_stay_revert_len):
            max_stay_revert_len = 0
    except (ValueError, TypeError):
        max_stay_revert_len = 0
    
    try:
        max_go_change_len = go_texture_change_time.apply(len).max()
        if pd.isna(max_go_change_len):
            max_go_change_len = 0
    except (ValueError, TypeError):
        max_go_change_len = 0
    
    try:
        max_go_revert_len = go_texture_revert_time.apply(len).max()
        if pd.isna(max_go_revert_len):
            max_go_revert_len = 0
    except (ValueError, TypeError):
        max_go_revert_len = 0
    
    has_texture_data = (max_hist_len > 0 or max_stay_change_len > 0 or max_stay_revert_len > 0 or 
                        max_go_change_len > 0 or max_go_revert_len > 0)
    
    if not has_texture_data:
        print("Warning: No texture change data found. Creating empty arrays.")
        return create_empty_texture_arrays(trial_log_df)
    
    # Process reward (stay zone) data
    # Pad arrays to same length for rewards
    max_reward_len = max(
        stay_texture_change_time.apply(len).max() if max_stay_change_len > 0 else 0,
        stay_texture_revert_time.apply(len).max() if max_stay_revert_len > 0 else 0
    )
    
    if max_reward_len > 0:
        reward_texture_change_time_padded = np.array(stay_texture_change_time.apply(lambda x: pad_list(x, max_reward_len)).tolist(), dtype=object)
        reward_revert_time_padded = np.array(stay_texture_revert_time.apply(lambda x: pad_list(x, max_reward_len)).tolist(), dtype=object)
        
        # Filter to rows that actually have reward data
        has_reward_data = (reward_texture_change_time_padded != None).any(axis=1)
        reward_texture_change_time = reward_texture_change_time_padded[has_reward_data]
        reward_revert_time = reward_revert_time_padded[has_reward_data]
        
        # Create reward array (for compatibility with existing code)
        # Build history array for filtered rows only
        reward_indices = np.where(has_reward_data)[0]
        reward_history_padded = np.array(
            [[texture_history.iloc[i][0] if len(texture_history.iloc[i]) > 0 else None] * max_reward_len for i in reward_indices], 
            dtype=object
        )
        
        reward_array = np.stack([reward_history_padded, reward_texture_change_time, reward_revert_time], axis=1)
    else:
        reward_texture_change_time = np.empty((0, 1))
        reward_revert_time = np.empty((0, 1))
        reward_array = np.empty((0, 3, 1))
    
    # Process punish (go zone) data
    # Pad arrays to same length for punish
    max_punish_len = max(
        go_texture_change_time.apply(len).max() if max_go_change_len > 0 else 0,
        go_texture_revert_time.apply(len).max() if max_go_revert_len > 0 else 0
    )
    
    if max_punish_len > 0:
        punish_texture_change_time_padded = np.array(go_texture_change_time.apply(lambda x: pad_list(x, max_punish_len)).tolist())
        punish_revert_time_padded = np.array(go_texture_revert_time.apply(lambda x: pad_list(x, max_punish_len)).tolist())
        
        # Filter to rows that actually have punish data
        has_punish_data = (punish_texture_change_time_padded != None).any(axis=1)
        punish_texture_change_time = punish_texture_change_time_padded[has_punish_data]
        punish_revert_time = punish_revert_time_padded[has_punish_data]
        
        # Create punish array (for compatibility with existing code)
        # Build history array for filtered rows only
        punish_indices = np.where(has_punish_data)[0]
        punish_history_padded = np.array(
            [[texture_history.iloc[i][0] if len(texture_history.iloc[i]) > 0 else None] * max_punish_len for i in punish_indices], 
            dtype=object
        )
        punish_array = np.stack([punish_history_padded, punish_texture_change_time, punish_revert_time], axis=1)
        
        # Create versions using only first puff per zone
        if punish_array.shape[0] > 0 and punish_array.shape[2] > 0:
            punish_texture_change_time_first = punish_array[:, 1, 0]
            punish_revert_time_first = punish_array[:, 2, 0]
        else:
            punish_texture_change_time_first = np.array([])
            punish_revert_time_first = np.array([])
    else:
        punish_texture_change_time = np.empty((0, 1))
        punish_revert_time = np.empty((0, 1))
        punish_array = np.empty((0, 3, 1))
        punish_texture_change_time_first = np.array([])
        punish_revert_time_first = np.array([])
    
    # For backward compatibility, create combined padded arrays
    # Use the maximum length across both types
    max_len = max(max_reward_len, max_punish_len) if (max_reward_len > 0 or max_punish_len > 0) else 1
    
    # Create texture_history_padded as a 2D object array to accommodate strings
    # Each row contains the texture string repeated max_len times (or None if no texture)
    # Extract the first element if list is non-empty, otherwise use None
    texture_history_padded = np.array(
        [[x[0] if len(x) > 0 else None] * max_len for x in texture_history],
        dtype=object
    )
    
    # Combine stay and go times into unified arrays (for backward compatibility)
    all_change_times = []
    all_revert_times = []
    
    for idx in range(len(trial_log_df)):
        stay_times = safe_literal_eval(trial_log_df['stay_texture_change_time'].iloc[idx])
        go_times = safe_literal_eval(trial_log_df['go_texture_change_time'].iloc[idx])
        stay_reverts = safe_literal_eval(trial_log_df['stay_texture_revert_time'].iloc[idx])
        go_reverts = safe_literal_eval(trial_log_df['go_texture_revert_time'].iloc[idx])
        
        # Combine lists, preferring non-empty ones
        if stay_times and len(stay_times) > 0:
            all_change_times.append(stay_times)
        elif go_times and len(go_times) > 0:
            all_change_times.append(go_times)
        else:
            all_change_times.append([])
        
        if stay_reverts and len(stay_reverts) > 0:
            all_revert_times.append(stay_reverts)
        elif go_reverts and len(go_reverts) > 0:
            all_revert_times.append(go_reverts)
        else:
            all_revert_times.append([])
    
    texture_change_time_padded = np.array([pad_list(x, max_len) for x in all_change_times])
    revert_time_padded = np.array([pad_list(x, max_len) for x in all_revert_times])
    
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
        'reward_revert_time': reward_revert_time,
        'trial_log_df': trial_log_df
    }


def create_empty_texture_arrays(trial_log_df):
    """Create empty texture arrays when no texture data is present
    
    Args:
        trial_log_df: Trial log DataFrame (needed for drawing any zones)
    
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
        'reward_revert_time': np.empty((0, 1)),
        'trial_log_df': trial_log_df
    }


def interpolate_treadmill_to_capacitive(treadmill_df, capacitive_df):
    """Interpolate treadmill data to match capacitive timeline
    
    Args:
        treadmill_df: Treadmill DataFrame
        capacitive_df: Capacitive DataFrame
        
    Returns:
        pd.Series: Interpolated treadmill speed in cm/s
    """
    return pd.Series(
        data=np.interp(
            capacitive_df['elapsed_time'],
            treadmill_df['global_time'],
            treadmill_df['speed']
        ) / 10.0,
        index=capacitive_df['elapsed_time']
    )


def interpolate_treadmill_distance_to_capacitive(treadmill_df, capacitive_df):
    """Interpolate treadmill distance to match capacitive timeline
    
    Finds the first non-zero distance value and subtracts it from all distances
    to get the distance moved in meters.
    
    Args:
        treadmill_df: Treadmill DataFrame
        capacitive_df: Capacitive DataFrame
        
    Returns:
        pd.Series: Interpolated treadmill distance moved in meters
    """
    # Find the first non-zero distance value
    non_zero_distances = treadmill_df['distance'][treadmill_df['distance'] != 0]
    
    if len(non_zero_distances) > 0:
        start_distance = non_zero_distances.iloc[0]
    else:
        # If all distances are zero, use 0 as start distance
        start_distance = 0
    
    # Create adjusted distance values (distance moved from start)
    distance_moved = treadmill_df['distance'] - start_distance
    
    # Interpolate to capacitive timeline and convert to meters
    return pd.Series(
        data=np.interp(
            capacitive_df['elapsed_time'],
            treadmill_df['global_time'],
            distance_moved
        ) / 1000.0,
        index=capacitive_df['elapsed_time']
    )


def process_pupil_data(pupil_df, frame_log_df, capacitive_df):
    """Process pupil diameter data and interpolate to capacitive timeline
    
    Args:
        pupil_df: Pupil tracking DataFrame
        frame_log_df: Frame log DataFrame with timestamps
        capacitive_df: Capacitive DataFrame for alignment
        
    Returns:
        pd.Series or None: Interpolated pupil diameter or None if processing fails
    """
    if pupil_df is None or frame_log_df is None:
        return None
    
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
    
    # Interpolate to capacitive timeline
    valid_data_mask = pupil_df['time_seconds'].notna() & pupil_df['pupil_diameter'].notna()
    
    if valid_data_mask.sum() > 1:
        valid_times = pupil_df.loc[valid_data_mask, 'time_seconds'].values
        valid_diameters = pupil_df.loc[valid_data_mask, 'pupil_diameter'].values
        
        # CRITICAL: Sort pupil data by timestamp (required for np.interp)
        sort_indices = np.argsort(valid_times)
        valid_times = valid_times[sort_indices]
        valid_diameters = valid_diameters[sort_indices]
        
        pupil_diameter_interp = pd.Series(
            data=np.interp(
                capacitive_df['elapsed_time'],
                valid_times,
                valid_diameters
            ),
            index=capacitive_df['elapsed_time']
        )
        
        print(f"Pupil data processed: {valid_data_mask.sum()} valid measurements")
        return pupil_diameter_interp
    else:
        print("Warning: Insufficient valid pupil data for interpolation")
        return None


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
    
    # Collect all reward zone entries from stay_texture_change_time column
    # (stay zones are always reward zones, regardless of texture_history)
    all_reward_zones = []
    for trial_idx in range(len(trial_log_df)):
        # Reward zones use stay_texture_change_time
        texture_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['stay_texture_change_time'])
        
        if len(texture_times) > 0:
            for zone_entry_time in texture_times:
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
    # (go zones are always punish zones, regardless of texture_history)
    all_puff_zones = []
    puff_zone_count = 0
    for trial_idx in range(len(trial_log_df)):
        # Check if this trial has go zone (punish) data
        go_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['go_texture_change_time'])
        
        # Only include trials with go zone times
        if len(go_times) > 0 and puff_zone_count < len(punish_texture_change_time_first):
            # Use the corresponding entry from punish_texture_change_time_first
            zone_entry_time = punish_texture_change_time_first[puff_zone_count]
            puff_zone_count += 1
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

def plot_main_timeline(capacitive_df, treadmill_interp, treadmill_distance_interp, pupil_diameter_interp,
                       trial_log_df, texture_data, has_pupil_data, output_folder):
    """Create the main timeline plot with all data streams
    
    Args:
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed
        treadmill_distance_interp: Interpolated treadmill distance
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        trial_log_df: Trial log DataFrame
        texture_data: Dictionary with processed texture data
        has_pupil_data: Whether pupil data is available
        output_folder: Directory to save figures
    """
    # Determine number of plots: treadmill speed, distance, capacitive, and optionally pupil
    num_plots = 4 if has_pupil_data and pupil_diameter_interp is not None else 3
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 12 if num_plots == 4 else 10), sharex=True)
    
    # Ensure axs is always a list
    axs = list(axs) if num_plots > 1 else [axs]
    
    # Get event times
    reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna() if 'puff_event' in trial_log_df.columns else pd.Series([])
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna() if 'probe_time' in trial_log_df.columns else pd.Series([])
    
    # Plot treadmill speed (top subplot)
    plot_treadmill_timeline(axs[0], capacitive_df, treadmill_interp, reward_times, 
                           puff_times, probe_times, texture_data, has_more_plots=True)
    
    # Plot treadmill distance (second subplot)
    plot_treadmill_distance_timeline(axs[1], capacitive_df, treadmill_distance_interp, reward_times,
                                     puff_times, probe_times, texture_data, has_more_plots=True)
    
    # Plot capacitive data (third subplot)
    plot_capacitive_timeline(axs[2], capacitive_df, reward_times, puff_times, probe_times, 
                             texture_data, "Capacitive", show_xlabel=not has_pupil_data)
    
    # Plot pupil data if available (fourth subplot)
    if num_plots == 4 and pupil_diameter_interp is not None:
        plot_pupil_timeline(axs[3], capacitive_df, pupil_diameter_interp, reward_times,
                          puff_times, probe_times, texture_data)
    
    # Set x-axis limits
    xmin = capacitive_df['elapsed_time'].min()
    xmax = capacitive_df['elapsed_time'].max()
    for ax in axs:
        ax.set_xlim([xmin, xmax])
    
    setup_plot_style(axs)
    plt.tight_layout()
    save_figure(fig, f"timeline_{'all_data' if num_plots == 4 else 'speed_distance_capacitive'}", 
                output_folder)
    plt.show()


def plot_capacitive_timeline(ax, capacitive_df, reward_times, puff_times, probe_times, 
                             texture_data, label_prefix="", show_xlabel=True):
    """Plot capacitive sensor data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], 
            label='Capacitive Value (a.u.)', color='C0')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    max_time = capacitive_df['elapsed_time'].max()
    add_texture_intervals(ax, texture_data, max_time)
    
    if show_xlabel:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Capacitive Value (a.u.)')
    ax.set_title(f'{label_prefix} Capacitive Sensor Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def plot_treadmill_timeline(ax, capacitive_df, treadmill_interp, reward_times, 
                           puff_times, probe_times, texture_data, has_more_plots=False):
    """Plot treadmill speed data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], treadmill_interp, 
            label='Treadmill Speed (interpolated)', color='purple')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    max_time = capacitive_df['elapsed_time'].max()
    add_texture_intervals(ax, texture_data, max_time)
    
    if not has_more_plots:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Interpolated Treadmill Speed Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')


def plot_treadmill_distance_timeline(ax, capacitive_df, treadmill_distance_interp, reward_times, 
                                     puff_times, probe_times, texture_data, has_more_plots=False):
    """Plot treadmill distance data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], treadmill_distance_interp, 
            label='Treadmill Distance (interpolated)', color='teal')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    max_time = capacitive_df['elapsed_time'].max()
    add_texture_intervals(ax, texture_data, max_time)
    
    if not has_more_plots:
        ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Interpolated Treadmill Distance Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def plot_pupil_timeline(ax, capacitive_df, pupil_diameter_interp, reward_times,
                       puff_times, probe_times, texture_data):
    """Plot pupil diameter data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], pupil_diameter_interp,
            label='Pupil Diameter (interpolated)', color='orange')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    max_time = capacitive_df['elapsed_time'].max()
    add_texture_intervals(ax, texture_data, max_time)
    
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


def add_texture_intervals(ax, texture_data, max_time):
    """Add shaded regions for reward and punish zones
    
    Draws zones directly from trial_log to ensure ALL zones are displayed,
    including re-entered zones that may not have texture_history entries.
    If a zone has no exit time, it extends to the end of the timeline.
    
    Args:
        ax: Matplotlib axis
        texture_data: Dictionary with texture data including trial_log_df
        max_time: Maximum time on the timeline (for zones without exit times)
    """
    trial_log_df = texture_data['trial_log_df']
    
    # Highlight ALL reward (stay zone) intervals from trial log
    for trial_idx in range(len(trial_log_df)):
        # Get stay zone times (reward zones)
        stay_change_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['stay_texture_change_time'])
        stay_revert_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['stay_texture_revert_time'])
        
        # Draw each stay zone - handle mismatched or missing exit times
        if len(stay_change_times) > 0:
            for i, start_time in enumerate(stay_change_times):
                if pd.notna(start_time) and start_time > 0:
                    # Get corresponding exit time, or use max_time if missing
                    if i < len(stay_revert_times) and pd.notna(stay_revert_times[i]) and stay_revert_times[i] > 0:
                        end_time = stay_revert_times[i]
                    else:
                        end_time = max_time  # Extend to end of timeline
                    ax.axvspan(start_time, end_time, color='green', alpha=0.1)
    
    # Highlight ALL punish (go zone) intervals from trial log
    for trial_idx in range(len(trial_log_df)):
        # Get go zone times (punish zones)
        go_change_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['go_texture_change_time'])
        go_revert_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['go_texture_revert_time'])
        
        # Draw each go zone - handle mismatched or missing exit times
        if len(go_change_times) > 0:
            for i, start_time in enumerate(go_change_times):
                if pd.notna(start_time) and start_time > 0:
                    # Get corresponding exit time, or use max_time if missing
                    if i < len(go_revert_times) and pd.notna(go_revert_times[i]) and go_revert_times[i] > 0:
                        end_time = go_revert_times[i]
                    else:
                        end_time = max_time  # Extend to end of timeline
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

def plot_average_traces_reward(reward_zone_trials, trial_log_df, capacitive_df, 
                               treadmill_interp, pupil_diameter_interp, output_folder, window=5, cap_vmax=None):
    """Plot average traces (mean ± SEM) for reward zone and delivery events
    
    Args:
        reward_zone_trials: List of reward zone trial tuples
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones for average trace analysis")
        return
    
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in reward_zone_trials]
    
    # Create speed windows aligned to zone entry
    speed_windows = []
    for rt in zone_entry_times:
        mask = (cap_time >= rt - window) & (cap_time <= rt + window)
        speed_segment = speed_val[mask]
        speed_windows.append(speed_segment)
    
    max_speed_len = max(len(seg) for seg in speed_windows)
    speed_windows_padded = np.array([
        np.pad(seg.astype(float), (0, max_speed_len - len(seg)), constant_values=np.nan)
        for seg in speed_windows
    ])
    
    aligned_time_speed = np.linspace(-window, window, max_speed_len)
    mean_speed = np.nanmean(speed_windows_padded, axis=0)
    sem_speed = np.nanstd(speed_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_windows_padded), axis=0))
    
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
    sem_event_vals = np.nanstd(cap_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(cap_event_windows_padded), axis=0))
    
    # Create combined subplot figure
    num_plots = 3 if pupil_diameter_interp is not None else 2
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
    if pupil_diameter_interp is not None:
        pupil_val = pupil_diameter_interp.values
        
        # Pupil aligned to zone entry
        pupil_zone_windows = []
        for rt in zone_entry_times:
            mask = (cap_time >= rt - window) & (cap_time <= rt + window)
            pupil_segment = pupil_val[mask]
            pupil_zone_windows.append(pupil_segment)
        
        max_pupil_len = max(len(seg) for seg in pupil_zone_windows)
        pupil_zone_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_len - len(seg)), constant_values=np.nan)
            for seg in pupil_zone_windows
        ])
        
        aligned_time_pupil = np.linspace(-window, window, max_pupil_len)
        mean_pupil = np.nanmean(pupil_zone_windows_padded, axis=0)
        sem_pupil = np.nanstd(pupil_zone_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(pupil_zone_windows_padded), axis=0))
        
        # Pupil aligned to reward events
        pupil_event_windows = []
        for rt in reward_event_times_flat:
            mask = (cap_time >= rt - window) & (cap_time <= rt + window)
            pupil_segment = pupil_val[mask]
            pupil_event_windows.append(pupil_segment)
        
        max_pupil_event_len = max(len(seg) for seg in pupil_event_windows)
        pupil_event_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_event_len - len(seg)), constant_values=np.nan)
            for seg in pupil_event_windows
        ])
        
        aligned_time_pupil_event = np.linspace(-window, window, max_pupil_event_len)
        mean_pupil_event = np.nanmean(pupil_event_windows_padded, axis=0)
        sem_pupil_event = np.nanstd(pupil_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(pupil_event_windows_padded), axis=0))
        
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


def plot_average_traces_puff(puff_zone_trials, trial_log_df, capacitive_df,
                             treadmill_interp, pupil_diameter_interp, output_folder, window=5, cap_vmax=None):
    """Plot average traces (mean ± SEM) for puff zone and delivery events
    
    Args:
        puff_zone_trials: List of puff zone trial tuples
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmax: Maximum value for capacitive y-axis (optional)
    """
    if len(puff_zone_trials) == 0:
        print("No puff zones for average trace analysis")
        return
    
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in puff_zone_trials]
    
    # Speed aligned to puff zone entry
    speed_puff_windows = []
    for puff_time in zone_entry_times:
        mask = (cap_time >= puff_time - window) & (cap_time <= puff_time + window)
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
    sem_speed_puff = np.nanstd(speed_puff_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_puff_windows_padded), axis=0))
    
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
                sem_cap_puff_event = np.nanstd(cap_puff_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(cap_puff_event_windows_padded), axis=0))
                
                puff_event_capacitive_data = {
                    'aligned_time': aligned_time_puff_cap,
                    'mean_values': mean_cap_puff_event,
                    'sem_values': sem_cap_puff_event,
                    'n_events': n_puff_event_cap
                }
            
            # Speed aligned to puff events
            speed_puff_event_windows = []
            for puff_event_time in puff_event_times:
                mask = (cap_time >= puff_event_time - window) & (cap_time <= puff_event_time + window)
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
                sem_speed_puff_event = np.nanstd(speed_puff_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_puff_event_windows_padded), axis=0))
                
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
    axs[2].set_xticks(np.arange(-5, 6, 1))
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, "puff_events_analysis", output_folder)
    plt.show()
    
    # Create pupil plots if available
    if pupil_diameter_interp is not None:
        pupil_val = pupil_diameter_interp.values
        window_pupil = 10
        
        # Pupil aligned to puff zone entry
        pupil_puff_windows = []
        for puff_time in zone_entry_times:
            mask = (cap_time >= puff_time - window_pupil) & (cap_time <= puff_time + window_pupil)
            pupil_segment = pupil_val[mask]
            pupil_puff_windows.append(pupil_segment)
        
        max_pupil_puff_len = max(len(seg) for seg in pupil_puff_windows)
        pupil_puff_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_pupil_puff_len - len(seg)), constant_values=np.nan)
            for seg in pupil_puff_windows
        ])
        
        aligned_time_pupil_puff = np.linspace(-window_pupil, window_pupil, max_pupil_puff_len)
        mean_pupil_puff = np.nanmean(pupil_puff_windows_padded, axis=0)
        sem_pupil_puff = np.nanstd(pupil_puff_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(pupil_puff_windows_padded), axis=0))
        n_puffs_pupil = pupil_puff_windows_padded.shape[0]
        
        # Pupil aligned to puff events (if available)
        pupil_puff_event_data = None
        if 'puff_event' in trial_log_df.columns:
            puff_event_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna().values
            
            if len(puff_event_times) > 0:
                pupil_puff_event_windows = []
                for puff_time in puff_event_times:
                    mask = (cap_time >= puff_time - window_pupil) & (cap_time <= puff_time + window_pupil)
                    pupil_segment = pupil_val[mask]
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
                    sem_pupil_puff_event = np.nanstd(pupil_puff_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(pupil_puff_event_windows_padded), axis=0))
                    
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

def analyze_reward_zones(reward_zone_trials, trial_log_df, capacitive_df, treadmill_interp, 
                         pupil_diameter_interp, output_folder, window=5):
    """Analyze data aligned to reward zone entries and deliveries
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry, reward_event) tuples
        trial_log_df: Trial log DataFrame (needed to get ALL reward events)
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
        
    Returns:
        tuple: (cap_vmin, cap_vmax) - Scale for capacitive plots
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones found. Skipping reward zone analysis.")
        return (0, 5000)  # Default scale
    
    print(f"\n=== ANALYZING {len(reward_zone_trials)} REWARD ZONES ===")
    
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in reward_zone_trials]
    
    # Create aligned windows for zone entries
    speed_windows, aligned_time_speed = create_aligned_windows(
        cap_time, speed_val, zone_entry_times, window
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
            sem_event_vals = np.nanstd(cap_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(cap_event_windows_padded), axis=0))
            
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
    analyze_reward_deliveries(reward_zone_trials, trial_log_df, cap_time, cap_val, speed_val,
                             pupil_diameter_interp, output_folder, window, cap_vmin, cap_vmax)
    
    return (cap_vmin, cap_vmax)


def analyze_reward_deliveries(reward_zone_trials, trial_log_df, cap_time, cap_val, speed_val,
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
        cap_time, speed_val, reward_times, window
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


# ============================================================================
# ANALYSIS FUNCTIONS: PUFF ZONES
# ============================================================================

def analyze_puff_zones(puff_zone_trials, trial_log_df, capacitive_df, treadmill_interp, 
                       pupil_diameter_interp, output_folder, window=10, cap_vmin=0, cap_vmax=5000):
    """Analyze data aligned to puff zone entries and deliveries
    
    Args:
        puff_zone_trials: List of (trial_idx, zone_entry, puff_event) tuples
        trial_log_df: Trial log DataFrame (needed to get ALL puff events)
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
        cap_vmin: Minimum value for capacitive colormap scale
        cap_vmax: Maximum value for capacitive colormap scale
    """
    if len(puff_zone_trials) == 0:
        print("No puff zones found. Skipping puff zone analysis.")
        return
    
    print(f"\n=== ANALYZING {len(puff_zone_trials)} PUFF ZONES ===")
    
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
    # Extract zone entry times
    zone_entry_times = [entry[1] for entry in puff_zone_trials]
    
    # Create aligned windows
    speed_windows, aligned_time_speed = create_aligned_windows(
        cap_time, speed_val, zone_entry_times, window
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
        analyze_puff_deliveries(puff_zone_trials, trial_log_df, cap_time, cap_val, speed_val,
                               output_folder, window, cap_vmin, cap_vmax)


def analyze_puff_deliveries(puff_zone_trials, trial_log_df, cap_time, cap_val, speed_val,
                            output_folder, window=10, cap_vmin=0, cap_vmax=5000):
    """Analyze data aligned to puff delivery times
    
    Args:
        puff_zone_trials: List of puff zone trials (trial_idx, zone_entry, puff_event)
                         Used to map puff events to their zone entries when available
        trial_log_df: Trial log DataFrame (to get ALL puff events)
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_val: Treadmill speed array
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
        cap_time, speed_val, puff_times, window
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
# ANALYSIS FUNCTIONS: PROBE EVENTS
# ============================================================================

def analyze_probe_events(trial_log_df, capacitive_df, treadmill_interp, output_folder, window=5, cap_vmax=None):
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
    
    # Extract base arrays
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
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
        mask = (cap_time >= pt - window) & (cap_time <= pt + window)
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
    sem_speed_probe = np.nanstd(speed_probe_windows_padded, axis=0) / np.sqrt(
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
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # --- Plot 2: Capacitive Value aligned to probe events ---
    mean_cap_probe = np.nanmean(cap_probe_windows_padded, axis=0)
    sem_cap_probe = np.nanstd(cap_probe_windows_padded, axis=0) / np.sqrt(
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


def match_probe_to_revert_times(trial_log_df, texture_data):
    """Match probe times with texture revert times (approximately 1 second before probe)
    
    Args:
        trial_log_df: Trial log DataFrame
        texture_data: Dictionary with processed texture data
        
    Returns:
        tuple: (probe_revert_array, all_revert_times) or (None, revert_times_array)
    """
    # Collect all texture_revert times from all trials (both stay and go zones)
    all_revert_times = []
    for trial_idx in range(len(trial_log_df)):
        # Collect from stay zone (reward) revert times
        stay_revert_list = safe_literal_eval(trial_log_df.iloc[trial_idx]['stay_texture_revert_time'])
        for revert_time in stay_revert_list:
            if not pd.isna(revert_time) and revert_time != '':
                try:
                    all_revert_times.append(float(revert_time))
                except (ValueError, TypeError):
                    continue
        
        # Collect from go zone (punish) revert times
        go_revert_list = safe_literal_eval(trial_log_df.iloc[trial_idx]['go_texture_revert_time'])
        for revert_time in go_revert_list:
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
                                   capacitive_df, treadmill_interp, output_folder, window=5, cap_vmax=None):
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
    
    # Extract base arrays
    cap_time = capacitive_df['elapsed_time'].values
    cap_val = capacitive_df['capacitive_value'].values
    speed_val = treadmill_interp.values
    
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
        mask = (cap_time >= sim_probe_time - window) & (cap_time <= sim_probe_time + window)
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
    sem_speed_sim = np.nanstd(speed_sim_windows_padded, axis=0) / np.sqrt(
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
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # --- Plot 2: Capacitive Value ---
    mean_cap_sim = np.nanmean(cap_sim_windows_padded, axis=0)
    sem_cap_sim = np.nanstd(cap_sim_windows_padded, axis=0) / np.sqrt(
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
    
    # Step 4: Interpolate treadmill data
    print("Interpolating treadmill data to capacitive timeline...")
    treadmill_interp = interpolate_treadmill_to_capacitive(
        data['treadmill'], data['capacitive']
    )
    
    print("Interpolating treadmill distance to capacitive timeline...")
    treadmill_distance_interp = interpolate_treadmill_distance_to_capacitive(
        data['treadmill'], data['capacitive']
    )
    
    # Step 5: Process pupil data (if available)
    pupil_diameter_interp = None
    if has_pupil_data:
        print("Processing pupil data...")
        pupil_diameter_interp = process_pupil_data(
            data['pupil'], data['frame_log'], data['capacitive']
        )
    
    # Step 6: Plot main timeline
    print("\nGenerating main timeline plot...")
    plot_main_timeline(
        data['capacitive'], treadmill_interp, treadmill_distance_interp, pupil_diameter_interp,
        data['trial_log'], texture_data, has_pupil_data, output_folder
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
            reward_zone_trials, data['trial_log'], data['capacitive'], treadmill_interp,
            pupil_diameter_interp, output_folder
        )
        
        # Create average trace plots for rewards
        print("\nCreating reward average trace plots...")
        plot_average_traces_reward(
            reward_zone_trials, data['trial_log'], data['capacitive'],
            treadmill_interp, pupil_diameter_interp, output_folder, cap_vmax=cap_vmax
        )
    
    # Step 8: Match and analyze puff zones (using capacitive scale from reward analysis)
    print("\nAnalyzing puff zones...")
    puff_zone_trials = match_puff_zones_to_events(
        data['trial_log'], texture_data['punish_texture_change_time_first']
    )
    
    if len(puff_zone_trials) > 0:
        analyze_puff_zones(
            puff_zone_trials, data['trial_log'], data['capacitive'], treadmill_interp,
            pupil_diameter_interp, output_folder, window=10, cap_vmin=cap_vmin, cap_vmax=cap_vmax
        )
        
        # Create average trace plots for puffs
        print("\nCreating puff average trace plots...")
        plot_average_traces_puff(
            puff_zone_trials, data['trial_log'], data['capacitive'],
            treadmill_interp, pupil_diameter_interp, output_folder, cap_vmax=cap_vmax
        )
    
    # Step 9: Analyze probe events (using capacitive scale from reward analysis)
    print("\nAnalyzing probe events...")
    analyze_probe_events(
        data['trial_log'], data['capacitive'], treadmill_interp, output_folder, cap_vmax=cap_vmax
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
            data['capacitive'], treadmill_interp, output_folder, cap_vmax=cap_vmax
        )
    
    # Step 11: Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if hasattr(save_figure, 'figure_count'):
        print(f"Total figures saved: {save_figure.figure_count}")
        print(f"Output directory: {output_folder}")
    print("\nAll plots displayed and saved as SVG files.")


if __name__ == "__main__":
    main()
