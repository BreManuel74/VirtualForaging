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


def interpolate_treadmill_to_capacitive(treadmill_df, capacitive_df):
    """Interpolate treadmill data to match capacitive timeline
    
    Args:
        treadmill_df: Treadmill DataFrame
        capacitive_df: Capacitive DataFrame
        
    Returns:
        pd.Series: Interpolated treadmill speed
    """
    return pd.Series(
        data=np.interp(
            capacitive_df['elapsed_time'],
            treadmill_df['global_time'],
            treadmill_df['speed']
        ),
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
    
    # Rename columns if needed
    if pupil_df.columns[0] != 'frame_number':
        pupil_df.columns = ['frame_number'] + list(pupil_df.columns[1:])
    
    # Extract relevant columns
    required_cols = ['point_3_x', 'point_3_y', 'point_3_likelihood', 
                     'point_7_x', 'point_7_y', 'point_7_likelihood']
    
    for col in required_cols:
        if col not in pupil_df.columns:
            print(f"Warning: Missing column {col} in pupil data")
            return None
    
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
                if time_diff < best_time_diff and time_diff < 2.0:  # Within 2 seconds
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
    
    # Collect all puff zone entries
    all_puff_zones = []
    for trial_idx in range(len(trial_log_df)):
        texture_hist = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_history'])
        texture_times = safe_literal_eval(trial_log_df.iloc[trial_idx]['texture_change_time'])
        
        for i, texture in enumerate(texture_hist):
            if texture == "assets/punish_mean100.jpg" and i < len(texture_times):
                zone_entry_time = texture_times[i]
                if pd.notna(zone_entry_time) and zone_entry_time > 0:
                    all_puff_zones.append((trial_idx, zone_entry_time))
                    break  # Only first puff per zone
    
    print(f"Found {len(all_puff_zones)} puff zone entries")
    
    # Collect all puff events
    puff_events = []
    for trial_idx in range(len(trial_log_df)):
        puff_event = trial_log_df.iloc[trial_idx]['puff_event']
        if pd.notna(puff_event) and puff_event > 0:
            puff_events.append((trial_idx, puff_event))
    
    print(f"Found {len(puff_events)} puff delivery events")
    
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
                if time_diff < best_time_diff and time_diff < 2.0:  # Within 2 seconds
                    best_match = i
                    best_time_diff = time_diff
        
        if best_match is not None:
            zone_trial_idx, zone_entry_time = all_puff_zones[best_match]
            puff_zone_trials.append((zone_trial_idx, zone_entry_time, puff_event_time))
            matched_zones.add(best_match)
    
    # Add unmatched zones
    for i, (zone_trial_idx, zone_entry_time) in enumerate(all_puff_zones):
        if i not in matched_zones:
            puff_zone_trials.append((zone_trial_idx, zone_entry_time, np.nan))
    
    puff_zone_trials.sort(key=lambda x: x[1])
    
    valid_deliveries = sum(1 for _, _, p in puff_zone_trials if pd.notna(p) and p > 0)
    print(f"Successfully matched {valid_deliveries}/{len(puff_zone_trials)} puff zones to deliveries")
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

def plot_main_timeline(capacitive_df, treadmill_interp, pupil_diameter_interp,
                       trial_log_df, texture_data, has_pupil_data, output_folder):
    """Create the main timeline plot with all data streams
    
    Args:
        capacitive_df: Capacitive sensor DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        trial_log_df: Trial log DataFrame
        texture_data: Dictionary with processed texture data
        has_pupil_data: Whether pupil data is available
        output_folder: Directory to save figures
    """
    num_plots = 3 if has_pupil_data and pupil_diameter_interp is not None else 2
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 10 if num_plots == 3 else 8), sharex=True)
    
    # Ensure axs is always a list
    if num_plots == 2:
        axs = [axs[0], axs[1]]
    else:
        axs = [axs[0], axs[1], axs[2]]
    
    # Get event times
    reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna() if 'puff_event' in trial_log_df.columns else pd.Series([])
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna() if 'probe_time' in trial_log_df.columns else pd.Series([])
    
    # Plot capacitive data
    plot_capacitive_timeline(axs[0], capacitive_df, reward_times, puff_times, probe_times, 
                             texture_data, "Capacitive")
    
    # Plot treadmill data
    plot_treadmill_timeline(axs[1], capacitive_df, treadmill_interp, reward_times, 
                           puff_times, probe_times, texture_data, has_pupil_data)
    
    # Plot pupil data if available
    if num_plots == 3 and pupil_diameter_interp is not None:
        plot_pupil_timeline(axs[2], capacitive_df, pupil_diameter_interp, reward_times,
                          puff_times, probe_times, texture_data)
    
    # Set x-axis limits
    xmin = capacitive_df['elapsed_time'].min()
    xmax = capacitive_df['elapsed_time'].max()
    for ax in axs:
        ax.set_xlim([xmin, xmax])
    
    setup_plot_style(axs)
    plt.tight_layout()
    save_figure(fig, f"timeline_{'capacitive_treadmill_pupil' if num_plots == 3 else 'capacitive_and_treadmill'}", 
                output_folder)
    plt.show()


def plot_capacitive_timeline(ax, capacitive_df, reward_times, puff_times, probe_times, 
                             texture_data, label_prefix=""):
    """Plot capacitive sensor data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], 
            label='Capacitive Value')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    ax.set_ylabel('Capacitive Value')
    ax.set_title(f'{label_prefix} Capacitive Sensor Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)


def plot_treadmill_timeline(ax, capacitive_df, treadmill_interp, reward_times, 
                           puff_times, probe_times, texture_data, has_pupil_data):
    """Plot treadmill speed data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], treadmill_interp, 
            label='Treadmill Speed (interpolated)', color='purple')
    
    # Add event markers
    add_event_markers(ax, reward_times, puff_times, probe_times)
    
    # Highlight reward and punish intervals
    add_texture_intervals(ax, texture_data)
    
    ax.set_xlabel('Elapsed Time (s)' if not has_pupil_data else '')
    ax.set_ylabel('Speed')
    ax.set_title('Interpolated Treadmill Speed Over Time with Reward and Puff Events')
    ax.legend(loc='upper right')


def plot_pupil_timeline(ax, capacitive_df, pupil_diameter_interp, reward_times,
                       puff_times, probe_times, texture_data):
    """Plot pupil diameter data with event markers"""
    ax.plot(capacitive_df['elapsed_time'], pupil_diameter_interp,
            label='Pupil Diameter (interpolated)', color='orange')
    
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
        # Show approximately 10 labels for larger datasets
        step = max(1, n_trials // 10)
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
                               treadmill_interp, pupil_diameter_interp, output_folder, window=5):
    """Plot average traces (mean ± SEM) for reward zone and delivery events
    
    Args:
        reward_zone_trials: List of reward zone trial tuples
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
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
    axs[0].set_ylabel('Treadmill Speed (interpolated)')
    axs[0].set_title('Treadmill Speed Aligned to Reward Zone Onset')
    axs[0].legend()
    axs[0].set_xlim(-5, 5)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Capacitive Value aligned to reward events
    n_rewards_event = cap_event_windows_padded.shape[0]
    axs[1].plot(aligned_time_event, mean_event_vals, color='green', label=f'Mean (n={n_rewards_event})')
    axs[1].fill_between(aligned_time_event, mean_event_vals - sem_event_vals, 
                        mean_event_vals + sem_event_vals, color='green', alpha=0.2, label='SEM')
    axs[1].axvline(0, color='red', linestyle='--', label='Reward Event (t=0)')
    axs[1].set_ylabel('Capacitive Value')
    axs[1].set_title('Capacitive Value Aligned to Reward Event')
    axs[1].legend()
    axs[1].set_xlim(-5, 5)
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
                             treadmill_interp, pupil_diameter_interp, output_folder, window=5):
    """Plot average traces (mean ± SEM) for puff zone and delivery events
    
    Args:
        puff_zone_trials: List of puff zone trial tuples
        trial_log_df: Trial log DataFrame
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
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
    axs[0].plot(aligned_time_puff, mean_speed_puff, color='red', linewidth=2, 
               label=f'Mean Speed (n={n_puff_events})')
    axs[0].fill_between(aligned_time_puff, mean_speed_puff - sem_speed_puff, 
                        mean_speed_puff + sem_speed_puff, color='red', alpha=0.2, label='SEM')
    axs[0].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Zone Entry (t=0)')
    axs[0].set_ylabel('Treadmill Speed (interpolated)')
    axs[0].set_title(f'Average Treadmill Speed Aligned to Puff Zone Entry Times (n={n_puff_events})')
    axs[0].legend()
    axs[0].set_xlim(-5, 5)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Plot 2: Capacitive Value aligned to puff events
    if puff_event_capacitive_data is not None:
        axs[1].plot(puff_event_capacitive_data['aligned_time'], puff_event_capacitive_data['mean_values'],
                   color='blue', linewidth=2, label=f'Mean Capacitive (n={puff_event_capacitive_data["n_events"]})')
        axs[1].fill_between(puff_event_capacitive_data['aligned_time'],
                           puff_event_capacitive_data['mean_values'] - puff_event_capacitive_data['sem_values'],
                           puff_event_capacitive_data['mean_values'] + puff_event_capacitive_data['sem_values'],
                           color='blue', alpha=0.2, label='SEM')
        axs[1].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Event (t=0)')
        axs[1].set_ylabel('Capacitive Value')
        axs[1].set_title(f'Average Capacitive Value Aligned to Puff Events (n={puff_event_capacitive_data["n_events"]})')
        axs[1].legend()
        axs[1].set_ylim(bottom=0)
    else:
        axs[1].text(0.5, 0.5, 'No puff event data available\nfor capacitive analysis',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=12)
        axs[1].set_ylabel('Capacitive Value')
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
        axs[2].set_ylabel('Treadmill Speed (interpolated)')
        axs[2].set_title(f'Average Treadmill Speed Aligned to Puff Events (n={puff_event_speed_data["n_events"]})')
        axs[2].legend()
    else:
        axs[2].text(0.5, 0.5, 'No puff event data available\nfor treadmill speed analysis',
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[2].transAxes, fontsize=12)
        axs[2].set_ylabel('Treadmill Speed')
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

def analyze_reward_zones(reward_zone_trials, capacitive_df, treadmill_interp, 
                         pupil_diameter_interp, output_folder, window=5):
    """Analyze data aligned to reward zone entries and deliveries
    
    Args:
        reward_zone_trials: List of (trial_idx, zone_entry, reward_event) tuples
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
    """
    if len(reward_zone_trials) == 0:
        print("No reward zones found. Skipping reward zone analysis.")
        return
    
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
    
    # Plot raster for treadmill speed at zone entry
    if speed_windows is not None:
        plot_raster_heatmap(
            speed_windows, aligned_time_speed, reward_zone_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Reward Zone Entry (n={len(reward_zone_trials)} trials)',
            'Treadmill Speed', 'coolwarm', output_folder, 
            'treadmill_speed_raster_reward_zones',
            vmin=-300,
            vmax=300,
            center_time=0, event_label="Reward Zone Entry",
            show_delivery_markers=True, center_line_color='black'
        )
    
    # Plot raster for capacitive at zone entry
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, reward_zone_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Reward Zone Entry (n={len(reward_zone_trials)} trials)',
            'Capacitive Value (Licking)', 'binary', output_folder,
            'capacitive_raster_reward_zones',
            center_time=0, event_label="Reward Zone Entry",
            show_delivery_markers=True, center_line_color='blue'
        )
    
    # Analyze reward deliveries (where reward_event is valid)
    reward_delivery_trials = [(t, z, r) for t, z, r in reward_zone_trials 
                             if pd.notna(r) and r > 0]
    
    if len(reward_delivery_trials) > 0:
        print(f"Analyzing {len(reward_delivery_trials)} reward deliveries...")
        analyze_reward_deliveries(reward_delivery_trials, cap_time, cap_val, speed_val,
                                 pupil_diameter_interp, output_folder, window)


def analyze_reward_deliveries(reward_delivery_trials, cap_time, cap_val, speed_val,
                              pupil_diameter_interp, output_folder, window=5):
    """Analyze data aligned to reward delivery times
    
    Args:
        reward_delivery_trials: List of trials with valid reward deliveries
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_val: Treadmill speed array
        pupil_diameter_interp: Interpolated pupil diameter
        output_folder: Directory to save figures
        window: Window size in seconds
    """
    # Extract reward delivery times (centered at t=0)
    reward_times = [r for _, _, r in reward_delivery_trials]
    
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
            speed_windows, aligned_time_speed, reward_delivery_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Reward Delivery (n={len(reward_delivery_trials)} trials)',
            'Treadmill Speed', 'coolwarm', output_folder,
            'treadmill_speed_raster_reward_delivery_centered',
            vmin=-300,
            vmax=300,
            center_time=0, event_label="Reward Delivery",
            show_zone_entries=True, zone_entry_color='black', center_line_color='green'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, reward_delivery_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Reward Delivery (n={len(reward_delivery_trials)} trials)',
            'Capacitive Value (Licking)', 'binary', output_folder,
            'capacitive_raster_reward_delivery_centered',
            center_time=0, event_label="Reward Delivery",
            show_zone_entries=True, zone_entry_color='blue', center_line_color='green'
        )


# ============================================================================
# ANALYSIS FUNCTIONS: PUFF ZONES
# ============================================================================

def analyze_puff_zones(puff_zone_trials, capacitive_df, treadmill_interp, 
                       pupil_diameter_interp, output_folder, window=10):
    """Analyze data aligned to puff zone entries and deliveries
    
    Args:
        puff_zone_trials: List of (trial_idx, zone_entry, puff_event) tuples
        capacitive_df: Capacitive DataFrame
        treadmill_interp: Interpolated treadmill speed
        pupil_diameter_interp: Interpolated pupil diameter (or None)
        output_folder: Directory to save figures
        window: Window size in seconds
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
            'Treadmill Speed', 'coolwarm', output_folder,
            'treadmill_speed_raster_puff_zones',
            vmin=-300,
            vmax=300,
            center_time=0, event_label="Puff Zone Entry",
            show_delivery_markers=True, center_line_color='black'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, puff_zone_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Puff Zone Entry (n={len(puff_zone_trials)} trials)',
            'Capacitive Value (Licking)', 'binary', output_folder,
            'capacitive_raster_puff_zones',
            center_time=0, event_label="Puff Zone Entry",
            show_delivery_markers=True, center_line_color='blue'
        )
    
    # Analyze puff deliveries
    puff_delivery_trials = [(t, z, p) for t, z, p in puff_zone_trials 
                           if pd.notna(p) and p > 0]
    
    if len(puff_delivery_trials) > 0:
        print(f"Analyzing {len(puff_delivery_trials)} puff deliveries...")
        analyze_puff_deliveries(puff_delivery_trials, cap_time, cap_val, speed_val,
                               output_folder, window)


def analyze_puff_deliveries(puff_delivery_trials, cap_time, cap_val, speed_val,
                            output_folder, window=10):
    """Analyze data aligned to puff delivery times
    
    Args:
        puff_delivery_trials: List of trials with valid puff deliveries
        cap_time: Capacitive time array
        cap_val: Capacitive value array
        speed_val: Treadmill speed array
        output_folder: Directory to save figures
        window: Window size in seconds
    """
    # Extract puff delivery times
    puff_times = [p for _, _, p in puff_delivery_trials]
    
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
            speed_windows, aligned_time_speed, puff_delivery_trials,
            f'Treadmill Speed Raster: Individual Trials Aligned to Puff Delivery (n={len(puff_delivery_trials)} trials)',
            'Treadmill Speed', 'coolwarm', output_folder,
            'treadmill_speed_raster_puff_delivery_centered',
            vmin=-400,
            vmax=400,
            center_time=0, event_label="Puff Delivery",
            show_zone_entries=True, zone_entry_color='black', center_line_color='green'
        )
    
    if cap_windows is not None:
        plot_raster_heatmap(
            cap_windows, aligned_time_cap, puff_delivery_trials,
            f'Capacitive (Licking) Raster: Individual Trials Aligned to Puff Delivery (n={len(puff_delivery_trials)} trials)',
            'Capacitive Value (Licking)', 'binary', output_folder,
            'capacitive_raster_puff_delivery_centered',
            center_time=0, event_label="Puff Delivery",
            show_zone_entries=True, zone_entry_color='blue', center_line_color='green'
        )


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
        data['capacitive'], treadmill_interp, pupil_diameter_interp,
        data['trial_log'], texture_data, has_pupil_data, output_folder
    )
    
    # Step 7: Match and analyze reward zones
    print("\nAnalyzing reward zones...")
    reward_zone_trials = match_reward_zones_to_events(
        data['trial_log'], texture_data['reward_texture_change_time']
    )
    
    if len(reward_zone_trials) > 0:
        analyze_reward_zones(
            reward_zone_trials, data['capacitive'], treadmill_interp,
            pupil_diameter_interp, output_folder
        )
        
        # Create average trace plots for rewards
        print("\nCreating reward average trace plots...")
        plot_average_traces_reward(
            reward_zone_trials, data['trial_log'], data['capacitive'],
            treadmill_interp, pupil_diameter_interp, output_folder
        )
    
    # Step 8: Match and analyze puff zones
    print("\nAnalyzing puff zones...")
    puff_zone_trials = match_puff_zones_to_events(
        data['trial_log'], texture_data['punish_texture_change_time_first']
    )
    
    if len(puff_zone_trials) > 0:
        analyze_puff_zones(
            puff_zone_trials, data['capacitive'], treadmill_interp,
            pupil_diameter_interp, output_folder
        )
        
        # Create average trace plots for puffs
        print("\nCreating puff average trace plots...")
        plot_average_traces_puff(
            puff_zone_trials, data['trial_log'], data['capacitive'],
            treadmill_interp, pupil_diameter_interp, output_folder
        )
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if hasattr(save_figure, 'figure_count'):
        print(f"Total figures saved: {save_figure.figure_count}")
        print(f"Output directory: {output_folder}")
    print("\nAll plots displayed and saved as SVG files.")


if __name__ == "__main__":
    main()
