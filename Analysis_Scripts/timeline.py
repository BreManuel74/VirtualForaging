import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
from tkinter import filedialog
import tkinter as tk

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Helper function to save figures as SVG files
def save_figure(fig, name):
    """Save a figure as an SVG file in the output folder
    
    Args:
        fig: The matplotlib figure to save
        name: Base name for the file (without extension)
    """
    global figure_count, output_folder
    if not hasattr(save_figure, 'figure_count'):
        save_figure.figure_count = 1
    
    filename = f"{name}_{save_figure.figure_count}.svg"
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, format="svg", bbox_inches="tight")
    print(f"Saved figure: {filename}")
    save_figure.figure_count += 1
    return filepath

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the main MousePortal directory
initial_dir = os.path.dirname(script_dir)

# Browse for the folder containing your data files
folder_path = filedialog.askdirectory(
    title="Select folder containing behavioral data files",
    initialdir=initial_dir
)

if not folder_path:
    print("No folder selected. Exiting...")
    exit()

# Find files containing the required keywords
trial_log_files = [f for f in os.listdir(folder_path) if 'trial_log.csv' in f]
treadmill_files = [f for f in os.listdir(folder_path) if 'treadmill.csv' in f]
capacitive_files = [f for f in os.listdir(folder_path) if 'capacitive.csv' in f]
pupil_files = [f for f in os.listdir(folder_path) if 'exposure.csv' in f]

# Check if all three required types of files are present
missing_types = []
if not trial_log_files:
    missing_types.append("trial_log.csv")
if not treadmill_files:
    missing_types.append("treadmill.csv")
if not capacitive_files:
    missing_types.append("capacitive.csv")

if missing_types:
    print(f"Warning: Missing required file types in selected folder: {missing_types}")
    print("Please ensure all three required file types are present:")
    print("  - A file containing 'trial_log.csv'")
    print("  - A file containing 'treadmill.csv'")
    print("  - A file containing 'capacitive.csv'")
    exit()

# Check for optional pupil file
has_pupil_data = len(pupil_files) > 0
if has_pupil_data:
    print(f"Pupil data file found: {pupil_files[0]}")
else:
    print("No pupil data file found (optional) - skipping pupil analyses")

# Use the first file found for each type
trial_log_path = os.path.join(folder_path, trial_log_files[0])
treadmill_path = os.path.join(folder_path, treadmill_files[0])
capacitive_path = os.path.join(folder_path, capacitive_files[0])

print(f"Loading files:")
print(f"  - {os.path.basename(trial_log_path)}")
print(f"  - {os.path.basename(treadmill_path)}")
print(f"  - {os.path.basename(capacitive_path)}")

# Conditionally set up pupil file path
if has_pupil_data:
    pupil_path = os.path.join(folder_path, pupil_files[0])
    #print(f"  - {os.path.basename(pupil_path)} (pupil data)")
else:
    pupil_path = None

# Create an output folder for SVG files
output_folder = os.path.join(folder_path, "svg_plots")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory for SVG files: {output_folder}")
else:
    print(f"Using existing directory for SVG files: {output_folder}")

# Read the CSV files into pandas DataFrames
trial_log_df = pd.read_csv(trial_log_path, engine='python')
treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')
capacitive_df = pd.read_csv(capacitive_path, comment='/', engine='python')

# Conditionally load pupil data
if has_pupil_data:
    # Skip first 3 rows (scorer, bodyparts, coords) as they contain metadata
    pupil_df = pd.read_csv(pupil_path, comment='/', engine='python', skiprows=3)
    #print(f"Pupil data loaded successfully: {pupil_df.shape[0]} rows, {pupil_df.shape[1]} columns")
    #print(f"Pupil data columns: {list(pupil_df.columns[:6])}...")  # Show first 6 columns
else:
    pupil_df = None

# Safe literal eval function
def safe_literal_eval(val):
    try:
        if isinstance(val, list):
            return val
        if pd.isna(val) or val == '':
            return []
        # If it's a number, wrap in a list
        if isinstance(val, (int, float)):
            return [val]
        # If it's a string that is not a list, try to convert to float
        if isinstance(val, str) and not (val.strip().startswith("[") and val.strip().endswith("]")):
            try:
                return [float(val)]
            except Exception:
                return [val]
        return ast.literal_eval(val)
    except Exception:
        return []

# Remove the is_list_like filter and just use safe_literal_eval
texture_history = trial_log_df['texture_history'].apply(safe_literal_eval)
texture_change_time = trial_log_df['texture_change_time'].apply(safe_literal_eval)
revert_time = trial_log_df['texture_revert'].apply(safe_literal_eval)

max_len = max(
    texture_history.apply(len).max(),
    texture_change_time.apply(len).max(),
    revert_time.apply(len).max()
)

def pad_list(lst, length):
    return lst + [np.nan] * (length - len(lst))

texture_history_padded = np.array(texture_history.apply(lambda x: pad_list(x, max_len)).tolist())
texture_change_time_padded = np.array(texture_change_time.apply(lambda x: pad_list(x, max_len)).tolist())
revert_time_padded = np.array(revert_time.apply(lambda x: pad_list(x, max_len)).tolist())

combined_array = np.stack(
    [texture_history_padded, texture_change_time_padded, revert_time_padded],
    axis=1
)

# combined_array now has shape (num_rows, 3, max_len)
# combined_array[:, 0, :] = texture_history
# combined_array[:, 1, :] = texture_change_time
# combined_array[:, 2, :] = revert_time

# Create boolean masks for each asset type
is_punish = texture_history_padded[:, 0] == "assets/punish_mean100.jpg"
is_reward = texture_history_padded[:, 0] == "assets/reward_mean100.jpg"

# Select rows for each type
punish_array = combined_array[is_punish]
reward_array = combined_array[is_reward]

# Now punish_array and reward_array contain only rows starting with the respective asset
# For punish:
punish_texture_change_time = punish_array[:, 1, :]
punish_revert_time = punish_array[:, 2, :]

# Create versions that only use the first puff per zone (for calculations)
punish_texture_change_time_first = punish_texture_change_time[:, 0]
punish_revert_time_first = punish_revert_time[:, 0]

# And for reward:
reward_texture_change_time = reward_array[:, 1, :]
reward_revert_time = reward_array[:, 2, :]

# Interpolate treadmill distance to match capacitive elapsed_time
treadmill_interp = pd.Series(
    data=np.interp(
        capacitive_df['elapsed_time'],
        treadmill_df['global_time'],
        treadmill_df['speed']
    ),
    index=capacitive_df['elapsed_time']
)

# Plot both on the same graph (capacitive only, with reward and puff events)
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- Plot 1: Capacitive Value ---
axs[0].plot(capacitive_df['elapsed_time'], capacitive_df['capacitive_value'], label='Capacitive Value')

# Reward events
reward_times = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
for i, rt in enumerate(reward_times):
    axs[0].axvline(x=rt, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Reward Event' if i == 0 else "")

# Puff events
if 'puff_event' in trial_log_df.columns:
    puff_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
    for i, pt in enumerate(puff_times):
        axs[0].axvline(x=pt, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Puff Event' if i == 0 else "")

# Probe events
if 'probe_time' in trial_log_df.columns:
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    for i, pt in enumerate(probe_times):
        axs[0].axvline(x=pt, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Probe Event' if i == 0 else "")

# Highlight reward intervals
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[0].axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals - only using first puff per zone
for trial_idx in range(punish_texture_change_time_first.shape[0]):
    try:
        start = float(punish_texture_change_time_first[trial_idx])
        end = float(punish_revert_time_first[trial_idx])
        if not np.isnan(start) and not np.isnan(end):
            axs[0].axvspan(start, end, color='red', alpha=0.15)
    except (ValueError, TypeError):
        continue

axs[0].set_ylabel('Capacitive Value')
axs[0].set_title('Capacitive Sensor Over Time with Reward and Puff Events')
axs[0].legend(loc='upper right')
axs[0].set_ylim(bottom=0)  # Set the bottom y-axis limit to 0

# Remove top and right borders for both subplots
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Plot 2: Treadmill Speed ---
axs[1].plot(
    capacitive_df['elapsed_time'],
    treadmill_interp,
    label='Treadmill Speed (interpolated)',
    color='purple'  # Set treadmill speed line to purple
)

# Reward events
for i, rt in enumerate(reward_times):
    axs[1].axvline(x=rt, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Reward Event' if i == 0 else "")

# Puff events
if 'puff_event' in trial_log_df.columns:
    for i, pt in enumerate(puff_times):
        axs[1].axvline(x=pt, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Puff Event' if i == 0 else "")

# Probe events
if 'probe_time' in trial_log_df.columns:
    for i, pt in enumerate(probe_times):
        axs[1].axvline(x=pt, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Probe Event' if i == 0 else "")

# Highlight reward intervals
for trial_idx in range(reward_texture_change_time.shape[0]):
    for seg_idx in range(reward_texture_change_time.shape[1]):
        try:
            start = float(reward_texture_change_time[trial_idx, seg_idx])
            end = float(reward_revert_time[trial_idx, seg_idx])
            if not np.isnan(start) and not np.isnan(end):
                axs[1].axvspan(start, end, color='green', alpha=0.15)
        except (ValueError, TypeError):
            continue

# Highlight punish intervals - only using first puff per zone
for trial_idx in range(punish_texture_change_time_first.shape[0]):
    try:
        start = float(punish_texture_change_time_first[trial_idx])
        end = float(punish_revert_time_first[trial_idx])
        if not np.isnan(start) and not np.isnan(end):
            axs[1].axvspan(start, end, color='red', alpha=0.15)
    except (ValueError, TypeError):
        continue

axs[1].set_xlabel('Elapsed Time (s)')
axs[1].set_ylabel('Speed')
axs[1].set_title('Interpolated Treadmill Speed Over Time with Reward and Puff Events')
axs[1].legend(loc='upper right')

# Set x-axis limits to the data range
xmin = capacitive_df['elapsed_time'].min()
xmax = capacitive_df['elapsed_time'].max()
for ax in axs:
    ax.set_xlim([xmin, xmax])

plt.tight_layout()
save_figure(fig, "timeline_capacitive_and_treadmill")
plt.show()

window = 5  # seconds before and after
reward_times_flat = reward_texture_change_time.flatten()
reward_times_flat = pd.to_numeric(reward_times_flat, errors='coerce')
reward_times_flat = reward_times_flat[~np.isnan(reward_times_flat)]

cap_time = capacitive_df['elapsed_time'].values
cap_val = capacitive_df['capacitive_value'].values

cap_windows = []
for rt in reward_times_flat:
    mask = (cap_time >= rt - window) & (cap_time <= rt + window)
    cap_segment = cap_val[mask]
    cap_windows.append(cap_segment)

# Pad all segments to the same length (max found)
max_len = max(len(seg) for seg in cap_windows)
cap_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_len - len(seg)), constant_values=np.nan)
    for seg in cap_windows
])

# cap_windows_padded is your 2D array: shape (num_reward_events, num_timepoints)
# Each row: capacitive values from 5s before to 5s after each reward_texture_change_time

# Example: print shape
#print("Shape of cap_windows_padded:", cap_windows_padded.shape)

# Create a common time axis centered at 0
dt = np.median(np.diff(cap_time))  # Estimate sampling interval
window_len = cap_windows_padded.shape[1]
aligned_time = np.linspace(-window, window, window_len)

# plt.figure(figsize=(10, 6))

n_rewards = cap_windows_padded.shape[0]  # Number of reward events

# Plot mean and SEM
# mean_vals = np.nanmean(cap_windows_padded, axis=0)
# sem_vals = np.nanstd(cap_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(cap_windows_padded), axis=0))
# plt.plot(aligned_time, mean_vals, color='blue', label=f'Mean (n={n_rewards})')
# plt.fill_between(aligned_time, mean_vals - sem_vals, mean_vals + sem_vals, color='blue', alpha=0.2, label='SEM')

# plt.axvline(0, color='red', linestyle='--', label='Reward Onset (t=0)')
# plt.xlabel('Time from Reward Zone Onset (s)')
# plt.ylabel('Capacitive Value')
# plt.title('Capacitive Value Aligned to Reward Zone Onset')
# plt.legend()
# plt.xticks(np.arange(-5, 6, 1))  # Set x-axis ticks from -5 to 5 with step 1
# plt.xlim(-5, 5)   
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.tick_params(axis='both', direction='out')
# plt.tight_layout()
# #plt.show()

# --- Interpolated Treadmill Speed aligned to reward_times_flat ---

# Get interpolated speed as numpy array
speed_val = treadmill_interp.values

speed_windows = []
for rt in reward_times_flat:
    mask = (cap_time >= rt - window) & (cap_time <= rt + window)
    speed_segment = speed_val[mask]
    speed_windows.append(speed_segment)

# Pad all segments to the same length (max found)
max_speed_len = max(len(seg) for seg in speed_windows)
speed_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_speed_len - len(seg)), constant_values=np.nan)
    for seg in speed_windows
])

# Create a common time axis centered at 0 for speed
aligned_time_speed = np.linspace(-window, window, max_speed_len)

n_rewards_speed = speed_windows_padded.shape[0]

# Plot mean and SEM for speed
mean_speed = np.nanmean(speed_windows_padded, axis=0)
sem_speed = np.nanstd(speed_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_windows_padded), axis=0))

# --- Capacitive Value aligned to reward_event_times_flat ---

reward_event_times_flat = pd.to_numeric(trial_log_df['reward_event'], errors='coerce').dropna()
reward_event_times_flat = reward_event_times_flat[~np.isnan(reward_event_times_flat)]

window_event = 5  # seconds before and after for capacitive analysis

# Ensure reward_event_times_flat is a numpy array of floats
reward_event_times_flat = np.array(reward_event_times_flat, dtype=float)

cap_event_windows = []
for rt in reward_event_times_flat:
    mask = (cap_time >= rt - window_event) & (cap_time <= rt + window_event)
    cap_segment = cap_val[mask]
    cap_event_windows.append(cap_segment)

#print(len(cap_event_windows))

# Pad all segments to the same length (max found)
max_event_len = max(len(seg) for seg in cap_event_windows)
cap_event_windows_padded = np.array([
    np.pad(seg.astype(float), (0, max_event_len - len(seg)), constant_values=np.nan)
    for seg in cap_event_windows
])

# Apply moving median filter (15-point window) to each row before averaging
def apply_moving_median_15point(data):
    """Apply 15-point moving median filter to 2D array along axis 1"""
    filtered = np.copy(data)
    half_window = 7  # 7 points before + current + 7 points after = 15 total
    
    for i in range(data.shape[0]):
        row = data[i, :]
        for j in range(half_window, len(row) - half_window):
            window = row[j-half_window:j+half_window+1]  # +1 to include current point
            if not np.isnan(window).any():
                filtered[i, j] = np.median(window)
    return filtered

cap_event_windows_filtered = apply_moving_median_15point(cap_event_windows_padded)

# Create a common time axis centered at 0
aligned_time_event = np.linspace(-window_event, window_event, max_event_len)

n_rewards_event = cap_event_windows_filtered.shape[0]

# Plot mean and SEM using filtered data
mean_event_vals = np.nanmean(cap_event_windows_filtered, axis=0)
sem_event_vals = np.nanstd(cap_event_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_event_windows_filtered), axis=0))

# --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to reward_event_times_flat ---

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Plot 1: Treadmill Speed aligned to reward_times_flat ---
axs[0].plot(aligned_time_speed, mean_speed, color='purple', label=f'Mean Speed (n={n_rewards_speed})')
axs[0].fill_between(aligned_time_speed, mean_speed - sem_speed, mean_speed + sem_speed, color='purple', alpha=0.2, label='SEM')
axs[0].axvline(0, color='red', linestyle='--', label='Reward Onset (t=0)')
axs[0].set_ylabel('Treadmill Speed (interpolated)')
axs[0].set_title('Treadmill Speed Aligned to Reward Zone Onset')
axs[0].legend()
axs[0].set_xlim(-5, 5)
#axs[0].set_ylim(bottom=0)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].tick_params(axis='both', direction='out')

# --- Plot 2: Capacitive Value aligned to reward_event_times_flat (5s window) ---
axs[1].plot(aligned_time_event, mean_event_vals, color='green', label=f'Mean (n={n_rewards_event})')
axs[1].fill_between(aligned_time_event, mean_event_vals - sem_event_vals, mean_event_vals + sem_event_vals, color='green', alpha=0.2, label='SEM')
axs[1].axvline(0, color='red', linestyle='--', label='Reward Event (t=0)')
axs[1].set_xlabel('Time from Reward Event (s)')
axs[1].set_ylabel('Capacitive Value')
axs[1].set_title('Capacitive Value Aligned to Reward Event')
axs[1].legend()
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(bottom=0)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].tick_params(axis='both', direction='out')
axs[1].set_xticks(np.arange(-5, 6, 1))

plt.tight_layout()
save_figure(fig, "reward_event_analysis")


# --- Probe Event Analysis: Treadmill Speed and Capacitive Value aligned to probe events ---

# Check if probe events exist
if 'probe_time' in trial_log_df.columns:
    probe_event_times_flat = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna()
    probe_event_times_flat = probe_event_times_flat[~np.isnan(probe_event_times_flat)]
    
    if len(probe_event_times_flat) > 0:
        window = 5  # seconds before and after
        
        # Ensure probe_event_times_flat is a numpy array of floats
        probe_event_times_flat = np.array(probe_event_times_flat, dtype=float)
        
        # --- Capacitive Value aligned to probe events ---
        cap_probe_windows = []
        for pt in probe_event_times_flat:
            mask = (cap_time >= pt - window) & (cap_time <= pt + window)
            cap_segment = cap_val[mask]
            cap_probe_windows.append(cap_segment)
        
        # Pad all segments to the same length (max found)
        max_probe_len = max(len(seg) for seg in cap_probe_windows) if cap_probe_windows else 0
        if max_probe_len > 0:
            cap_probe_windows_padded = np.array([
                np.pad(seg.astype(float), (0, max_probe_len - len(seg)), constant_values=np.nan)
                for seg in cap_probe_windows
            ])
            
            # Apply moving median filter (21-point window) to probe capacitive data
            cap_probe_windows_filtered = apply_moving_median_15point(cap_probe_windows_padded)
            
            # --- Treadmill Speed aligned to probe events ---
            speed_probe_windows = []
            for pt in probe_event_times_flat:
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
            sem_speed_probe = np.nanstd(speed_probe_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_probe_windows_padded), axis=0))
            axs[0].plot(aligned_time_probe, mean_speed_probe, color='purple', label=f'Mean Speed (n={n_probes})')
            axs[0].fill_between(aligned_time_probe, mean_speed_probe - sem_speed_probe, mean_speed_probe + sem_speed_probe, color='purple', alpha=0.2, label='SEM')
            axs[0].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)')
            axs[0].set_ylabel('Treadmill Speed (interpolated)')
            axs[0].set_title('Treadmill Speed Aligned to Probe Events')
            axs[0].legend()
            axs[0].set_xlim(-5, 5)
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[0].tick_params(axis='both', direction='out')
            
            # --- Plot 2: Capacitive Value aligned to probe events ---
            mean_cap_probe = np.nanmean(cap_probe_windows_filtered, axis=0)
            sem_cap_probe = np.nanstd(cap_probe_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_probe_windows_filtered), axis=0))
            axs[1].plot(aligned_time_probe, mean_cap_probe, color='green', label=f'Mean (n={n_probes})')
            axs[1].fill_between(aligned_time_probe, mean_cap_probe - sem_cap_probe, mean_cap_probe + sem_cap_probe, color='green', alpha=0.2, label='SEM')
            axs[1].axvline(0, color='black', linestyle='--', label='Probe Event (t=0)')
            axs[1].set_xlabel('Time from Probe Event (s)')
            axs[1].set_ylabel('Capacitive Value')
            axs[1].set_title('Capacitive Value Aligned to Probe Events')
            axs[1].legend()
            axs[1].set_xlim(-5, 5)
            axs[1].set_ylim(bottom=0)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            axs[1].tick_params(axis='both', direction='out')
            axs[1].set_xticks(np.arange(-5, 6, 1))
            
            plt.tight_layout()
            save_figure(fig, "probe_event_analysis")
            plt.show()
        else:
            print("No valid probe event data found for alignment analysis.")
    else:
        print("No probe events found in the data.")
else:
    print("No 'probe_time' column found in trial_log_df.")

plt.tight_layout()

# --- Prepare data for heatmap with 2-second window ---
window_heatmap = 5  # seconds before and after for heatmap

cap_event_windows_heatmap = []
for rt in reward_event_times_flat:
    mask = (cap_time >= rt - window_heatmap) & (cap_time <= rt + window_heatmap)
    cap_segment = cap_val[mask]
    cap_event_windows_heatmap.append(cap_segment)

# Pad all segments to the same length (max found)
max_event_len_heatmap = max(len(seg) for seg in cap_event_windows_heatmap)
cap_event_windows_padded_heatmap = np.array([
    np.pad(seg.astype(float), (0, max_event_len_heatmap - len(seg)), constant_values=np.nan)
    for seg in cap_event_windows_heatmap
])

# --- Heatmap of cap_event_windows (2s window) ---

plt.figure(figsize=(12, 8))

# Create the heatmap using 2-second window data
im = plt.imshow(cap_event_windows_padded_heatmap, aspect='auto', cmap='gray_r', interpolation='nearest')

# Set up the time axis labels for 2-second window
n_timepoints = cap_event_windows_padded_heatmap.shape[1]
#print("Number of timepoints:", n_timepoints)
time_labels = np.linspace(-window_heatmap, window_heatmap, n_timepoints)
tick_indices = np.linspace(0, n_timepoints-1, 11, dtype=int)  # 11 ticks 
tick_labels = [f'{time_labels[i]:.1f}' for i in tick_indices]

plt.xticks(tick_indices, tick_labels)
plt.xlabel('Time from Reward Event (s)')
plt.ylabel('Reward Event #')
plt.title(f'Heatmap: Capacitive Value Across All Reward Events (5s window, n={n_rewards_event})')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Capacitive Value')

# Add vertical line at t=0
plt.axvline(x=n_timepoints//2, color='red', linestyle='--', alpha=0.8, linewidth=2)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
current_fig = plt.gcf()
save_figure(current_fig, "reward_events_heatmap")
plt.show()

# --- Match probe_time with texture_revert_time (approximately 1 second before) ---

# Collect all texture_revert times from all trials (needed for both probe matching and unpaired analysis)
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

# Check if probe_time column exists and has data
if 'probe_time' not in trial_log_df.columns:
    print("Warning: No 'probe_time' column found in trial_log_df. Skipping probe-revert matching analysis.")
    probe_revert_array = None
else:
    # Extract probe times
    probe_times = pd.to_numeric(trial_log_df['probe_time'], errors='coerce').dropna().values
    
    if len(probe_times) == 0:
        print("Warning: No valid probe times found in the data. Skipping probe-revert matching analysis.")
        probe_revert_array = None
    else:
        if len(all_revert_times) == 0:
            print("Warning: No valid texture revert times found in the data. Cannot match with probe times.")
            probe_revert_array = None
        else:
            # Match each probe_time with the closest texture_revert_time that occurs ~1 second before
            probe_revert_matches = []
            tolerance = 0.5  # Allow ±0.5 seconds around the 1-second target
            
            for probe_time in probe_times:
                # Find revert times that occur before the probe time
                candidate_reverts = all_revert_times[all_revert_times < probe_time]
                
                if len(candidate_reverts) > 0:
                    # Calculate time differences (probe_time - revert_time)
                    time_diffs = probe_time - candidate_reverts
                    
                    # Find revert times that are approximately 1 second before (within tolerance)
                    target_diff = 1.0  # 1 second
                    valid_matches = np.abs(time_diffs - target_diff) <= tolerance
                    
                    if np.any(valid_matches):
                        # Get the closest match to exactly 1 second before
                        valid_diffs = time_diffs[valid_matches]
                        valid_reverts = candidate_reverts[valid_matches]
                        closest_idx = np.argmin(np.abs(valid_diffs - target_diff))
                        matched_revert = valid_reverts[closest_idx]
                        actual_diff = time_diffs[valid_matches][closest_idx]
                    else:
                        # If no matches within tolerance, find the closest revert time before probe
                        closest_idx = np.argmin(time_diffs)
                        matched_revert = candidate_reverts[closest_idx]
                        actual_diff = time_diffs[closest_idx]
                else:
                    # No revert times before this probe
                    matched_revert = np.nan
                    actual_diff = np.nan
                
                probe_revert_matches.append([probe_time, matched_revert, actual_diff])
            
            # Convert to 2D numpy array
            probe_revert_array = np.array(probe_revert_matches)
            
            # Print results only if we have matches
            successful_matches = np.sum(~np.isnan(probe_revert_array[:, 1]))
            
            if successful_matches == 0:
                print("Warning: No successful matches found between probe times and revert times.")
                print("This could indicate:")
                print("- Probe times occur before any texture revert events")
                print("- Time tolerance (±0.5s around 1s) is too strict")
                print("- Data timing issues or missing events")

# --- Find unpaired revert times (not matched with any probe event) ---

if probe_revert_array is not None and len(probe_revert_array) > 0:
    # Get all matched revert times (excluding NaN values)
    matched_revert_times = probe_revert_array[:, 1]  # Column 1 contains matched revert times
    matched_revert_times = matched_revert_times[~np.isnan(matched_revert_times)]
    
    # Find unpaired revert times by comparing with all revert times
    unpaired_revert_times = []
    for revert_time in all_revert_times:
        # Check if this revert time was matched with any probe
        # Use a small tolerance for floating point comparison
        tolerance_match = 1e-6
        is_matched = np.any(np.abs(matched_revert_times - revert_time) < tolerance_match)
        
        if not is_matched:
            unpaired_revert_times.append(revert_time)
    unpaired_revert_times = np.array(unpaired_revert_times)
    
    if len(unpaired_revert_times) == 0:
        print("All revert times have been matched with probe events.")
        unpaired_revert_times = np.array([])

elif len(all_revert_times) > 0:
    # If no probe analysis was performed but we have revert times
    unpaired_revert_times = all_revert_times.copy()
else:
    print("No revert times found in the data.")
    unpaired_revert_times = np.array([])

# --- Analysis of unpaired revert times: Simulate probe events 1 second after revert ---

# Only perform simulated probe analysis if there were actual probe events in the data
if 'probe_time' in trial_log_df.columns and len(unpaired_revert_times) > 0 and probe_revert_array is not None:
    # Create simulated probe times (1 second after each unpaired revert time)
    simulated_probe_times = unpaired_revert_times + 1.0
    
    window_sim = 5  # seconds before and after simulated probe
    
    # --- Capacitive Value aligned to simulated probe events ---
    cap_sim_windows = []
    for sim_probe_time in simulated_probe_times:
        mask = (cap_time >= sim_probe_time - window_sim) & (cap_time <= sim_probe_time + window_sim)
        cap_segment = cap_val[mask]
        cap_sim_windows.append(cap_segment)
    
    # Check if we have any valid windows
    if cap_sim_windows and max(len(seg) for seg in cap_sim_windows) > 0:
        # Pad all segments to the same length
        max_sim_len = max(len(seg) for seg in cap_sim_windows)
        cap_sim_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
            for seg in cap_sim_windows
        ])
        
        # Apply moving median filter (21-point window) to simulated probe capacitive data
        cap_sim_windows_filtered = apply_moving_median_15point(cap_sim_windows_padded)
        
        # --- Treadmill Speed aligned to simulated probe events ---
        speed_sim_windows = []
        for sim_probe_time in simulated_probe_times:
            mask = (cap_time >= sim_probe_time - window_sim) & (cap_time <= sim_probe_time + window_sim)
            speed_segment = speed_val[mask]
            speed_sim_windows.append(speed_segment)
        
        # Pad speed segments to the same length
        speed_sim_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_sim_len - len(seg)), constant_values=np.nan)
            for seg in speed_sim_windows
        ])
        
        # Create a common time axis centered at 0 (simulated probe at t=0)
        aligned_time_sim = np.linspace(-window_sim, window_sim, max_sim_len)
        
        # --- Combined Subplots: Treadmill Speed and Capacitive Value aligned to simulated probe events ---
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        n_sim_probes = cap_sim_windows_padded.shape[0]
        
        # --- Plot 1: Treadmill Speed aligned to simulated probe events ---
        mean_speed_sim = np.nanmean(speed_sim_windows_padded, axis=0)
        sem_speed_sim = np.nanstd(speed_sim_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_sim_windows_padded), axis=0))
        axs[0].plot(aligned_time_sim, mean_speed_sim, color='purple', label=f'Mean Speed (n={n_sim_probes})')
        axs[0].fill_between(aligned_time_sim, mean_speed_sim - sem_speed_sim, mean_speed_sim + sem_speed_sim, color='purple', alpha=0.2, label='SEM')
        axs[0].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)')
        #axs[0].axvline(-1, color='red', linestyle=':', alpha=0.7, label='Revert Time (t=-1s)')
        axs[0].set_ylabel('Treadmill Speed (interpolated)')
        axs[0].set_title('Treadmill Speed Aligned to Simulated Probe Events (1s after unpaired reverts)')
        axs[0].legend()
        axs[0].set_xlim(-5, 5)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', direction='out')
        
        # --- Plot 2: Capacitive Value aligned to simulated probe events ---
        mean_cap_sim = np.nanmean(cap_sim_windows_filtered, axis=0)
        sem_cap_sim = np.nanstd(cap_sim_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_sim_windows_filtered), axis=0))
        axs[1].plot(aligned_time_sim, mean_cap_sim, color='green', label=f'Mean (n={n_sim_probes})')
        axs[1].fill_between(aligned_time_sim, mean_cap_sim - sem_cap_sim, mean_cap_sim + sem_cap_sim, color='green', alpha=0.2, label='SEM')
        axs[1].axvline(0, color='black', linestyle='--', label='Simulated Probe (t=0)')
        #axs[1].axvline(-1, color='red', linestyle=':', alpha=0.7, label='Revert Time (t=-1s)')
        axs[1].set_xlabel('Time from Simulated Probe Event (s)')
        axs[1].set_ylabel('Capacitive Value')
        axs[1].set_title('Capacitive Value Aligned to Simulated Probe Events (1s after unpaired reverts)')
        axs[1].legend()
        axs[1].set_xlim(-5, 5)
        axs[1].set_ylim(bottom=0)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].tick_params(axis='both', direction='out')
        axs[1].set_xticks(np.arange(-5, 6, 1))
        
        plt.tight_layout()
        save_figure(fig, "simulated_probe_events")
        plt.show()
        
    else:
        print("Warning: No valid data windows found for simulated probe analysis.")
        print("This could be due to unpaired revert times being too close to data boundaries.")

elif len(unpaired_revert_times) > 0:
    # If no probe events exist but we have unpaired revert times, skip simulated analysis
    pass
else:
    # No unpaired revert times available
    pass

#plt.show()

# --- Analysis of Treadmill Speed aligned to Puff Zone Entry Times ---

# Extract ONLY FIRST punish_texture_change_times (puff zone entry times) from all trials
# Use first puff per zone for all calculations
puff_zone_times_flat = punish_texture_change_time_first
puff_zone_times_flat = pd.to_numeric(puff_zone_times_flat, errors='coerce')
puff_zone_times_flat = puff_zone_times_flat[~np.isnan(puff_zone_times_flat)]

if len(puff_zone_times_flat) > 0:
    window_puff = 5  # seconds before and after puff zone entry
    
    # Get interpolated speed as numpy array (already calculated above)
    speed_val = treadmill_interp.values
    cap_time = capacitive_df['elapsed_time'].values
    
    # Extract speed windows around each puff zone entry time
    speed_puff_windows = []
    for puff_time in puff_zone_times_flat:
        mask = (cap_time >= puff_time - window_puff) & (cap_time <= puff_time + window_puff)
        speed_segment = speed_val[mask]
        speed_puff_windows.append(speed_segment)
    
    # Pad all segments to the same length (max found)
    if speed_puff_windows and max(len(seg) for seg in speed_puff_windows) > 0:
        max_puff_len = max(len(seg) for seg in speed_puff_windows)
        speed_puff_windows_padded = np.array([
            np.pad(seg.astype(float), (0, max_puff_len - len(seg)), constant_values=np.nan)
            for seg in speed_puff_windows
        ])
        
        # Create a common time axis centered at 0 (puff zone entry at t=0)
        aligned_time_puff = np.linspace(-window_puff, window_puff, max_puff_len)
        
        # Calculate mean and SEM
        n_puff_events = speed_puff_windows_padded.shape[0]
        mean_speed_puff = np.nanmean(speed_puff_windows_padded, axis=0)
        sem_speed_puff = np.nanstd(speed_puff_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_puff_windows_padded), axis=0))
        
        # --- Also analyze capacitive values aligned to puff events ---
        
        # Check if puff_event column exists
        puff_event_capacitive_data = None
        if 'puff_event' in trial_log_df.columns:
            puff_event_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
            puff_event_times = puff_event_times[~np.isnan(puff_event_times)]
            
            if len(puff_event_times) > 0:
                # Ensure puff_event_times is a numpy array of floats
                puff_event_times = np.array(puff_event_times, dtype=float)
                
                # Extract capacitive windows around each puff event time
                cap_puff_event_windows = []
                for puff_event_time in puff_event_times:
                    mask = (cap_time >= puff_event_time - window_puff) & (cap_time <= puff_event_time + window_puff)
                    cap_segment = cap_val[mask]
                    cap_puff_event_windows.append(cap_segment)
                
                # Pad all segments to the same length (max found)
                if cap_puff_event_windows and max(len(seg) for seg in cap_puff_event_windows) > 0:
                    max_puff_cap_len = max(len(seg) for seg in cap_puff_event_windows)
                    cap_puff_event_windows_padded = np.array([
                        np.pad(seg.astype(float), (0, max_puff_cap_len - len(seg)), constant_values=np.nan)
                        for seg in cap_puff_event_windows
                    ])
                    
                    # Apply moving median filter (21-point window) to puff event capacitive data
                    cap_puff_event_windows_filtered = apply_moving_median_15point(cap_puff_event_windows_padded)
                    
                    # Create a common time axis centered at 0 for capacitive data
                    aligned_time_puff_cap = np.linspace(-window_puff, window_puff, max_puff_cap_len)
                    
                    # Calculate mean and SEM for capacitive data
                    n_puff_event_cap = cap_puff_event_windows_filtered.shape[0]
                    mean_cap_puff_event = np.nanmean(cap_puff_event_windows_filtered, axis=0)
                    sem_cap_puff_event = np.nanstd(cap_puff_event_windows_filtered, axis=0) / np.sqrt(np.sum(~np.isnan(cap_puff_event_windows_filtered), axis=0))
                    
                    puff_event_capacitive_data = {
                        'aligned_time': aligned_time_puff_cap,
                        'mean_values': mean_cap_puff_event,
                        'sem_values': sem_cap_puff_event,
                        'n_events': n_puff_event_cap
                    }
        
        # --- Also analyze treadmill speed aligned to puff events ---
        puff_event_speed_data = None
        if 'puff_event' in trial_log_df.columns:
            puff_event_times = pd.to_numeric(trial_log_df['puff_event'], errors='coerce').dropna()
            puff_event_times = puff_event_times[~np.isnan(puff_event_times)]
            
            if len(puff_event_times) > 0:
                # Ensure puff_event_times is a numpy array of floats
                puff_event_times = np.array(puff_event_times, dtype=float)
                
                # Extract speed windows around each puff event time
                speed_puff_event_windows = []
                for puff_event_time in puff_event_times:
                    mask = (cap_time >= puff_event_time - window_puff) & (cap_time <= puff_event_time + window_puff)
                    speed_segment = speed_val[mask]
                    speed_puff_event_windows.append(speed_segment)
                
                # Pad all segments to the same length (max found)
                if speed_puff_event_windows and max(len(seg) for seg in speed_puff_event_windows) > 0:
                    max_puff_speed_len = max(len(seg) for seg in speed_puff_event_windows)
                    speed_puff_event_windows_padded = np.array([
                        np.pad(seg.astype(float), (0, max_puff_speed_len - len(seg)), constant_values=np.nan)
                        for seg in speed_puff_event_windows
                    ])
                    
                    # Create a common time axis centered at 0 for speed data
                    aligned_time_puff_speed = np.linspace(-window_puff, window_puff, max_puff_speed_len)
                    
                    # Calculate mean and SEM for speed data
                    n_puff_event_speed = speed_puff_event_windows_padded.shape[0]
                    mean_speed_puff_event = np.nanmean(speed_puff_event_windows_padded, axis=0)
                    sem_speed_puff_event = np.nanstd(speed_puff_event_windows_padded, axis=0) / np.sqrt(np.sum(~np.isnan(speed_puff_event_windows_padded), axis=0))
                    
                    puff_event_speed_data = {
                        'aligned_time': aligned_time_puff_speed,
                        'mean_values': mean_speed_puff_event,
                        'sem_values': sem_speed_puff_event,
                        'n_events': n_puff_event_speed
                    }

        # Create subplot figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # --- Plot 1: Treadmill Speed aligned to puff zone entry times ---
        axs[0].plot(aligned_time_puff, mean_speed_puff, color='red', linewidth=2, label=f'Mean Speed (n={n_puff_events})')
        axs[0].fill_between(aligned_time_puff, mean_speed_puff - sem_speed_puff, mean_speed_puff + sem_speed_puff, color='red', alpha=0.2, label='SEM')
        axs[0].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Zone Entry (t=0)')
        axs[0].set_ylabel('Treadmill Speed (interpolated)')
        axs[0].set_title(f'Average Treadmill Speed Aligned to Puff Zone Entry Times (n={n_puff_events})')
        axs[0].legend()
        axs[0].set_xlim(-5, 5)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', direction='out')
        
        # --- Plot 2: Capacitive Value aligned to puff events ---
        if puff_event_capacitive_data is not None:
            axs[1].plot(puff_event_capacitive_data['aligned_time'], puff_event_capacitive_data['mean_values'], 
                       color='blue', linewidth=2, label=f'Mean Capacitive (n={puff_event_capacitive_data["n_events"]})')
            axs[1].fill_between(puff_event_capacitive_data['aligned_time'], 
                               puff_event_capacitive_data['mean_values'] - puff_event_capacitive_data['sem_values'], 
                               puff_event_capacitive_data['mean_values'] + puff_event_capacitive_data['sem_values'], 
                               color='blue', alpha=0.2, label='SEM')
            axs[1].axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Puff Event (t=0)')
            axs[1].set_ylabel('Capacitive Value (15-pt smoothed)')
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
        axs[1].tick_params(axis='both', direction='out')
        
        # --- Plot 3: Treadmill Speed aligned to puff events ---
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
        axs[2].tick_params(axis='both', direction='out')
        
        plt.tight_layout()
        save_figure(fig, "puff_events_analysis")
        plt.show()
        
        # print(f"Puff Zone Analysis Complete:")
        # print(f"- Total puff zone entry events: {n_puff_events}")
        # print(f"- Time window: ±{window_puff} seconds around puff zone entry")
        # print(f"- Data points per trace: {max_puff_len}")
        
    else:
        print("Warning: No valid speed data found around puff zone entry times.")
        print("This could be due to puff zone times being too close to data boundaries.")
else:
    print("Warning: No valid puff zone entry times found in punish_texture_change_time data.")
    print("Check if the punish_texture_change_time data contains valid numeric values.")

plt.show()

# --- PUPIL DATA ANALYSIS SECTION ---
# This section will only run if pupil data is available

if has_pupil_data and pupil_df is not None:
    print(f"\n=== PUPIL DATA ANALYSIS ===")
    print(f"Pupil data shape: {pupil_df.shape}")
    
    # Rename the columns for clarity
    pupil_df.columns.values[0] = 'frame_number'
    pupil_df.columns.values[1] = 'point_1_x'
    pupil_df.columns.values[2] = 'point_1_y'
    pupil_df.columns.values[3] = 'point_1_likelihood'
    pupil_df.columns.values[4] = 'point_2_x'
    pupil_df.columns.values[5] = 'point_2_y'
    pupil_df.columns.values[6] = 'point_2_likelihood'
    pupil_df.columns.values[7] = 'point_3_x'
    pupil_df.columns.values[8] = 'point_3_y'
    pupil_df.columns.values[9] = 'point_3_likelihood'
    pupil_df.columns.values[10] = 'point_4_x'
    pupil_df.columns.values[11] = 'point_4_y'
    pupil_df.columns.values[12] = 'point_4_likelihood'
    pupil_df.columns.values[13] = 'point_5_x'
    pupil_df.columns.values[14] = 'point_5_y'
    pupil_df.columns.values[15] = 'point_5_likelihood'
    pupil_df.columns.values[16] = 'point_6_x'
    pupil_df.columns.values[17] = 'point_6_y'
    pupil_df.columns.values[18] = 'point_6_likelihood'
    pupil_df.columns.values[19] = 'point_7_x'
    pupil_df.columns.values[20] = 'point_7_y'
    pupil_df.columns.values[21] = 'point_7_likelihood'
    pupil_df.columns.values[22] = 'point_8_x'
    pupil_df.columns.values[23] = 'point_8_y'
    pupil_df.columns.values[24] = 'point_8_likelihood'
    
    # Calculate euclidean distance between points 3 and 7, but only for frames with high likelihood
    # Create condition: both point 3 and point 7 must have likelihood >= 0.80
    high_likelihood_mask = (pupil_df['point_3_likelihood'] >= 0.80) & (pupil_df['point_7_likelihood'] >= 0.80)
    
    # Calculate diameter only where both points have high likelihood, otherwise NaN
    pupil_df['pupil_diameter'] = np.where(
        high_likelihood_mask,
        np.sqrt((pupil_df['point_7_x'] - pupil_df['point_3_x'])**2 + 
                (pupil_df['point_7_y'] - pupil_df['point_3_y'])**2),
        np.nan
    )
    
    # Report statistics
    total_frames = len(pupil_df)
    valid_frames = high_likelihood_mask.sum()
    invalid_frames = total_frames - valid_frames
    
    print(f"\nCalculated pupil diameter as euclidean distance between points 3 and 7")
    print(f"Quality control: {valid_frames}/{total_frames} frames passed likelihood threshold (≥0.80)")
    print(f"Frames with low likelihood (set to NaN): {invalid_frames}")
    print(f"Pupil diameter stats (valid frames only): mean={pupil_df['pupil_diameter'].mean():.2f}, std={pupil_df['pupil_diameter'].std():.2f}")
    
    # Create plot of pupil diameter over frame numbers
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot pupil diameter vs frame number
    ax.plot(pupil_df['frame_number'], pupil_df['pupil_diameter'], 'b-', alpha=0.7, linewidth=0.8, label='Pupil Diameter')
    
    # Add horizontal line at mean diameter for reference
    mean_diameter = pupil_df['pupil_diameter'].mean()
    ax.axhline(y=mean_diameter, color='red', linestyle='--', alpha=0.8, label=f'Mean = {mean_diameter:.2f}')
    
    # Formatting
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Pupil Diameter (pixels)')
    ax.set_title(f'Pupil Diameter Over Time\n({valid_frames}/{total_frames} frames with likelihood ≥ 0.80)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to start from 0 if there are valid measurements
    if not pupil_df['pupil_diameter'].isna().all():
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_figure(fig, "pupil_diameter_timeseries")
    plt.show()
    
    # TODO: Add your pupil-specific analysis code here
    # The pupil data appears to be DLC (DeepLabCut) output with x, y, likelihood columns
    # for multiple body parts. Common analyses might include:
    # 1. Extract pupil coordinates and calculate pupil diameter
    # 2. Pupil diameter changes aligned to reward events
    # 3. Pupil diameter changes aligned to puff events  
    # 4. Pupil diameter changes aligned to probe events
    # 5. Correlation between pupil diameter and treadmill speed
    # 6. Correlation between pupil diameter and capacitive sensor values
    
    print("\nPupil analysis data structure inspection complete - ready for implementation")
    
else:
    print(f"\n=== PUPIL DATA ANALYSIS SKIPPED ===")
    print("No pupil data available for this session")
    print("Expected file: '*exposure.csv' (DeepLabCut output with 3-row header)")

# Show a summary of saved figures
if hasattr(save_figure, 'figure_count'):
    print(f"\nAnalysis complete! All figures have been saved as SVG files.")
    print(f"Location: {output_folder}")
    print(f"Number of figures saved: {save_figure.figure_count - 1}")
else:
    print("\nAnalysis complete, but no figures were saved.")