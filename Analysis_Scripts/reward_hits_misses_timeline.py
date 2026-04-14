"""
Timeline plot showing reward and punishment zone outcomes across a session.

Two stacked panels:
  Top panel    — Reward zone events
    Green  : Hit             (reward zone → reward delivered)
    Purple : Miss            (reward zone → no reward)
  Bottom panel — Punishment zone events
    Blue   : Correct rejection (punishment zone → no puff; kept running)
    Red    : False alarm       (punishment zone → puff delivered; stopped)

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
    """Load trial log and treadmill CSV files"""
    trial_log_files = [f for f in os.listdir(folder_path) if 'trial_log.csv' in f]
    treadmill_files = [f for f in os.listdir(folder_path) if 'treadmill.csv' in f]

    if not trial_log_files or not treadmill_files:
        print("Error: Missing required files (trial_log.csv or treadmill.csv)")
        return None

    print(f"\nLoading files:")
    print(f"  - {trial_log_files[0]}")
    print(f"  - {treadmill_files[0]}")

    data = {
        'trial_log': pd.read_csv(os.path.join(folder_path, trial_log_files[0]), engine='python'),
        'treadmill': pd.read_csv(os.path.join(folder_path, treadmill_files[0]), comment='/', engine='python')
    }

    print("Files loaded successfully.\n")
    return data


def uniformly_sample_treadmill(treadmill_df):
    """Uniformly sample treadmill speed at 50 Hz and convert mm/s to cm/s."""
    time_min = treadmill_df['global_time'].min()
    time_max = treadmill_df['global_time'].max()
    uniform_time = np.arange(time_min, time_max, 1.0 / 50.0)
    uniform_speed = np.interp(
        uniform_time,
        treadmill_df['global_time'].values,
        treadmill_df['speed'].values
    ) / 10.0  # Convert mm/s to cm/s
    return pd.Series(uniform_speed, index=uniform_time)


def extract_reward_zone_entries(trial_log_df):
    """Extract reward zone entry times, excluding re-entries.

    New hallway format (stay_texture_change_time column):
        Reads stay_texture_change_time as a scalar per row via pd.to_numeric.
        Re-entries are excluded with a 0.05 s tolerance to match the source
        recording logic (same as longitudinal_analysis_new_hallway-Sally.py).

    Old hallway format (fallback):
        Loops texture_history / texture_change_time lists, keeping only
        entries where texture == 'assets/reward_mean100.jpg'.

    Returns:
        list[float]: Sorted zone entry timestamps (seconds), re-entries excluded.
    """
    # New hallway format ──────────────────────────────────────────────────────
    if 'stay_texture_change_time' in trial_log_df.columns:
        raw_times = pd.to_numeric(
            trial_log_df['stay_texture_change_time'], errors='coerce'
        ).dropna().values

        # Build re-entry set from zone_re_entry_time column
        re_entry_times_set = set()
        if 'zone_re_entry_time' in trial_log_df.columns:
            for val in trial_log_df['zone_re_entry_time']:
                if pd.isna(val) or val == '':
                    continue
                try:
                    if isinstance(val, str) and val.strip():
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple)):
                            for t in parsed:
                                if pd.notna(t):
                                    re_entry_times_set.add(float(t))
                        else:
                            re_entry_times_set.add(float(parsed))
                    elif not isinstance(val, str):
                        re_entry_times_set.add(float(val))
                except (ValueError, TypeError):
                    pass

        if re_entry_times_set:
            re_entry_arr = np.array(sorted(re_entry_times_set))
            zone_entry_times = [
                float(t) for t in raw_times
                if t > 0 and np.min(np.abs(re_entry_arr - t)) > 0.05
            ]
        else:
            zone_entry_times = [float(t) for t in raw_times if t > 0]

        return sorted(zone_entry_times)

    # Old hallway format (fallback) ───────────────────────────────────────────
    zone_entry_times = []
    for _, log_row in trial_log_df.iterrows():
        texture_hist  = safe_literal_eval(log_row.get('texture_history',  '[]'))
        texture_times = safe_literal_eval(log_row.get('texture_change_time', '[]'))
        for i, texture in enumerate(texture_hist):
            if texture == "assets/reward_mean100.jpg" and i < len(texture_times):
                try:
                    t = float(texture_times[i])
                except (TypeError, ValueError):
                    continue
                if not np.isnan(t) and t > 0:
                    zone_entry_times.append(t)
    return sorted(zone_entry_times)


def extract_reward_events(trial_log_df):
    """Extract all reward delivery times (reward_event and hits_event combined).

    Returns:
        list[float]: Sorted reward event timestamps.
    """
    hit_times = []
    for trial_idx in range(len(trial_log_df)):
        t = pd.to_numeric(trial_log_df.loc[trial_idx, 'reward_event'], errors='coerce')
        if pd.notna(t) and t > 0:
            hit_times.append(float(t))
    if 'hits_event' in trial_log_df.columns:
        for trial_idx in range(len(trial_log_df)):
            t = pd.to_numeric(trial_log_df.loc[trial_idx, 'hits_event'], errors='coerce')
            if pd.notna(t) and t > 0:
                hit_times.append(float(t))
    return sorted(hit_times)


def extract_punish_zone_entries(trial_log_df):
    """Extract punishment zone entry times.

    New hallway format (go_texture_change_time column):
        Every non-null value in go_texture_change_time is a punishment zone
        entry.  The length of this column directly gives the number of
        punishment zone encounters.

    Old hallway format (fallback via texture_history list-per-row):
        Loops texture_history / texture_change_time lists, keeping only
        entries where texture == 'assets/punish_mean100.jpg'.

    Returns:
        list[float]: Sorted punish zone entry timestamps (seconds).
    """
    # New hallway format ───────────────────────────────────────────────────────
    if 'go_texture_change_time' in trial_log_df.columns:
        raw_times = pd.to_numeric(
            trial_log_df['go_texture_change_time'], errors='coerce'
        ).dropna().values
        return sorted(float(t) for t in raw_times if t > 0)

    # Old hallway format (fallback) ───────────────────────────────────────────
    zone_entry_times = []
    for _, log_row in trial_log_df.iterrows():
        texture_hist  = safe_literal_eval(log_row.get('texture_history',  '[]'))
        texture_times = safe_literal_eval(log_row.get('texture_change_time', '[]'))
        for i, texture in enumerate(texture_hist):
            if texture == 'assets/punish_mean100.jpg' and i < len(texture_times):
                try:
                    t = float(texture_times[i])
                except (TypeError, ValueError):
                    continue
                if not np.isnan(t) and t > 0:
                    zone_entry_times.append(t)
    return sorted(zone_entry_times)


def extract_puff_events(trial_log_df):
    """Extract all puff (air-puff) delivery times.

    Returns:
        list[float]: Sorted puff event timestamps.
    """
    puff_times = []
    if 'puff_event' not in trial_log_df.columns:
        return puff_times
    for trial_idx in range(len(trial_log_df)):
        t = pd.to_numeric(trial_log_df.loc[trial_idx, 'puff_event'], errors='coerce')
        if pd.notna(t) and t > 0:
            puff_times.append(float(t))
    return sorted(puff_times)


def classify_correct_rejections_and_false_alarms(punish_zone_times, puff_event_times):
    """Classify punishment zone entries as correct rejections or false alarms.

    Uses the same reward-centric backward nearest-prior matching as
    daily_analysis_new_hallway.py: each puff event claims the single most
    recent preceding punishment zone.  Zones not claimed by any puff are
    correct rejections.

    Args:
        punish_zone_times: list[float] of punishment zone entry times
        puff_event_times:  list[float] of puff event times

    Returns:
        (correct_rejections, false_alarms): each a list[float] of zone times
    """
    punish_zone_arr = np.array(sorted(punish_zone_times))
    matched_zone_set = set()

    for t_puff in puff_event_times:
        prior_zones = punish_zone_arr[punish_zone_arr < t_puff]
        if len(prior_zones) > 0:
            matched_zone_set.add(float(prior_zones[-1]))

    false_alarms = [t for t in punish_zone_times if t in matched_zone_set]
    correct_rejections = [t for t in punish_zone_times if t not in matched_zone_set]
    return correct_rejections, false_alarms


def classify_hits_and_misses(reward_zone_times, reward_event_times):
    """Classify reward zones as hits or misses.

    Uses the same reward-centric backward nearest-prior matching as
    daily_analysis_new_hallway.py: each reward event claims the single most
    recent preceding reward zone.  Zones not claimed by any reward are misses.

    Args:
        reward_zone_times:  list[float] of reward zone entry times
        reward_event_times: list[float] of reward event times

    Returns:
        (hits, misses): each a list[float] of zone entry times
    """
    reward_zone_arr = np.array(sorted(reward_zone_times))
    matched_zone_set = set()

    for t_reward in reward_event_times:
        prior_zones = reward_zone_arr[reward_zone_arr < t_reward]
        if len(prior_zones) > 0:
            matched_zone_set.add(float(prior_zones[-1]))

    hits = [t for t in reward_zone_times if t in matched_zone_set]
    misses = [t for t in reward_zone_times if t not in matched_zone_set]
    return hits, misses


def plot_behavioral_timeline(treadmill_interp, hits, misses,
                             correct_rejections, false_alarms, output_folder):
    """Create a single-panel timeline with all four event types as vertical lines.

    Green      : Hit              (reward zone → reward delivered)
    Purple     : Miss             (reward zone → no reward)
    Steel blue : Correct rejection (punishment zone → no puff; kept running)
    Red        : False alarm       (punishment zone → puff delivered; stopped)
    """
    fig, ax = plt.subplots(figsize=(16, 4))

    time_min = treadmill_interp.index.min()
    time_max = treadmill_interp.index.max()

    for i, t in enumerate(hits):
        ax.axvline(x=t, color='#004492', alpha=1.0, linewidth=3.0,
                   label='Hit' if i == 0 else None)
    for i, t in enumerate(misses):
        ax.axvline(x=t, color='#b0e0e6', alpha=1.0, linewidth=3.0,
                   label='Miss' if i == 0 else None)
    for i, t in enumerate(correct_rejections):
        ax.axvline(x=t, color='#ec958d', alpha=1.0, linewidth=3.0,
                   label='Correct Rejection' if i == 0 else None)
    for i, t in enumerate(false_alarms):
        ax.axvline(x=t, color='#a24147', alpha=1.0, linewidth=3.0,
                   label='False Alarm' if i == 0 else None)

    ax.set_xlabel('Elapsed Time (s)', fontsize=15)
    ax.set_title('Behavioral Outcomes Across Session', fontsize=18, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(output_folder, "behavioral_timeline.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")

    plt.show()


def main():
    """Main execution function"""
    print("=" * 60)
    print("BEHAVIORAL TIMELINE")
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
    
    # Sample treadmill speed at native 50 Hz
    print("Sampling treadmill speed at 50 Hz...")
    treadmill_interp = uniformly_sample_treadmill(data['treadmill'])

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

    # Extract punishment zone entries and puff events
    print("\nExtracting punishment zone entries...")
    all_punish_zones = extract_punish_zone_entries(data['trial_log'])
    print(f"Found {len(all_punish_zones)} punishment zone entries")

    print("Extracting puff events...")
    puff_events = extract_puff_events(data['trial_log'])
    print(f"Found {len(puff_events)} puff delivery events")

    # Classify correct rejections and false alarms
    print("\nClassifying correct rejections and false alarms...")
    correct_rejections, false_alarms = classify_correct_rejections_and_false_alarms(
        all_punish_zones, puff_events
    )
    print(f"  Correct rejections (kept running): {len(correct_rejections)}")
    print(f"  False alarms (stopped → puff):     {len(false_alarms)}")

    # Create plot
    print("\nGenerating timeline plot...")
    plot_behavioral_timeline(
        treadmill_interp, hits, misses, correct_rejections, false_alarms, output_folder
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
