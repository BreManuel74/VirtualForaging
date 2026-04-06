"""Longitudinal analysis of behavioral data across multiple mice.
Original Author: Brenna Manuel

TEST COUNTING LOGIC:
To test the zone/event counting logic on a single trial_log file before running the full analysis:
    
    from longitudinal_analysis_new_hallway import test_matching_logic
    test_matching_logic('path/to/your/trial_log.csv')
    
Or run this script and call the function interactively in Python console.
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
import pickle
import hashlib
import warnings
import math

# Add Analysis_Scripts to path to import lick detection algorithm
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import lick_detection_algorithm as lda
from scipy.stats import norm
from scipy.signal import butter, filtfilt
from timeline_refactored import (
    safe_literal_eval,
    uniformly_sample_treadmill,
    uniformly_sample_capacitive,
    create_aligned_windows,
)

# Cache directory for KDE values
KDE_CACHE_DIR = os.path.join(script_dir, '.kde_cache')
if not os.path.exists(KDE_CACHE_DIR):
    os.makedirs(KDE_CACHE_DIR)

def get_file_hash(filepath):
    """Generate a hash based on file path and modification time."""
    stat = os.stat(filepath)
    # Use file path, size, and modification time to create unique identifier
    hash_input = f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()

def get_cached_kde(capacitive_filepath):
    """Retrieve cached KDE value for a capacitive file if it exists."""
    file_hash = get_file_hash(capacitive_filepath)
    cache_file = os.path.join(KDE_CACHE_DIR, f"{file_hash}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['kde_value']
        except Exception as e:
            # If cache is corrupted, ignore and recalculate
            print(f"Warning: Cache read failed for {capacitive_filepath}: {e}")
            return None
    return None

def cache_kde_value(capacitive_filepath, kde_value):
    """Cache the KDE value for a capacitive file."""
    file_hash = get_file_hash(capacitive_filepath)
    cache_file = os.path.join(KDE_CACHE_DIR, f"{file_hash}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'kde_value': kde_value}, f)
    except Exception as e:
        # If caching fails, just continue without caching
        print(f"Warning: Cache write failed for {capacitive_filepath}: {e}")

def compute_session_distance(treadmill_df):
    """Compute total distance traversed in a session from raw treadmill data.

    Finds the first non-zero value in the 'distance' column, subtracts it from
    all rows to normalise the cumulative odometer to zero at the start of
    motion, then returns both the full normalised series and the total distance
    (final normalised value) for the session.

    Parameters
    ----------
    treadmill_df : pd.DataFrame
        Raw treadmill CSV loaded as a DataFrame (must contain a 'distance' column).

    Returns
    -------
    normalised : pd.Series
        Per-row cumulative distance from the first non-zero sample (same index
        as treadmill_df).  All rows before the first non-zero value are NaN.
    total_distance : float
        Distance traversed from first non-zero sample to end of session (cm).
        NaN if no non-zero distance is found.
    """
    dist = pd.to_numeric(treadmill_df['distance'], errors='coerce')
    nonzero_mask = dist != 0
    if not nonzero_mask.any():
        return pd.Series(np.nan, index=treadmill_df.index), float('nan')

    first_nonzero_val = dist[nonzero_mask].iloc[0]
    first_nonzero_idx = nonzero_mask.idxmax()           # label of first True

    normalised = dist.copy()
    # Rows before first non-zero are meaningless pre-motion artefacts; set NaN
    normalised.loc[:first_nonzero_idx] = np.nan
    normalised = normalised - first_nonzero_val          # shift so first point = 0
    normalised.loc[first_nonzero_idx] = 0.0              # ensure exactly 0

    total_distance = float(normalised.dropna().iloc[-1]) if len(normalised.dropna()) > 0 else float('nan')
    return normalised, total_distance


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

# ── Locomotion bout detection constants ──────────────────────────────────────
BOUT_THRESHOLD_CM_S  = 2.0   # cm/s — minimum speed to count as moving
MIN_BOUT_DURATION_S  = 2.0   # s    — minimum continuous time above threshold
MAX_INTER_BOUT_GAP_S = 2.0   # s    — gaps <= this between bouts are merged


def detect_locomotion_bouts(time, speed_cm_s,
                             threshold=BOUT_THRESHOLD_CM_S,
                             min_duration=MIN_BOUT_DURATION_S,
                             max_inter_bout_gap=MAX_INTER_BOUT_GAP_S):
    """
    1. Build contiguous above/below-threshold runs.
    2. Bridge short below-threshold gaps (<= max_inter_bout_gap).
    3. Keep merged above-threshold spans >= min_duration.
    Returns list of (t_start, t_end) tuples in seconds.
    """
    above = speed_cm_s >= threshold
    n = len(above)

    runs = []
    start = 0
    for i in range(1, n):
        if above[i] != above[start]:
            runs.append((bool(above[start]), start, i - 1))
            start = i
    runs.append((bool(above[start]), start, n - 1))

    merged = []
    i = 0
    while i < len(runs):
        if runs[i][0]:
            j = i
            while j + 2 < len(runs):
                gap_dur = time[runs[j + 1][2]] - time[runs[j + 1][1]]
                if (not runs[j + 1][0]) and gap_dur <= max_inter_bout_gap:
                    j += 2
                else:
                    break
            merged.append((True, runs[i][1], runs[j][2]))
            i = j + 1
        else:
            merged.append(runs[i])
            i += 1

    bouts = []
    for is_above, s, e in merged:
        if is_above and (time[e] - time[s]) >= min_duration:
            bouts.append((time[s], time[e]))
    return bouts


def test_matching_logic(trial_log_path):
    """
    Test the zone/event counting logic on a single trial_log CSV file.
    
    Logic:
    - Valid zones (opportunities) = stay_texture_change_time entries (excluding re-entries within 0.05s)
    - Hits = count(reward_event) + count(hits_event)
    - Misses = valid zones - hits
    - Sensitivity = hits / valid zones
    
    Args:
        trial_log_path: Path to a single trial_log.csv file
    
    Example usage:
        test_matching_logic('f:/virtual_foraging_cohort_4/VF11/Session_20/beh/1774635104trial_log.csv')
    """
    print("\n" + "="*80)
    print("TESTING ZONE/EVENT COUNTING LOGIC")
    print("="*80)
    print(f"File: {trial_log_path}")
    
    # Read trial log
    trial_log = pd.read_csv(trial_log_path)
    
    print(f"\nTotal rows in trial_log: {len(trial_log)}")
    
    # Extract ALL stay zone entry times (if column exists; older data may not have it)
    if 'stay_texture_change_time' in trial_log.columns:
        stay_zone_times = pd.to_numeric(trial_log['stay_texture_change_time'], errors='coerce').dropna().values
        print(f"Total stay zone entry times: {len(stay_zone_times)}")
        
        # Collect all re-entry times to exclude them
        re_entry_times_set = set()
        if 'zone_re_entry_time' in trial_log.columns:
            for val in trial_log['zone_re_entry_time']:
                if pd.notna(val) and val != '':
                    try:
                        # Handle both single values and arrays
                        if isinstance(val, str) and val.strip():
                            import ast
                            try:
                                re_entry_list = ast.literal_eval(val)
                                if isinstance(re_entry_list, (list, tuple)):
                                    for t in re_entry_list:
                                        if pd.notna(t):
                                            re_entry_times_set.add(float(t))
                                else:
                                    re_entry_times_set.add(float(re_entry_list))
                            except:
                                pass
                        elif not isinstance(val, str):
                            re_entry_times_set.add(float(val))
                    except (ValueError, TypeError):
                        pass
        
        print(f"Re-entry times found: {len(re_entry_times_set)}")
        if len(re_entry_times_set) > 0:
            print(f"  Sample re-entry times: {sorted(list(re_entry_times_set))[:5]}")
        
        # Filter out stay_zone_times that match re-entries (within 0.05 seconds)
        valid_zone_times = []
        excluded_zones = []
        excluded_details = []  # Store details for reporting
        for zone_time in stay_zone_times:
            is_reentry = False
            for re_entry_time in re_entry_times_set:
                if abs(zone_time - re_entry_time) <= 0.05:
                    is_reentry = True
                    excluded_zones.append(zone_time)
                    excluded_details.append((zone_time, re_entry_time, abs(zone_time - re_entry_time)))
                    break
            if not is_reentry:
                valid_zone_times.append(zone_time)
        
        valid_zone_times = np.array(valid_zone_times)
        print(f"\nZones excluded as re-entries: {len(excluded_zones)}")
        if len(excluded_zones) > 0:
            print(f"  Excluded zone details (first 5):")
            for zone, reentry, diff in sorted(excluded_details)[:5]:
                print(f"    Zone {zone:.2f} ← Re-entry {reentry:.2f} (diff={diff:.3f}s)")
            if len(excluded_zones) > 5:
                print(f"    ... and {len(excluded_zones) - 5} more")
        print(f"Valid zone times (after re-entry filtering): {len(valid_zone_times)}")
        total_opportunities = len(valid_zone_times)
    else:
        print("\nNote: 'stay_texture_change_time' not found. Using texture_history fallback for opportunities.")
        total_opportunities = len(trial_log[trial_log['texture_history'] == 'assets/reward_mean100.jpg'])
        print(f"Reward opportunities (texture_history fallback): {total_opportunities}")
    
    # Collect all reward_event and hits_event times
    reward_event_times = pd.to_numeric(trial_log['reward_event'], errors='coerce').dropna().values
    
    # Check if hits_event column exists (older data may not have it)
    if 'hits_event' in trial_log.columns:
        hits_event_times = pd.to_numeric(trial_log['hits_event'], errors='coerce').dropna().values
    else:
        hits_event_times = np.array([])  # Empty array if column doesn't exist
        print("\nNote: 'hits_event' column not found. Using only 'reward_event' as hits.")
    
    print(f"\nReward events found: {len(reward_event_times)}")
    print(f"Hits events found: {len(hits_event_times)}")
    
    # Simple counting: hits = rewards + hits, misses = opportunities - hits
    if total_opportunities > 0:
        reward_count = len(reward_event_times) + len(hits_event_times)
        misses = total_opportunities - reward_count
        sensitivity = float(reward_count) / float(total_opportunities) if total_opportunities > 0 else 0.0
        
        print(f"\n" + "-"*80)
        print("FINAL RESULTS:")
        print("-"*80)
        print(f"Total valid zones (opportunities): {total_opportunities}")
        print(f"Hits (total successful outcomes): {reward_count}")
        print(f"  - reward_event count: {len(reward_event_times)}")
        print(f"  - hits_event count: {len(hits_event_times)}")
        print(f"Misses: {misses}")
        print(f"Sensitivity: {sensitivity:.3f} ({reward_count}/{total_opportunities})")
        print("="*80 + "\n")
    else:
        print("\nNo valid zones found!")
        print("="*80 + "\n")

# ── Behavioral epoch constants & helpers ────────────────────────────────────
EPOCH_WINDOW_S       = 5            # half-window size (seconds each side of event)
EPOCH_N_SAMPLES      = 501          # canonical time-axis length (10 s / 0.02 s + 1)
EPOCH_CANONICAL_TIME = np.linspace(-EPOCH_WINDOW_S, EPOCH_WINDOW_S, EPOCH_N_SAMPLES)


def _extract_reward_zone_entry_times(trial_log_df):
    """Return sorted list of reward zone entry timestamps from a trial log.

    Handles two trial-log formats and excludes re-entries in the new format:

    New hallway format (preferred, detected by 'stay_texture_change_time' column):
        Each row may contain a scalar zone-entry timestamp in
        'stay_texture_change_time'.  Any entry whose time matches a timestamp
        in 'zone_re_entry_time' (within 0.05 s) is excluded as a re-entry.

    Old hallway format (fallback):
        Each row stores lists in 'texture_history' and 'texture_change_time'.
        Every index where texture == 'assets/reward_mean100.jpg' yields one
        zone-entry timestamp.

    Parameters
    ----------
    trial_log_df : pd.DataFrame
        Trial log loaded from a *_trial_log.csv file.

    Returns
    -------
    list[float] : Sorted zone entry timestamps (seconds), re-entries excluded.
    """
    # ── New hallway format ────────────────────────────────────────────────────
    if 'stay_texture_change_time' in trial_log_df.columns:
        raw_times = pd.to_numeric(
            trial_log_df['stay_texture_change_time'], errors='coerce'
        ).dropna().values

        # Build re-entry set (values can be scalars, lists, or string-encoded lists)
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

        # Filter out re-entries (tolerance 0.05 s, matching the source code)
        if re_entry_times_set:
            re_entry_arr = np.array(sorted(re_entry_times_set))
            zone_entry_times = [
                float(t) for t in raw_times
                if np.min(np.abs(re_entry_arr - t)) > 0.05
            ]
        else:
            zone_entry_times = [float(t) for t in raw_times if t > 0]

        return sorted(zone_entry_times)

    # ── Old hallway format (fallback) ─────────────────────────────────────────
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


def _build_epoch_matrix(time_array, data_array, event_times,
                        window_s=EPOCH_WINDOW_S,
                        canonical_time=EPOCH_CANONICAL_TIME):
    """Extract ±window_s windows around each event and resample to a canonical grid.

    Parameters
    ----------
    time_array     : np.ndarray (1-D) — uniformly sampled time axis (seconds).
    data_array     : np.ndarray (1-D) — signal values, same length as time_array.
    event_times    : sequence of float — event timestamps (seconds).
    window_s       : float — half-window size in seconds.
    canonical_time : np.ndarray (1-D) — target time axis for every window.

    Returns
    -------
    np.ndarray of shape (n_events, len(canonical_time)), dtype float64.
    Values outside the recorded range are NaN (events near session edges).
    Returns None if event_times is empty.
    """
    if len(event_times) == 0:
        return None
    n_pts = len(canonical_time)
    rows  = []
    for t0 in event_times:
        mask     = (time_array >= t0 - window_s) & (time_array <= t0 + window_s)
        seg_time = time_array[mask]
        seg_data = data_array[mask].astype(float)
        if len(seg_time) < 2:
            rows.append(np.full(n_pts, np.nan))
            continue
        rel_time = seg_time - t0
        row = np.interp(canonical_time, rel_time, seg_data, left=np.nan, right=np.nan)
        rows.append(row)
    return np.array(rows, dtype=float)


# ── Plot selection ────────────────────────────────────────────────────────────
_ALL_PLOT_KEYS = {
    'speed', 'sensitivity', 'lick_count', 'reward_count',
    'false_alarms', 'correct_rejections', 'specificity', 'dprime',
    'avg_reward', 'sex_reward',
    'distance', 'sex_distance', 'condition_distance',
    'condition_distance_bar', 'total_distance_bar',
    'condition_reward', 'condition_speed', 'condition_lick', 'condition_bar', 'condition_speed_bar',
    'levels', 'level_speed', 'level_speed_condition',
    'avg_lick_rate', 'sex_lick_rate', 'condition_lick_rate', 'condition_lick_bar',
    'level_lick', 'level_lick_condition',
    'level_dist', 'level_dist_condition', 'level_dist_condition_excl_last',
    'bout_count', 'avg_bout_count', 'condition_bout_count', 'condition_bout_count_bar',
    'level_bout', 'level_bout_condition',
    'bout_avg_speed', 'condition_bout_avg_speed', 'condition_bout_avg_speed_bar',
    'bout_avg_dist',  'condition_bout_avg_dist',  'condition_bout_avg_dist_bar',
    'level_bout_avg_speed', 'level_bout_avg_speed_condition',
    'level_bout_avg_dist',  'level_bout_avg_dist_condition',
    'epoch_reward_speed', 'epoch_reward_cap',
    'epoch_reward_speed_sess', 'epoch_reward_cap_sess',
    'epoch_reward_speed_early_late', 'epoch_reward_cap_early_late',
    'epoch_reward_speed_early_late_ev', 'epoch_reward_cap_early_late_ev',
}

_PLOT_LABELS = [
    ('speed',               'Individual: Average speed over time'),
    ('sensitivity',         'Individual: Sensitivity over time'),
    ('lick_count',          'Individual: Lick count over time'),
    ('reward_count',        'Individual: Reward count over time'),
    ('false_alarms',        'Individual: False alarms over time'),
    ('correct_rejections',  'Individual: Correct rejections over time'),
    ('specificity',         'Individual: Specificity over time'),
    ('dprime',              "Individual: d' over time"),
    ('avg_reward',          'Aggregate: Average reward rate across all mice'),
    ('sex_reward',          'Aggregate: Sex-specific average reward rate'),
    ('distance',            'Individual: Total distance per session (m)'),
    ('bout_count',               'Individual: Locomotion bout count per session'),
    ('avg_bout_count',           'Aggregate: Average bout count across all mice'),
    ('condition_bout_count',     'Condition: Bout count over time by condition'),
    ('condition_bout_count_bar', 'Condition: Average bout count — collapsed bar chart'),
    ('sex_distance',        'Aggregate: Sex-specific average distance per session (m)'),
    ('condition_distance',  'Condition: Distance per session by starting condition (m)'),
    ('condition_distance_bar', 'Condition: Average distance per session — collapsed bar chart (m)'),
    ('total_distance_bar',    'Condition: Total distance per mouse — collapsed bar chart (m)'),
    ('avg_lick_rate',       'Aggregate: Average lick rate across all mice'),
    ('sex_lick_rate',       'Aggregate: Sex-specific average lick rate'),
    ('condition_reward',    'Condition: Reward rate over time (line)'),
    ('condition_speed',     'Condition: Speed over time'),
    ('condition_lick',      'Condition: Lick count over time'),
    ('condition_lick_rate', 'Condition: Lick rate over time (line)'),
    ('condition_bar',       'Condition: Reward rate — collapsed bar chart'),
    ('condition_speed_bar', 'Condition: Average speed — collapsed bar chart'),
    ('condition_lick_bar',  'Condition: Lick rate — collapsed bar chart'),
    ('levels',              'Level: Reward rate by level (requires transitions CSV)'),
    ('level_speed',           'Level: Average speed by level — collapsed (requires transitions CSV)'),
    ('level_speed_condition', 'Level: Average speed by level — by condition (requires transitions CSV)'),
    ('level_lick',            'Level: Average lick rate by level — collapsed (requires transitions CSV)'),
    ('level_lick_condition',  'Level: Average lick rate by level — by condition (requires transitions CSV)'),
    ('level_dist',            'Level: Distance traveled by level — collapsed (requires transitions CSV)'),
    ('level_dist_condition',  'Level: Distance traveled by level — by condition (requires transitions CSV)'),
    ('level_dist_condition_excl_last', 'Level: Distance traveled by level — by condition, last level excluded (requires transitions CSV)'),
    ('level_bout',           'Level: Locomotion bout count by level — collapsed (requires transitions CSV)'),
    ('level_bout_condition', 'Level: Locomotion bout count by level — by condition (requires transitions CSV)'),
    ('level_bout_avg_speed',           'Level: Avg speed per bout by level — collapsed (requires transitions CSV)'),
    ('level_bout_avg_speed_condition', 'Level: Avg speed per bout by level — by condition (requires transitions CSV)'),
    ('level_bout_avg_dist',            'Level: Avg distance per bout by level — collapsed (requires transitions CSV)'),
    ('level_bout_avg_dist_condition',  'Level: Avg distance per bout by level — by condition (requires transitions CSV)'),
    ('bout_avg_speed',               'Individual: Average speed per locomotion bout'),
    ('condition_bout_avg_speed',     'Condition: Average speed per bout over time'),
    ('condition_bout_avg_speed_bar', 'Condition: Average speed per bout — collapsed bar chart'),
    ('bout_avg_dist',                'Individual: Average distance per locomotion bout'),
    ('condition_bout_avg_dist',      'Condition: Average distance per bout over time'),
    ('condition_bout_avg_dist_bar',  'Condition: Average distance per bout — collapsed bar chart'),
    ('epoch_reward_speed',      'Epoch: Treadmill speed — event-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_cap',        'Epoch: Capacitive value — event-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_speed_sess', 'Epoch: Treadmill speed — session-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_cap_sess',   'Epoch: Capacitive value — session-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_speed_early_late',    'Epoch: Treadmill speed — early vs late sessions, session-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_cap_early_late',      'Epoch: Capacitive value — early vs late sessions, session-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_speed_early_late_ev', 'Epoch: Treadmill speed — early vs late sessions, event-averaged, aligned to reward zone entry (per-mouse + condition)'),
    ('epoch_reward_cap_early_late_ev',   'Epoch: Capacitive value — early vs late sessions, event-averaged, aligned to reward zone entry (per-mouse + condition)'),
]


def _ask_plot_selection(root):
    """Show a scrollable checkbox dialog and return the frozenset of selected plot keys."""
    dialog = tk.Toplevel(root)
    dialog.title('Select Plots to Generate')
    dialog.resizable(True, True)
    dialog.grab_set()

    # ── Header (outside the scroll area) ─────────────────────────────────────
    tk.Label(dialog, text='Select which plots to generate:',
             font=('Arial', 11, 'bold')).pack(anchor='w', padx=14, pady=(12, 4))

    # ── Scrollable canvas + inner frame ──────────────────────────────────────
    container = tk.Frame(dialog)
    container.pack(fill='both', expand=True, padx=14, pady=4)

    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)

    inner = tk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')

    def _on_inner_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    def _on_canvas_configure(event):
        canvas.itemconfig(inner_id, width=event.width)

    inner.bind('<Configure>', _on_inner_configure)
    canvas.bind('<Configure>', _on_canvas_configure)

    # Mouse-wheel scrolling (cross-platform)
    def _on_mousewheel(event):
        if event.num == 4:          # Linux scroll up
            canvas.yview_scroll(-1, 'units')
        elif event.num == 5:        # Linux scroll down
            canvas.yview_scroll(1, 'units')
        else:                       # Windows / macOS
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    canvas.bind('<MouseWheel>', _on_mousewheel)
    canvas.bind('<Button-4>',   _on_mousewheel)
    canvas.bind('<Button-5>',   _on_mousewheel)
    inner.bind('<MouseWheel>',  _on_mousewheel)
    inner.bind('<Button-4>',    _on_mousewheel)
    inner.bind('<Button-5>',    _on_mousewheel)

    # ── Checkboxes ────────────────────────────────────────────────────────────
    vars_ = {}
    for key, label in _PLOT_LABELS:
        var = tk.BooleanVar(value=True)
        vars_[key] = var
        cb = tk.Checkbutton(inner, text=label, variable=var, anchor='w')
        cb.pack(fill='x', padx=8, pady=1)
        cb.bind('<MouseWheel>', _on_mousewheel)
        cb.bind('<Button-4>',   _on_mousewheel)
        cb.bind('<Button-5>',   _on_mousewheel)

    # Cap the dialog height to 80 % of the screen
    dialog.update_idletasks()
    screen_h = dialog.winfo_screenheight()
    max_h    = int(screen_h * 0.80)
    natural_h = inner.winfo_reqheight() + 120   # header + buttons
    dialog_h  = min(natural_h, max_h)
    dialog.geometry(f'520x{dialog_h}')

    # ── Buttons (outside the scroll area) ────────────────────────────────────
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(fill='x', padx=14, pady=(8, 12))

    def _select_all():
        for v in vars_.values():
            v.set(True)

    def _deselect_all():
        for v in vars_.values():
            v.set(False)

    tk.Button(btn_frame, text='Select All',   command=_select_all).pack(side='left',  padx=4)
    tk.Button(btn_frame, text='Deselect All', command=_deselect_all).pack(side='left', padx=4)
    tk.Button(btn_frame, text='OK', command=dialog.destroy,
              default='active').pack(side='right', padx=4)

    root.wait_window(dialog)
    return frozenset(key for key, var in vars_.items() if var.get())


def analyze_levels(data_files, transitions_csv_path, animal_conditions=None, selected_plots=None):
    """Analyze rewards/min for each level across all mice, split by starting condition.

    Uses a transitions CSV (produced by level_sorter.py) to slice each session's
    trial_log into per-level time windows.  Multi-session level visits (where a
    mouse ends a session mid-level and continues the next day) are aggregated
    before computing rpm: total rewards and total active time are summed across
    ALL sessions for a given (animal, level) pair, producing ONE rpm value per
    animal per level.  This prevents time gaps between sessions from penalising
    the rate calculation and ensures partial sessions at either end of a level
    are fully included.

    Boundary rules per session slice:
        start_ts = previous level's transition_ts in this session, or 0
        end_ts   = this level's transition_ts (completed) or capacitive
                   elapsed_time.max() (incomplete last level of session)

    Parameters
    ----------
    animal_conditions : dict[str, str] | None
        Mapping of animal_id -> starting_condition (e.g. {'CAH1': 'SC', 'CAH2': 'VF'}).
        When provided, the bar chart is split into grouped bars by condition.
    """
    # (animal_id, level) -> {'rewards': int, 'duration_min': float, 'condition': str}
    animal_level_accum: dict[tuple, dict] = {}
    # animal_id -> last level seen (updated each session; final value = potentially incomplete level)
    animal_last_level: dict[str, str] = {}

    # Load transitions CSV -------------------------------------------------------
    if not transitions_csv_path:
        print("  [WARN] No transitions CSV provided — level plot will be empty.")
        return plt.figure(figsize=(15, 8)), None, None, None, None, None, None, None, None, None, None, None, None, None, None
    try:
        transitions_df = pd.read_csv(transitions_csv_path)
        transitions_df['date'] = pd.to_datetime(transitions_df['date'])
    except Exception as e:
        print(f"  [ERROR] Cannot read transitions CSV: {e}")
        return plt.figure(figsize=(15, 8)), None, None, None, None, None, None, None, None, None, None, None, None, None, None

    # Build lookup (animal_id, date_normalised) -> {trial_log, capacitive} ------
    session_lookup = {}
    for data_file in data_files:
        animal_id = os.path.basename(data_file).split('_')[0]
        try:
            df = pd.read_csv(data_file)
        except Exception:
            continue
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        for _, row in df.iterrows():
            date_key = pd.Timestamp(row['date']).normalize()
            session_lookup[(animal_id, date_key)] = {
                'trial_log':  row.get('trial_log'),
                'capacitive': row.get('capacitive'),
                'treadmill':  row.get('treadmill'),
            }

    # Accumulate rewards and time per (animal, level) across all sessions ---------
    for animal_id, animal_group in transitions_df.groupby('animal_id'):
        for session_num, session_group in animal_group.groupby('session_num'):
            session_group = session_group.reset_index(drop=True)
            date_key = pd.Timestamp(session_group.iloc[0]['date']).normalize()

            sess_info = session_lookup.get((animal_id, date_key))
            if sess_info is None:
                continue
            trial_log_path  = sess_info.get('trial_log')
            capacitive_path = sess_info.get('capacitive')
            treadmill_path  = sess_info.get('treadmill')

            if not isinstance(trial_log_path, str) or pd.isna(trial_log_path):
                continue
            try:
                tl = pd.read_csv(trial_log_path)
            except Exception as e:
                print(f"  [WARN] {animal_id} session {session_num}: cannot read trial_log — {e}")
                continue
            if 'reward_event' not in tl.columns or 'texture_change_time' not in tl.columns:
                continue

            # Session end time and lick event detection from capacitive data -----
            lick_event_times = np.array([])
            session_end_time = None
            if isinstance(capacitive_path, str) and not pd.isna(capacitive_path):
                try:
                    cap = pd.read_csv(capacitive_path)
                    if 'elapsed_time' in cap.columns:
                        session_end_time = float(cap['elapsed_time'].max())
                    if 'capacitive_value' in cap.columns:
                        cap_df = cap.copy()
                        cap_df['Time_sec'] = cap_df['elapsed_time']
                        kde_val = get_cached_kde(capacitive_path)
                        if kde_val is None:
                            kde_val = lda.compute_KDE(cap_df, 'capacitive_value')
                            cache_kde_value(capacitive_path, kde_val)
                        cap_df = lda.compute_KDE_normalizations(cap_df, 'capacitive_value', kde_val)
                        events_df, _ = lda.detect_events_above_threshold(
                            cap_df, 'capacitive_value', threshold=None)
                        mask_evt = events_df['capacitive_value_event'] == 1
                        lick_event_times = events_df.loc[mask_evt, 'Time_sec'].values.astype(float)
                except Exception:
                    pass
            if session_end_time is None:
                tc_vals = tl['texture_change_time'].dropna()
                session_end_time = float(tc_vals.max()) if len(tc_vals) else None

            # Load treadmill speed data for this session -----------------------
            treadmill_df = None
            if isinstance(treadmill_path, str) and not pd.isna(treadmill_path):
                try:
                    tm = pd.read_csv(treadmill_path)
                    if 'global_time' in tm.columns and 'speed' in tm.columns:
                        load_cols = ['global_time', 'speed']
                        if 'distance' in tm.columns:
                            load_cols.append('distance')
                        treadmill_df = tm[load_cols].dropna(subset=['global_time', 'speed'])
                        # Apply Butterworth low-pass (0.25 Hz, order 3) to full session speed
                        if len(treadmill_df) >= 15:
                            _fs_lvl = 1.0 / treadmill_df['global_time'].diff().median()
                            _b_lvl, _a_lvl = butter(3, 0.25 / (_fs_lvl / 2.0), btype='low')
                            treadmill_df = treadmill_df.copy()
                            treadmill_df['speed'] = filtfilt(_b_lvl, _a_lvl, treadmill_df['speed'].values)
                except Exception:
                    pass

            levels   = session_group['level'].tolist()
            trans_ts = session_group['transition_ts'].tolist()

            # start_ts for each level = end_ts of the previous level (or 0)
            start_times = [0.0]
            for ts in trans_ts[:-1]:
                start_times.append(float(ts) if pd.notna(ts) else float('inf'))

            # end_ts for each level = its own transition_ts, or session end time
            end_times = []
            for ts in trans_ts:
                if pd.notna(ts):
                    end_times.append(float(ts))
                elif session_end_time is not None:
                    end_times.append(session_end_time)
                else:
                    end_times.append(None)

            reward_events = tl['reward_event'].dropna().to_numpy(dtype=float)
            hits_events   = (tl['hits_event'].dropna().to_numpy(dtype=float)
                             if 'hits_event' in tl.columns else np.array([]))

            for i, level in enumerate(levels):
                start_t = start_times[i]
                end_t   = end_times[i]
                if end_t is None or start_t >= end_t:
                    continue
                duration_min = (end_t - start_t) / 60.0
                if duration_min <= 0:
                    continue
                count = int(np.sum((reward_events >= start_t) & (reward_events < end_t)))
                count += int(np.sum((hits_events   >= start_t) & (hits_events   < end_t)))

                condition = (animal_conditions or {}).get(animal_id, 'Unknown')
                key = (animal_id, level)
                if key not in animal_level_accum:
                    animal_level_accum[key] = {'rewards': 0, 'duration_min': 0.0,
                                               'condition': condition,
                                               'speed_sum': 0.0, 'speed_count': 0,
                                               'lick_count': 0, 'dist_sum': 0.0,
                                               'bout_count_lvl': 0,
                                               'bout_spd_sum': 0.0, 'bout_spd_cnt': 0,
                                               'bout_dist_sum': 0.0, 'bout_dist_cnt': 0}
                animal_level_accum[key]['rewards']      += count
                animal_level_accum[key]['duration_min'] += duration_min
                if treadmill_df is not None and len(treadmill_df) > 0:
                    mask = (treadmill_df['global_time'] >= start_t) & (treadmill_df['global_time'] < end_t)
                    lvl_speeds = treadmill_df.loc[mask, 'speed'].values / 10.0
                    lvl_times  = treadmill_df.loc[mask, 'global_time'].values
                    animal_level_accum[key]['speed_sum']   += float(np.sum(lvl_speeds))
                    animal_level_accum[key]['speed_count'] += len(lvl_speeds)
                    if len(lvl_times) >= 2:
                        _bouts = detect_locomotion_bouts(lvl_times, lvl_speeds)
                        animal_level_accum[key]['bout_count_lvl'] += len(_bouts)
                        _lvl_dist_arr = (
                            pd.to_numeric(treadmill_df.loc[mask, 'distance'], errors='coerce').values
                            if 'distance' in treadmill_df.columns else None
                        )
                        for _t0, _t1 in _bouts:
                            _bmask = (lvl_times >= _t0) & (lvl_times <= _t1)
                            if np.any(_bmask):
                                animal_level_accum[key]['bout_spd_sum'] += float(np.mean(lvl_speeds[_bmask]))
                                animal_level_accum[key]['bout_spd_cnt'] += 1
                            if _lvl_dist_arr is not None:
                                _dv = _lvl_dist_arr[_bmask]
                                _dv = _dv[~np.isnan(_dv)]
                                if len(_dv) >= 2:
                                    animal_level_accum[key]['bout_dist_sum'] += float(_dv[-1] - _dv[0])
                                    animal_level_accum[key]['bout_dist_cnt'] += 1
                if len(lick_event_times) > 0:
                    lick_mask = (lick_event_times >= start_t) & (lick_event_times < end_t)
                    animal_level_accum[key]['lick_count'] += int(np.sum(lick_mask))
                if treadmill_df is not None and 'distance' in treadmill_df.columns:
                    dist_col = pd.to_numeric(treadmill_df['distance'], errors='coerce')
                    dist_window = dist_col[
                        (treadmill_df['global_time'] >= start_t) &
                        (treadmill_df['global_time'] < end_t)
                    ].dropna()
                    if len(dist_window) >= 2:
                        animal_level_accum[key]['dist_sum'] += float(
                            dist_window.iloc[-1] - dist_window.iloc[0])
            # Record the last level of this session; final session's value = the incomplete level
            if levels:
                animal_last_level[animal_id] = levels[-1]

    # Collapse accumulators: one rpm, mean speed, and mean lick rate per (animal, level)
    # condition_level_data[condition][level] = [rpm_animal1, rpm_animal2, ...]
    condition_level_data: dict[str, dict[str, list[float]]] = {}
    condition_level_speed: dict[str, dict[str, list[float]]] = {}
    collapsed_level_speed: dict[str, list[float]] = {}
    condition_level_lick: dict[str, dict[str, list[float]]] = {}
    collapsed_level_lick: dict[str, list[float]] = {}
    condition_level_dist: dict[str, dict[str, list[float]]] = {}
    collapsed_level_dist: dict[str, list[float]] = {}
    condition_level_bout: dict[str, dict[str, list[float]]] = {}
    collapsed_level_bout: dict[str, list[float]] = {}
    condition_level_bout_avg_spd: dict[str, dict[str, list[float]]] = {}
    collapsed_level_bout_avg_spd: dict[str, list[float]] = {}
    condition_level_bout_avg_dist_lvl: dict[str, dict[str, list[float]]] = {}
    collapsed_level_bout_avg_dist_lvl: dict[str, list[float]] = {}
    for (animal_id, level), accum in animal_level_accum.items():
        condition = accum['condition']
        if accum['duration_min'] > 0:
            rpm = accum['rewards'] / accum['duration_min']
            condition_level_data.setdefault(condition, {}).setdefault(level, []).append(rpm)
        if accum['speed_count'] > 0:
            mean_spd = accum['speed_sum'] / accum['speed_count']
            condition_level_speed.setdefault(condition, {}).setdefault(level, []).append(mean_spd)
            collapsed_level_speed.setdefault(level, []).append(mean_spd)
            # bout count shares the same treadmill-data gate as speed
            bc = accum.get('bout_count_lvl', 0)
            condition_level_bout.setdefault(condition, {}).setdefault(level, []).append(float(bc))
            collapsed_level_bout.setdefault(level, []).append(float(bc))
        if accum.get('bout_spd_cnt', 0) > 0:
            avg_spd_bout = accum['bout_spd_sum'] / accum['bout_spd_cnt']
            condition_level_bout_avg_spd.setdefault(condition, {}).setdefault(level, []).append(avg_spd_bout)
            collapsed_level_bout_avg_spd.setdefault(level, []).append(avg_spd_bout)
        if accum.get('bout_dist_cnt', 0) > 0:
            avg_dist_bout_m = accum['bout_dist_sum'] / accum['bout_dist_cnt'] / 1000.0  # mm → m
            condition_level_bout_avg_dist_lvl.setdefault(condition, {}).setdefault(level, []).append(avg_dist_bout_m)
            collapsed_level_bout_avg_dist_lvl.setdefault(level, []).append(avg_dist_bout_m)
        if accum['lick_count'] > 0 and accum['duration_min'] > 0:
            lpm = accum['lick_count'] / accum['duration_min']
            condition_level_lick.setdefault(condition, {}).setdefault(level, []).append(lpm)
            collapsed_level_lick.setdefault(level, []).append(lpm)
        if accum.get('dist_sum', 0.0) > 0:
            dist_m = accum['dist_sum'] / 1000.0  # mm → m
            condition_level_dist.setdefault(condition, {}).setdefault(level, []).append(dist_m)
            collapsed_level_dist.setdefault(level, []).append(dist_m)

    # Per-condition distance excluding each mouse's last (potentially incomplete) level
    condition_level_dist_excl_last: dict[str, dict[str, list[float]]] = {}
    for (animal_id, level), accum in animal_level_accum.items():
        if level == animal_last_level.get(animal_id):
            continue  # skip last level — final session may be incomplete
        if accum.get('dist_sum', 0.0) > 0:
            condition = accum['condition']
            dist_m = accum['dist_sum'] / 1000.0  # mm → m
            condition_level_dist_excl_last.setdefault(condition, {}).setdefault(level, []).append(dist_m)

    # Sort levels ----------------------------------------------------------------
    def sort_key(x):
        if x.startswith('level_'):
            try:
                return (0, int(x.split('_')[1].split('.')[0]))
            except (ValueError, IndexError):
                pass
        return (1, x)

    all_levels = sorted(
        {lv for cond_data in condition_level_data.values() for lv in cond_data},
        key=sort_key,
    )
    conditions_sorted = sorted(condition_level_data.keys())
    n_conditions = len(conditions_sorted)
    n_levels     = len(all_levels)

    # Assign one color per condition using the existing generate_colors helper
    cond_colors = generate_colors(max(n_conditions, 1))
    cond_color_map = {c: cond_colors[i] for i, c in enumerate(conditions_sorted)}

    # Grouped bar chart ----------------------------------------------------------
    level_reward_fig, ax = plt.subplots(figsize=(max(15, n_levels * 0.8), 8))

    bar_width   = 0.8 / max(n_conditions, 1)
    x_positions = np.arange(n_levels)

    for cond_idx, condition in enumerate(conditions_sorted):
        cond_data = condition_level_data[condition]
        means = []
        sems  = []
        ns    = []
        for level in all_levels:
            vals = cond_data.get(level, [])
            if vals:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals) / np.sqrt(len(vals))))
                ns.append(len(vals))
            else:
                means.append(np.nan)
                sems.append(np.nan)
                ns.append(0)

        offset = (cond_idx - (n_conditions - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset, means,
            width=bar_width, yerr=sems,
            capsize=4,
            color=cond_color_map[condition],
            label=condition,
            error_kw={'elinewidth': 1.2},
        )

        # n= annotations above each bar
        for xi, (m, n_val) in enumerate(zip(means, ns)):
            if n_val > 0 and not np.isnan(m):
                ax.text(
                    x_positions[xi] + offset,
                    m + (sems[xi] if not np.isnan(sems[xi]) else 0),
                    f'n={n_val}',
                    ha='center', va='bottom',
                    fontsize=7,
                    color=cond_color_map[condition],
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels],
        rotation=45, ha='right',
    )
    ax.set_title('Average Rewards Per Minute by Level — by Starting Condition')
    ax.set_xlabel('Level')
    ax.set_ylabel('Rewards per Minute (Mean ± SEM)')
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Starting Condition')
    level_reward_fig.tight_layout()

    if selected_plots is None:
        selected_plots = set(_ALL_PLOT_KEYS)

    # ── Collapsed speed by level (all mice) ───────────────────────────────────
    level_speed_collapsed_fig = None
    if 'level_speed' in selected_plots and collapsed_level_speed:
        all_levels_spd = sorted(collapsed_level_speed.keys(), key=sort_key)
        level_speed_collapsed_fig, ax_sc = plt.subplots(
            figsize=(max(15, len(all_levels_spd) * 0.8), 8))
        x_sc = np.arange(len(all_levels_spd))
        sc_means, sc_sems, sc_ns = [], [], []
        for lv in all_levels_spd:
            vals = collapsed_level_speed[lv]
            sc_means.append(float(np.mean(vals)))
            sc_sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            sc_ns.append(len(vals))
        ax_sc.bar(x_sc, sc_means, yerr=sc_sems, capsize=4, color='steelblue',
                  error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(sc_means, sc_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_sc.text(xi, m + (sc_sems[xi] if not np.isnan(sc_sems[xi]) else 0),
                           f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_sc.set_xticks(x_sc)
        ax_sc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_spd],
            rotation=45, ha='right',
        )
        ax_sc.set_title('Average Speed by Level — All Mice')
        ax_sc.set_xlabel('Level')
        ax_sc.set_ylabel('Average Speed cm/s (Mean ± SEM)')
        ax_sc.set_ylim(bottom=0)
        ax_sc.tick_params(axis='both', direction='in')
        ax_sc.spines['top'].set_visible(False)
        ax_sc.spines['right'].set_visible(False)
        level_speed_collapsed_fig.tight_layout()

    # ── Speed by level split by condition ────────────────────────────────────
    level_speed_condition_fig = None
    if 'level_speed_condition' in selected_plots and condition_level_speed:
        all_levels_sc = sorted(
            {lv for cd in condition_level_speed.values() for lv in cd},
            key=sort_key,
        )
        conds_sc = sorted(condition_level_speed.keys())
        n_conds_sc = len(conds_sc)
        level_speed_condition_fig, ax_scc = plt.subplots(
            figsize=(max(15, len(all_levels_sc) * 0.8), 8))
        bw_sc = 0.8 / max(n_conds_sc, 1)
        x_scc = np.arange(len(all_levels_sc))
        for ci, cond in enumerate(conds_sc):
            cdata = condition_level_speed[cond]
            means_c, sems_c, ns_c = [], [], []
            for lv in all_levels_sc:
                vals = cdata.get(lv, [])
                if vals:
                    means_c.append(float(np.mean(vals)))
                    sems_c.append(float(np.std(vals) / np.sqrt(len(vals))))
                    ns_c.append(len(vals))
                else:
                    means_c.append(np.nan)
                    sems_c.append(np.nan)
                    ns_c.append(0)
            offset_c = (ci - (n_conds_sc - 1) / 2) * bw_sc
            ax_scc.bar(
                x_scc + offset_c, means_c,
                width=bw_sc, yerr=sems_c, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_c, ns_c)):
                if n_val > 0 and not np.isnan(m):
                    ax_scc.text(
                        x_scc[xi] + offset_c,
                        m + (sems_c[xi] if not np.isnan(sems_c[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_scc.set_xticks(x_scc)
        ax_scc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_sc],
            rotation=45, ha='right',
        )
        ax_scc.set_title('Average Speed by Level — by Starting Condition')
        ax_scc.set_xlabel('Level')
        ax_scc.set_ylabel('Average Speed cm/s (Mean ± SEM)')
        ax_scc.set_ylim(bottom=0)
        ax_scc.tick_params(axis='both', direction='in')
        ax_scc.spines['top'].set_visible(False)
        ax_scc.spines['right'].set_visible(False)
        ax_scc.legend(title='Starting Condition')
        level_speed_condition_fig.tight_layout()

    # ── Collapsed lick rate by level (all mice) ───────────────────────────────
    level_lick_collapsed_fig = None
    if 'level_lick' in selected_plots and collapsed_level_lick:
        all_levels_lk = sorted(collapsed_level_lick.keys(), key=sort_key)
        level_lick_collapsed_fig, ax_lk = plt.subplots(
            figsize=(max(15, len(all_levels_lk) * 0.8), 8))
        x_lk = np.arange(len(all_levels_lk))
        lk_means, lk_sems, lk_ns = [], [], []
        for lv in all_levels_lk:
            vals = collapsed_level_lick[lv]
            lk_means.append(float(np.mean(vals)))
            lk_sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            lk_ns.append(len(vals))
        ax_lk.bar(x_lk, lk_means, yerr=lk_sems, capsize=4, color='steelblue',
                  error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(lk_means, lk_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_lk.text(xi, m + (lk_sems[xi] if not np.isnan(lk_sems[xi]) else 0),
                           f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_lk.set_xticks(x_lk)
        ax_lk.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_lk],
            rotation=45, ha='right',
        )
        ax_lk.set_title('Average Lick Rate by Level — All Mice')
        ax_lk.set_xlabel('Level')
        ax_lk.set_ylabel('Licks per Minute (Mean ± SEM)')
        ax_lk.set_ylim(bottom=0)
        ax_lk.tick_params(axis='both', direction='in')
        ax_lk.spines['top'].set_visible(False)
        ax_lk.spines['right'].set_visible(False)
        level_lick_collapsed_fig.tight_layout()

    # ── Lick rate by level split by condition ─────────────────────────────────
    level_lick_condition_fig = None
    if 'level_lick_condition' in selected_plots and condition_level_lick:
        all_levels_lkc = sorted(
            {lv for cd in condition_level_lick.values() for lv in cd},
            key=sort_key,
        )
        conds_lkc = sorted(condition_level_lick.keys())
        n_conds_lkc = len(conds_lkc)
        level_lick_condition_fig, ax_lkc = plt.subplots(
            figsize=(max(15, len(all_levels_lkc) * 0.8), 8))
        bw_lkc = 0.8 / max(n_conds_lkc, 1)
        x_lkc = np.arange(len(all_levels_lkc))
        for ci, cond in enumerate(conds_lkc):
            cdata = condition_level_lick[cond]
            means_lkc, sems_lkc, ns_lkc = [], [], []
            for lv in all_levels_lkc:
                vals = cdata.get(lv, [])
                if vals:
                    means_lkc.append(float(np.mean(vals)))
                    sems_lkc.append(float(np.std(vals) / np.sqrt(len(vals))))
                    ns_lkc.append(len(vals))
                else:
                    means_lkc.append(np.nan)
                    sems_lkc.append(np.nan)
                    ns_lkc.append(0)
            offset_lkc = (ci - (n_conds_lkc - 1) / 2) * bw_lkc
            ax_lkc.bar(
                x_lkc + offset_lkc, means_lkc,
                width=bw_lkc, yerr=sems_lkc, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_lkc, ns_lkc)):
                if n_val > 0 and not np.isnan(m):
                    ax_lkc.text(
                        x_lkc[xi] + offset_lkc,
                        m + (sems_lkc[xi] if not np.isnan(sems_lkc[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_lkc.set_xticks(x_lkc)
        ax_lkc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_lkc],
            rotation=45, ha='right',
        )
        ax_lkc.set_title('Average Lick Rate by Level — by Starting Condition')
        ax_lkc.set_xlabel('Level')
        ax_lkc.set_ylabel('Licks per Minute (Mean ± SEM)')
        ax_lkc.set_ylim(bottom=0)
        ax_lkc.tick_params(axis='both', direction='in')
        ax_lkc.spines['top'].set_visible(False)
        ax_lkc.spines['right'].set_visible(False)
        ax_lkc.legend(title='Starting Condition')
        level_lick_condition_fig.tight_layout()

    # ── Collapsed distance by level (all mice) ───────────────────────────────
    level_dist_collapsed_fig = None
    if 'level_dist' in selected_plots and collapsed_level_dist:
        all_levels_dc = sorted(collapsed_level_dist.keys(), key=sort_key)
        level_dist_collapsed_fig, ax_dc = plt.subplots(
            figsize=(max(15, len(all_levels_dc) * 0.8), 8))
        x_dc = np.arange(len(all_levels_dc))
        dc_means, dc_sems, dc_ns = [], [], []
        for lv in all_levels_dc:
            vals = collapsed_level_dist[lv]
            dc_means.append(float(np.mean(vals)))
            dc_sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            dc_ns.append(len(vals))
        ax_dc.bar(x_dc, dc_means, yerr=dc_sems, capsize=4, color='steelblue',
                  error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(dc_means, dc_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_dc.text(xi, m + (dc_sems[xi] if not np.isnan(dc_sems[xi]) else 0),
                           f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_dc.set_xticks(x_dc)
        ax_dc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_dc],
            rotation=45, ha='right',
        )
        ax_dc.set_title('Distance Traveled by Level \u2014 All Mice')
        ax_dc.set_xlabel('Level')
        ax_dc.set_ylabel('Distance (m, Mean \u00b1 SEM)')
        ax_dc.set_ylim(bottom=0)
        ax_dc.tick_params(axis='both', direction='in')
        ax_dc.spines['top'].set_visible(False)
        ax_dc.spines['right'].set_visible(False)
        level_dist_collapsed_fig.tight_layout()

    # ── Distance by level split by condition ─────────────────────────────────
    level_dist_condition_fig = None
    if 'level_dist_condition' in selected_plots and condition_level_dist:
        all_levels_dcc = sorted(
            {lv for cd in condition_level_dist.values() for lv in cd},
            key=sort_key,
        )
        conds_dcc = sorted(condition_level_dist.keys())
        n_conds_dcc = len(conds_dcc)
        level_dist_condition_fig, ax_dcc = plt.subplots(
            figsize=(max(15, len(all_levels_dcc) * 0.8), 8))
        bw_dcc = 0.8 / max(n_conds_dcc, 1)
        x_dcc = np.arange(len(all_levels_dcc))
        for ci, cond in enumerate(conds_dcc):
            cdata = condition_level_dist[cond]
            means_dcc, sems_dcc, ns_dcc = [], [], []
            for lv in all_levels_dcc:
                vals = cdata.get(lv, [])
                if vals:
                    means_dcc.append(float(np.mean(vals)))
                    sems_dcc.append(float(np.std(vals) / np.sqrt(len(vals))))
                    ns_dcc.append(len(vals))
                else:
                    means_dcc.append(np.nan)
                    sems_dcc.append(np.nan)
                    ns_dcc.append(0)
            offset_dcc = (ci - (n_conds_dcc - 1) / 2) * bw_dcc
            ax_dcc.bar(
                x_dcc + offset_dcc, means_dcc,
                width=bw_dcc, yerr=sems_dcc, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_dcc, ns_dcc)):
                if n_val > 0 and not np.isnan(m):
                    ax_dcc.text(
                        x_dcc[xi] + offset_dcc,
                        m + (sems_dcc[xi] if not np.isnan(sems_dcc[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_dcc.set_xticks(x_dcc)
        ax_dcc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_dcc],
            rotation=45, ha='right',
        )
        ax_dcc.set_title('Distance Traveled by Level \u2014 by Starting Condition')
        ax_dcc.set_xlabel('Level')
        ax_dcc.set_ylabel('Distance (m, Mean \u00b1 SEM)')
        ax_dcc.set_ylim(bottom=0)
        ax_dcc.tick_params(axis='both', direction='in')
        ax_dcc.spines['top'].set_visible(False)
        ax_dcc.spines['right'].set_visible(False)
        ax_dcc.legend(title='Starting Condition')
        level_dist_condition_fig.tight_layout()

    # ── Distance by level split by condition (each mouse's last level excluded) ─
    level_dist_condition_excl_last_fig = None
    if 'level_dist_condition_excl_last' in selected_plots and condition_level_dist_excl_last:
        all_levels_dcx = sorted(
            {lv for cd in condition_level_dist_excl_last.values() for lv in cd},
            key=sort_key,
        )
        conds_dcx = sorted(condition_level_dist_excl_last.keys())
        n_conds_dcx = len(conds_dcx)
        level_dist_condition_excl_last_fig, ax_dcx = plt.subplots(
            figsize=(max(15, len(all_levels_dcx) * 0.8), 8))
        bw_dcx = 0.8 / max(n_conds_dcx, 1)
        x_dcx = np.arange(len(all_levels_dcx))
        for ci, cond in enumerate(conds_dcx):
            cdata = condition_level_dist_excl_last[cond]
            means_dcx, sems_dcx, ns_dcx = [], [], []
            for lv in all_levels_dcx:
                vals = cdata.get(lv, [])
                if vals:
                    means_dcx.append(float(np.mean(vals)))
                    sems_dcx.append(float(np.std(vals) / np.sqrt(len(vals))))
                    ns_dcx.append(len(vals))
                else:
                    means_dcx.append(np.nan)
                    sems_dcx.append(np.nan)
                    ns_dcx.append(0)
            offset_dcx = (ci - (n_conds_dcx - 1) / 2) * bw_dcx
            ax_dcx.bar(
                x_dcx + offset_dcx, means_dcx,
                width=bw_dcx, yerr=sems_dcx, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_dcx, ns_dcx)):
                if n_val > 0 and not np.isnan(m):
                    ax_dcx.text(
                        x_dcx[xi] + offset_dcx,
                        m + (sems_dcx[xi] if not np.isnan(sems_dcx[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_dcx.set_xticks(x_dcx)
        ax_dcx.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_dcx],
            rotation=45, ha='right',
        )
        ax_dcx.set_title('Distance Traveled by Level \u2014 by Starting Condition (last level excluded)')
        ax_dcx.set_xlabel('Level')
        ax_dcx.set_ylabel('Distance (m, Mean \u00b1 SEM)')
        ax_dcx.set_ylim(bottom=0)
        ax_dcx.tick_params(axis='both', direction='in')
        ax_dcx.spines['top'].set_visible(False)
        ax_dcx.spines['right'].set_visible(False)
        ax_dcx.legend(title='Starting Condition')
        level_dist_condition_excl_last_fig.tight_layout()

    # ── Collapsed bout count by level (all mice) ─────────────────────────────
    level_bout_collapsed_fig = None
    if 'level_bout' in selected_plots and collapsed_level_bout:
        all_levels_bc = sorted(collapsed_level_bout.keys(), key=sort_key)
        level_bout_collapsed_fig, ax_bc = plt.subplots(
            figsize=(max(15, len(all_levels_bc) * 0.8), 8))
        x_bc = np.arange(len(all_levels_bc))
        bc_means, bc_sems, bc_ns = [], [], []
        for lv in all_levels_bc:
            vals = collapsed_level_bout[lv]
            bc_means.append(float(np.mean(vals)))
            bc_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            bc_ns.append(len(vals))
        ax_bc.bar(x_bc, bc_means, yerr=bc_sems, capsize=4, color='steelblue',
                  error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(bc_means, bc_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_bc.text(xi, m + (bc_sems[xi] if not np.isnan(bc_sems[xi]) else 0),
                           f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_bc.set_xticks(x_bc)
        ax_bc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_bc],
            rotation=45, ha='right',
        )
        ax_bc.set_title('Locomotion Bout Count by Level \u2014 All Mice')
        ax_bc.set_xlabel('Level')
        ax_bc.set_ylabel('Bout Count (Mean \u00b1 SEM)')
        ax_bc.set_ylim(bottom=0)
        ax_bc.tick_params(axis='both', direction='in')
        ax_bc.spines['top'].set_visible(False)
        ax_bc.spines['right'].set_visible(False)
        level_bout_collapsed_fig.tight_layout()

    # ── Bout count by level split by condition ────────────────────────────────
    level_bout_condition_fig = None
    if 'level_bout_condition' in selected_plots and condition_level_bout:
        all_levels_bcc = sorted(
            {lv for cd in condition_level_bout.values() for lv in cd},
            key=sort_key,
        )
        conds_bcc = sorted(condition_level_bout.keys())
        n_conds_bcc = len(conds_bcc)
        level_bout_condition_fig, ax_bcc = plt.subplots(
            figsize=(max(15, len(all_levels_bcc) * 0.8), 8))
        bw_bcc = 0.8 / max(n_conds_bcc, 1)
        x_bcc = np.arange(len(all_levels_bcc))
        for ci, cond in enumerate(conds_bcc):
            cdata = condition_level_bout[cond]
            means_bcc, sems_bcc, ns_bcc = [], [], []
            for lv in all_levels_bcc:
                vals = cdata.get(lv, [])
                if vals:
                    means_bcc.append(float(np.mean(vals)))
                    sems_bcc.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                    ns_bcc.append(len(vals))
                else:
                    means_bcc.append(np.nan)
                    sems_bcc.append(np.nan)
                    ns_bcc.append(0)
            offset_bcc = (ci - (n_conds_bcc - 1) / 2) * bw_bcc
            ax_bcc.bar(
                x_bcc + offset_bcc, means_bcc,
                width=bw_bcc, yerr=sems_bcc, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_bcc, ns_bcc)):
                if n_val > 0 and not np.isnan(m):
                    ax_bcc.text(
                        x_bcc[xi] + offset_bcc,
                        m + (sems_bcc[xi] if not np.isnan(sems_bcc[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_bcc.set_xticks(x_bcc)
        ax_bcc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_bcc],
            rotation=45, ha='right',
        )
        ax_bcc.set_title('Locomotion Bout Count by Level \u2014 by Starting Condition')
        ax_bcc.set_xlabel('Level')
        ax_bcc.set_ylabel('Bout Count (Mean \u00b1 SEM)')
        ax_bcc.set_ylim(bottom=0)
        ax_bcc.tick_params(axis='both', direction='in')
        ax_bcc.spines['top'].set_visible(False)
        ax_bcc.spines['right'].set_visible(False)
        ax_bcc.legend(title='Starting Condition')
        level_bout_condition_fig.tight_layout()

    # ── Collapsed avg speed per bout by level (all mice) ─────────────────────
    level_bout_avg_speed_collapsed_fig = None
    if 'level_bout_avg_speed' in selected_plots and collapsed_level_bout_avg_spd:
        all_levels_bas = sorted(collapsed_level_bout_avg_spd.keys(), key=sort_key)
        level_bout_avg_speed_collapsed_fig, ax_bas = plt.subplots(
            figsize=(max(15, len(all_levels_bas) * 0.8), 8))
        x_bas = np.arange(len(all_levels_bas))
        bas_means, bas_sems, bas_ns = [], [], []
        for lv in all_levels_bas:
            vals = collapsed_level_bout_avg_spd[lv]
            bas_means.append(float(np.mean(vals)))
            bas_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            bas_ns.append(len(vals))
        ax_bas.bar(x_bas, bas_means, yerr=bas_sems, capsize=4, color='steelblue',
                   error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(bas_means, bas_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_bas.text(xi, m + (bas_sems[xi] if not np.isnan(bas_sems[xi]) else 0),
                            f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_bas.set_xticks(x_bas)
        ax_bas.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_bas],
            rotation=45, ha='right',
        )
        ax_bas.set_title('Average Speed per Locomotion Bout by Level \u2014 All Mice')
        ax_bas.set_xlabel('Level')
        ax_bas.set_ylabel('Speed per Bout (cm/s, Mean \u00b1 SEM)')
        ax_bas.set_ylim(bottom=0)
        ax_bas.tick_params(axis='both', direction='in')
        ax_bas.spines['top'].set_visible(False)
        ax_bas.spines['right'].set_visible(False)
        level_bout_avg_speed_collapsed_fig.tight_layout()

    # ── Avg speed per bout by level split by condition ────────────────────────
    level_bout_avg_speed_condition_fig = None
    if 'level_bout_avg_speed_condition' in selected_plots and condition_level_bout_avg_spd:
        all_levels_basc = sorted(
            {lv for cd in condition_level_bout_avg_spd.values() for lv in cd},
            key=sort_key,
        )
        conds_basc = sorted(condition_level_bout_avg_spd.keys())
        n_conds_basc = len(conds_basc)
        level_bout_avg_speed_condition_fig, ax_basc = plt.subplots(
            figsize=(max(15, len(all_levels_basc) * 0.8), 8))
        bw_basc = 0.8 / max(n_conds_basc, 1)
        x_basc = np.arange(len(all_levels_basc))
        for ci, cond in enumerate(conds_basc):
            cdata = condition_level_bout_avg_spd[cond]
            means_basc, sems_basc, ns_basc = [], [], []
            for lv in all_levels_basc:
                vals = cdata.get(lv, [])
                if vals:
                    means_basc.append(float(np.mean(vals)))
                    sems_basc.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                    ns_basc.append(len(vals))
                else:
                    means_basc.append(np.nan)
                    sems_basc.append(np.nan)
                    ns_basc.append(0)
            offset_basc = (ci - (n_conds_basc - 1) / 2) * bw_basc
            ax_basc.bar(
                x_basc + offset_basc, means_basc,
                width=bw_basc, yerr=sems_basc, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_basc, ns_basc)):
                if n_val > 0 and not np.isnan(m):
                    ax_basc.text(
                        x_basc[xi] + offset_basc,
                        m + (sems_basc[xi] if not np.isnan(sems_basc[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_basc.set_xticks(x_basc)
        ax_basc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_basc],
            rotation=45, ha='right',
        )
        ax_basc.set_title('Average Speed per Locomotion Bout by Level \u2014 by Starting Condition')
        ax_basc.set_xlabel('Level')
        ax_basc.set_ylabel('Speed per Bout (cm/s, Mean \u00b1 SEM)')
        ax_basc.set_ylim(bottom=0)
        ax_basc.tick_params(axis='both', direction='in')
        ax_basc.spines['top'].set_visible(False)
        ax_basc.spines['right'].set_visible(False)
        ax_basc.legend(title='Starting Condition')
        level_bout_avg_speed_condition_fig.tight_layout()

    # ── Collapsed avg distance per bout by level (all mice) ───────────────────
    level_bout_avg_dist_collapsed_fig = None
    if 'level_bout_avg_dist' in selected_plots and collapsed_level_bout_avg_dist_lvl:
        all_levels_bad = sorted(collapsed_level_bout_avg_dist_lvl.keys(), key=sort_key)
        level_bout_avg_dist_collapsed_fig, ax_bad = plt.subplots(
            figsize=(max(15, len(all_levels_bad) * 0.8), 8))
        x_bad = np.arange(len(all_levels_bad))
        bad_means, bad_sems, bad_ns = [], [], []
        for lv in all_levels_bad:
            vals = collapsed_level_bout_avg_dist_lvl[lv]
            bad_means.append(float(np.mean(vals)))
            bad_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            bad_ns.append(len(vals))
        ax_bad.bar(x_bad, bad_means, yerr=bad_sems, capsize=4, color='steelblue',
                   error_kw={'elinewidth': 1.2})
        for xi, (m, n_val) in enumerate(zip(bad_means, bad_ns)):
            if n_val > 0 and not np.isnan(m):
                ax_bad.text(xi, m + (bad_sems[xi] if not np.isnan(bad_sems[xi]) else 0),
                            f'n={n_val}', ha='center', va='bottom', fontsize=7)
        ax_bad.set_xticks(x_bad)
        ax_bad.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_bad],
            rotation=45, ha='right',
        )
        ax_bad.set_title('Average Distance per Locomotion Bout by Level \u2014 All Mice')
        ax_bad.set_xlabel('Level')
        ax_bad.set_ylabel('Distance per Bout (m, Mean \u00b1 SEM)')
        ax_bad.set_ylim(bottom=0)
        ax_bad.tick_params(axis='both', direction='in')
        ax_bad.spines['top'].set_visible(False)
        ax_bad.spines['right'].set_visible(False)
        level_bout_avg_dist_collapsed_fig.tight_layout()

    # ── Avg distance per bout by level split by condition ─────────────────────
    level_bout_avg_dist_condition_fig = None
    if 'level_bout_avg_dist_condition' in selected_plots and condition_level_bout_avg_dist_lvl:
        all_levels_badc = sorted(
            {lv for cd in condition_level_bout_avg_dist_lvl.values() for lv in cd},
            key=sort_key,
        )
        conds_badc = sorted(condition_level_bout_avg_dist_lvl.keys())
        n_conds_badc = len(conds_badc)
        level_bout_avg_dist_condition_fig, ax_badc = plt.subplots(
            figsize=(max(15, len(all_levels_badc) * 0.8), 8))
        bw_badc = 0.8 / max(n_conds_badc, 1)
        x_badc = np.arange(len(all_levels_badc))
        for ci, cond in enumerate(conds_badc):
            cdata = condition_level_bout_avg_dist_lvl[cond]
            means_badc, sems_badc, ns_badc = [], [], []
            for lv in all_levels_badc:
                vals = cdata.get(lv, [])
                if vals:
                    means_badc.append(float(np.mean(vals)))
                    sems_badc.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                    ns_badc.append(len(vals))
                else:
                    means_badc.append(np.nan)
                    sems_badc.append(np.nan)
                    ns_badc.append(0)
            offset_badc = (ci - (n_conds_badc - 1) / 2) * bw_badc
            ax_badc.bar(
                x_badc + offset_badc, means_badc,
                width=bw_badc, yerr=sems_badc, capsize=4,
                color=cond_color_map.get(cond, 'gray'),
                label=cond,
                error_kw={'elinewidth': 1.2},
            )
            for xi, (m, n_val) in enumerate(zip(means_badc, ns_badc)):
                if n_val > 0 and not np.isnan(m):
                    ax_badc.text(
                        x_badc[xi] + offset_badc,
                        m + (sems_badc[xi] if not np.isnan(sems_badc[xi]) else 0),
                        f'n={n_val}', ha='center', va='bottom', fontsize=7,
                        color=cond_color_map.get(cond, 'gray'),
                    )
        ax_badc.set_xticks(x_badc)
        ax_badc.set_xticklabels(
            [lv.replace('level_', 'L').replace('.json', '') for lv in all_levels_badc],
            rotation=45, ha='right',
        )
        ax_badc.set_title('Average Distance per Locomotion Bout by Level \u2014 by Starting Condition')
        ax_badc.set_xlabel('Level')
        ax_badc.set_ylabel('Distance per Bout (m, Mean \u00b1 SEM)')
        ax_badc.set_ylim(bottom=0)
        ax_badc.tick_params(axis='both', direction='in')
        ax_badc.spines['top'].set_visible(False)
        ax_badc.spines['right'].set_visible(False)
        ax_badc.legend(title='Starting Condition')
        level_bout_avg_dist_condition_fig.tight_layout()

    # ── Package level data for the descriptive stats report ──────────────────
    _level_stats = {
        'condition_level_reward_rate':   condition_level_data,
        'condition_level_speed':         condition_level_speed,
        'condition_level_lick':          condition_level_lick,
        'condition_level_dist':          condition_level_dist,
        'condition_level_bout':          condition_level_bout,
        'condition_level_bout_avg_spd':  condition_level_bout_avg_spd,
        'condition_level_bout_avg_dist': condition_level_bout_avg_dist_lvl,
    } if condition_level_data else None

    return level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig, level_bout_collapsed_fig, level_bout_condition_fig, level_bout_avg_speed_collapsed_fig, level_bout_avg_speed_condition_fig, level_bout_avg_dist_collapsed_fig, level_bout_avg_dist_condition_fig, _level_stats


# ── Descriptive statistics report ────────────────────────────────────────────

def generate_descriptive_stats_report(all_results, level_stats_data=None, output_dir=None):
    """Generate a descriptive statistics Excel workbook (.xlsx) with one sheet
    per metric sliced across training sessions (time) and, if level data are
    available, per level.  Metrics are grouped by starting condition; statistics
    reported per group are: mean, SEM, SD, N, min, max, median.

    Parameters
    ----------
    all_results : list[dict]
        Session-level result list produced by analyze_mouse_data.
    level_stats_data : dict | None
        Dict produced by analyze_levels containing condition_level_* dicts.
        Pass None to skip level sheets.
    output_dir : str | None
        Directory to save the report.  Defaults to current working directory.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from scipy.stats import t as t_dist
    except ImportError:
        print("[WARN] openpyxl not found — descriptive stats report skipped. "
              "Install it with:  pip install openpyxl")
        return

    STAT_COLS = ['mean', 'sem', 'sd', 'n', 'min', 'max', 'median', 'ci95_lo', 'ci95_hi']
    HEADER_FONT   = Font(bold=True, color='FFFFFF')
    HEADER_FILL_A = PatternFill(start_color='1F497D', end_color='1F497D', fill_type='solid')
    ROW_FILLS = [
        PatternFill(start_color='DCE6F1', end_color='DCE6F1', fill_type='solid'),
        PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid'),
    ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _clean(v):
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        return True

    def _desc_stats(vals):
        a = np.array([v for v in vals if _clean(v)], dtype=float)
        if len(a) == 0:
            return dict(mean=np.nan, sem=np.nan, sd=np.nan, n=0,
                        min=np.nan, max=np.nan, median=np.nan,
                        ci95_lo=np.nan, ci95_hi=np.nan)
        _mean = float(np.mean(a))
        _sd   = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
        _sem  = float(_sd / np.sqrt(len(a))) if len(a) > 1 else np.nan
        if len(a) > 1:
            _t = t_dist.ppf(0.975, df=len(a) - 1)
            _ci_lo = _mean - _t * _sem
            _ci_hi = _mean + _t * _sem
        else:
            _ci_lo = _ci_hi = np.nan
        return dict(
            mean=_mean,
            sem=_sem,
            sd=_sd,
            n=int(len(a)),
            min=float(np.min(a)),
            max=float(np.max(a)),
            median=float(np.median(a)),
            ci95_lo=_ci_lo,
            ci95_hi=_ci_hi,
        )

    def _sort_level(lv):
        if lv.startswith('level_'):
            try:
                return (0, int(lv.split('_')[1].split('.')[0]))
            except (ValueError, IndexError):
                pass
        return (1, lv)

    def _write_sheet(writer, sheet_name, index_label, rows):
        """Write a stats sheet to the Excel workbook.

        rows : list[dict]
            Each dict has {index_label: display_value, cond_name: [float_list], ...}.
        """
        all_conds = sorted({k for row in rows for k in row if k != index_label})
        col_names = [index_label]
        for cond in all_conds:
            for stat in STAT_COLS:
                col_names.append(f'{cond}_{stat}')

        records = []
        for row in rows:
            rec = {index_label: row[index_label]}
            for cond in all_conds:
                st = _desc_stats(row.get(cond, []))
                for stat in STAT_COLS:
                    rec[f'{cond}_{stat}'] = st[stat]
            records.append(rec)

        df_sheet = pd.DataFrame(records, columns=col_names)
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

        ws = writer.sheets[sheet_name]
        ws.freeze_panes = 'B2'

        # Style header
        for cell in ws[1]:
            cell.font  = HEADER_FONT
            cell.fill  = HEADER_FILL_A
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Alternating condition shading
        for ci, cond in enumerate(all_conds):
            fill = ROW_FILLS[ci % len(ROW_FILLS)]
            col_start = 2 + ci * len(STAT_COLS)
            col_end   = col_start + len(STAT_COLS) - 1
            for row_idx in range(2, len(records) + 2):
                for col_idx in range(col_start, col_end + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = fill

        # Auto-fit column widths (capped 10–25)
        for col_cells in ws.columns:
            max_len = max(
                (len(str(c.value)) if c.value is not None else 0) for c in col_cells
            )
            ws.column_dimensions[
                get_column_letter(col_cells[0].column)
            ].width = max(10, min(max_len + 2, 25))

    # ── Time-based rows ───────────────────────────────────────────────────────

    conditions_sorted = sorted({r['starting_condition'] for r in all_results})
    max_sessions = max(len(r['df']) for r in all_results)

    def _extract(df_r, sess_idx, col):
        if sess_idx >= len(df_r):
            return np.nan
        v = df_r.iat[sess_idx, df_r.columns.get_loc(col)]
        return float(v) if pd.notna(v) else np.nan

    def _session_rows(col_or_fn):
        """Build rows for a time-based sheet.
        col_or_fn: column name (str) or callable(df_r, sess_idx) -> float.
        """
        rows = []
        for sess_idx in range(max_sessions):
            row = {'Session': sess_idx + 1}
            for cond in conditions_sorted:
                vals = []
                for r in all_results:
                    if r['starting_condition'] != cond:
                        continue
                    df_r = r['df'].reset_index(drop=True)
                    if callable(col_or_fn):
                        v = col_or_fn(df_r, sess_idx)
                    else:
                        v = _extract(df_r, sess_idx, col_or_fn)
                    if _clean(v):
                        vals.append(float(v))
                row[cond] = vals
            rows.append(row)
        return rows

    def _reward_rate(df_r, i):
        h = _extract(df_r, i, 'hits')
        s = _extract(df_r, i, 'session_length')
        return h / s if _clean(h) and _clean(s) and s > 0 else np.nan

    def _lick_rate(df_r, i):
        lc = _extract(df_r, i, 'lick_count')
        s  = _extract(df_r, i, 'session_length')
        return lc / s if _clean(lc) and _clean(s) and s > 0 else np.nan

    def _dist_m(df_r, i):
        v = _extract(df_r, i, 'total_distance')
        return v / 1000.0 if _clean(v) else np.nan

    def _bout_dist_m(df_r, i):
        v = _extract(df_r, i, 'avg_dist_per_bout')
        return v / 1000.0 if _clean(v) else np.nan

    time_sheets = [
        ('Time - Speed',          _session_rows('average_speed')),
        ('Time - Reward Count',   _session_rows('hits')),
        ('Time - Reward Rate',    _session_rows(_reward_rate)),
        ('Time - Lick Rate',      _session_rows(_lick_rate)),
        ('Time - Distance (m)',   _session_rows(_dist_m)),
        ('Time - Bout Count',     _session_rows('bout_count')),
        ('Time - Bout Avg Speed', _session_rows('avg_speed_per_bout')),
        ('Time - Bout Avg Dist',  _session_rows(_bout_dist_m)),
    ]

    # ── Level-based rows ──────────────────────────────────────────────────────

    level_sheets = []
    if level_stats_data is not None:

        def _level_rows(cond_level_dict):
            if not cond_level_dict:
                return []
            lvls = sorted(
                {lv for cd in cond_level_dict.values() for lv in cd},
                key=_sort_level,
            )
            rows = []
            for lv in lvls:
                row = {'Level': lv.replace('level_', 'L').replace('.json', '')}
                for cond in sorted(cond_level_dict.keys()):
                    row[cond] = cond_level_dict[cond].get(lv, [])
                rows.append(row)
            return rows

        level_sheets = [
            ('Level - Reward Rate',    _level_rows(level_stats_data['condition_level_reward_rate'])),
            ('Level - Speed',          _level_rows(level_stats_data['condition_level_speed'])),
            ('Level - Lick Rate',      _level_rows(level_stats_data['condition_level_lick'])),
            ('Level - Distance (m)',   _level_rows(level_stats_data['condition_level_dist'])),
            ('Level - Bout Count',     _level_rows(level_stats_data['condition_level_bout'])),
            ('Level - Bout Avg Speed', _level_rows(level_stats_data['condition_level_bout_avg_spd'])),
            ('Level - Bout Avg Dist',  _level_rows(level_stats_data['condition_level_bout_avg_dist'])),
        ]

    # ── Write workbook ────────────────────────────────────────────────────────

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f'descriptive_stats_{timestamp_str}.xlsx')

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for sheet_name, rows in time_sheets:
            if rows:
                _write_sheet(writer, sheet_name, 'Session', rows)
        for sheet_name, rows in level_sheets:
            if rows:
                _write_sheet(writer, sheet_name, 'Level', rows)

    print(f"\nDescriptive stats report saved to:\n  {out_path}")


def _plot_epoch_panels(all_results, signal_key, ylabel, title_prefix,
                      condition_color_map, window_s=EPOCH_WINDOW_S,
                      hierarchy='event'):
    """Create per-mouse and condition-averaged epoch figures for a given signal.

    Parameters
    ----------
    all_results         : list[dict] — one entry per mouse from analyze_mouse_data.
    signal_key          : str — key in each result dict holding the epoch matrix.
                          For hierarchy='event'  : shape (n_events   × EPOCH_N_SAMPLES).
                          For hierarchy='session': shape (n_sessions  × EPOCH_N_SAMPLES),
                          where each row is already the per-session mean trace.
    ylabel              : str — y-axis label.
    title_prefix        : str — base title string.
    condition_color_map : dict — maps condition name → matplotlib colour.
    window_s            : float — half-window size (seconds) for x-axis limits.
    hierarchy           : 'event' or 'session'.
                          'event'   — rows are individual zone-entry epochs; mean ± SEM
                                      weights every zone entry equally.
                          'session' — rows are per-session mean traces; mean ± SEM
                                      weights every session equally.  Individual session
                                      traces are shown as thin background lines in the
                                      per-mouse panel.

    Returns
    -------
    fig_per_mouse : matplotlib.figure.Figure  (one subplot per mouse)
    fig_cond      : matplotlib.figure.Figure  (condition-averaged overlay)
    """
    canonical_time = EPOCH_CANONICAL_TIME
    unit_label     = 'sessions' if hierarchy == 'session' else 'events'
    hier_label     = '(session-averaged)' if hierarchy == 'session' else '(event-averaged)'

    # ── Figure 1: per-mouse ───────────────────────────────────────────────────
    n_mice = len(all_results)
    ncols  = min(3, n_mice)
    nrows  = (n_mice + ncols - 1) // ncols
    fig_per_mouse, axs = plt.subplots(nrows, ncols,
                                      figsize=(5 * ncols, 4 * nrows),
                                      sharex=True, sharey=True,
                                      squeeze=False)
    axs_flat = axs.flatten()
    _yvals_per = []

    for i, result in enumerate(all_results):
        matrix     = result.get(signal_key)
        ax         = axs_flat[i]
        mouse_name = result['mouse']
        condition  = result['starting_condition']
        color      = condition_color_map.get(condition, 'steelblue')

        if matrix is not None and matrix.shape[0] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mean_trace = np.nanmean(matrix, axis=0)
                n_valid    = np.sum(~np.isnan(matrix), axis=0)
                sem_trace  = np.where(n_valid > 1,
                                      np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n_valid),
                                      0.0)
            _yvals_per.append(float(np.nanmax(mean_trace + sem_trace)))
            _yvals_per.append(float(np.nanmin(mean_trace - sem_trace)))
            ax.plot(canonical_time, mean_trace, color=color, linewidth=1.8,
                    label=f'n={matrix.shape[0]} {unit_label}')
            ax.fill_between(canonical_time,
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            color=color, alpha=0.25)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)

        ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(0.65, color='black', linestyle='--', linewidth=1.0)
        ax.set_title(mouse_name, fontsize=10)
        ax.set_xlim(-window_s, window_s)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='upper right')

    for j in range(n_mice, len(axs_flat)):
        axs_flat[j].set_visible(False)

    for ax in axs_flat[:n_mice]:
        ax.set_xlabel('Time from reward zone entry (s)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    # Shared y-axis: fitted to mean ± SEM of averaged data (5 % headroom).
    # With sharey=True one set_ylim call propagates everywhere.
    if _yvals_per:
        _ymax_per = float(np.nanmax(_yvals_per))
        _ymin_per = float(np.nanmin(_yvals_per))
    else:
        _ymax_per, _ymin_per = 1.0, 0.0
    _bot_per = _ymin_per * 1.05 if _ymin_per < 0 else 0.0
    axs_flat[0].set_ylim(_bot_per, _ymax_per * 1.05)

    fig_per_mouse.suptitle(f'{title_prefix} {hier_label} — Per Mouse', fontsize=13)
    fig_per_mouse.tight_layout()

    # ── Figure 2: condition-averaged ─────────────────────────────────────────
    fig_cond, ax_cond = plt.subplots(figsize=(9, 5))

    condition_mouse_means = {}
    _yvals_cond = []
    for result in all_results:
        matrix    = result.get(signal_key)
        condition = result['starting_condition']
        if matrix is not None and matrix.shape[0] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mouse_mean = np.nanmean(matrix, axis=0)
            condition_mouse_means.setdefault(condition, []).append(mouse_mean)

    for condition in sorted(condition_mouse_means.keys()):
        color       = condition_color_map.get(condition, 'steelblue')
        mouse_means = np.array(condition_mouse_means[condition])  # (n_mice_in_cond, 501)
        n_mice_cond = mouse_means.shape[0]

        # Thin per-mouse lines for within-condition spread
        for mm in mouse_means:
            ax_cond.plot(canonical_time, mm, color=color, linewidth=0.8, alpha=0.4)
            _yvals_cond.append(float(np.nanmax(mm)))
            _yvals_cond.append(float(np.nanmin(mm)))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cond_mean = np.nanmean(mouse_means, axis=0)
            cond_sem  = (np.nanstd(mouse_means, axis=0, ddof=1) / np.sqrt(n_mice_cond)
                         if n_mice_cond > 1 else np.zeros_like(cond_mean))
        _yvals_cond.append(float(np.nanmax(cond_mean + cond_sem)))
        _yvals_cond.append(float(np.nanmin(cond_mean - cond_sem)))

        ax_cond.plot(canonical_time, cond_mean, color=color, linewidth=2.2,
                     label=f'{condition} (n={n_mice_cond} mice)')
        ax_cond.fill_between(canonical_time,
                             cond_mean - cond_sem,
                             cond_mean + cond_sem,
                             color=color, alpha=0.20)

    ax_cond.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zone entry (t=0)')
    ax_cond.axvline(0.65, color='black', linestyle='--', linewidth=1.0, label='Reward delivery (t=0.65 s)')
    ax_cond.set_xlabel('Time from reward zone entry (s)')
    ax_cond.set_ylabel(ylabel)
    ax_cond.set_title(f'{title_prefix} {hier_label} — By Starting Condition')
    ax_cond.set_xlim(-window_s, window_s)
    # Fit y-axis to mean ± SEM of averaged data (5 % headroom)
    if _yvals_cond:
        _ymax_cond = float(np.nanmax(_yvals_cond))
        _ymin_cond = float(np.nanmin(_yvals_cond))
    else:
        _ymax_cond, _ymin_cond = 1.0, 0.0
    _bot_cond = _ymin_cond * 1.05 if _ymin_cond < 0 else 0.0
    ax_cond.set_ylim(_bot_cond, _ymax_cond * 1.05)
    ax_cond.legend()
    ax_cond.spines['top'].set_visible(False)
    ax_cond.spines['right'].set_visible(False)
    fig_cond.tight_layout()

    return fig_per_mouse, fig_cond


def _plot_epoch_early_late_panels(all_results, signal_key, ylabel, title_prefix,
                                   condition_color_map, window_s=EPOCH_WINDOW_S,
                                   indices_key=None, row_unit='sessions'):
    """Early vs late session epoch panels — each half in its own independent figure.

    Uses a 2-level hierarchy: individual row traces (thin background)
    → group mean ± SEM (bold foreground).  The early/late boundary is the same
    global split point for every mouse so that a mouse with missing sessions is
    always split at the same session index as its complete cohort-mates.

    Parameters
    ----------
    indices_key : str | None
        Key in each result dict holding the per-row session index array.
        Defaults to ``signal_key.replace('_means', '_indices')``.
    row_unit : str
        Label string used in the legend (e.g. 'sessions' or 'events').

    Returns
    -------
    fig_pm_early, fig_pm_late, fig_cond_early, fig_cond_late
    """
    canonical_time = EPOCH_CANONICAL_TIME
    if indices_key is None:
        indices_key = signal_key.replace('_means', '_indices')
    n_mice         = len(all_results)

    # ── global split point ────────────────────────────────────────────────────
    _all_idx = [result.get(indices_key) for result in all_results
                if result.get(indices_key) is not None and len(result.get(indices_key)) > 0]
    if _all_idx:
        global_n    = int(max(idx.max() for idx in _all_idx)) + 1
        global_half = global_n // 2
    else:
        global_n    = 0
        global_half = 0

    def _make_half_figs(half_tag):
        """Build per-mouse and condition figures for one half (early or late)."""
        is_early   = (half_tag == 'early')
        half_label = (f'Early Sessions (1\u2013{global_half})'
                      if is_early else
                      f'Late Sessions ({global_half + 1}\u2013{global_n})')

        # ── per-mouse figure (ceil(n_mice/2) rows × 2 cols) ──────────────────
        n_cols = 2
        n_rows = math.ceil(n_mice / n_cols)
        fig_pm, axs = plt.subplots(n_rows, n_cols,
                                   figsize=(12, 4 * n_rows),
                                   sharex=True, sharey=True,
                                   squeeze=False)
        # hide any unused axes in the last row
        for _empty in range(n_mice, n_rows * n_cols):
            axs[_empty // n_cols, _empty % n_cols].set_visible(False)
        _yvals_pm = []

        for i, result in enumerate(all_results):
            matrix     = result.get(signal_key)
            indices    = result.get(indices_key)
            mouse_name = result['mouse']
            condition  = result['starting_condition']
            color      = condition_color_map.get(condition, 'steelblue')
            ax         = axs[i // n_cols, i % n_cols]

            ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
            ax.axvline(0.65, color='black', linestyle='--', linewidth=1.0)
            ax.set_xlim(-window_s, window_s)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f'{mouse_name} — {half_label}', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # x-label on bottom row only
            if i // n_cols == n_rows - 1:
                ax.set_xlabel('Time from reward zone entry (s)', fontsize=9)

            if matrix is not None and indices is not None and len(indices) > 0:
                mask = (indices < global_half) if is_early else (indices >= global_half)
                sub  = matrix[mask]

                if sub.shape[0] == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        mean_trace = np.nanmean(sub, axis=0)
                        n_valid    = np.sum(~np.isnan(sub), axis=0)
                        sem_trace  = np.where(n_valid > 1,
                                              np.nanstd(sub, axis=0, ddof=1) / np.sqrt(n_valid),
                                              0.0)
                    _yvals_pm.append(float(np.nanmax(mean_trace + sem_trace)))
                    _yvals_pm.append(float(np.nanmin(mean_trace - sem_trace)))
                    ax.plot(canonical_time, mean_trace, color=color, linewidth=1.8,
                            label=f'n={sub.shape[0]} {row_unit}')
                    ax.fill_between(canonical_time,
                                    mean_trace - sem_trace,
                                    mean_trace + sem_trace,
                                    color=color, alpha=0.25)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)

            ax.legend(fontsize=8, loc='upper right')

        if _yvals_pm:
            _ymax_pm = float(np.nanmax(_yvals_pm))
            _ymin_pm = float(np.nanmin(_yvals_pm))
        else:
            _ymax_pm, _ymin_pm = 1.0, 0.0
        _bot_pm = _ymin_pm * 1.05 if _ymin_pm < 0 else 0.0
        axs[0, 0].set_ylim(_bot_pm, _ymax_pm * 1.05)
        fig_pm.suptitle(f'{title_prefix} — {half_label} (Per Mouse)', fontsize=13)
        fig_pm.tight_layout()

        # ── condition figure ──────────────────────────────────────────────────
        fig_cond, ax_cond = plt.subplots(figsize=(10, 5))

        cond_data: dict = {}
        _yvals_cond_half = []
        for result in all_results:
            matrix    = result.get(signal_key)
            indices   = result.get(indices_key)
            condition = result['starting_condition']
            if matrix is None or indices is None or len(indices) == 0:
                continue
            mask = (indices < global_half) if is_early else (indices >= global_half)
            sub  = matrix[mask]
            if sub.shape[0] == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cond_data.setdefault(condition, []).append(np.nanmean(sub, axis=0))

        for condition in sorted(cond_data.keys()):
            color       = condition_color_map.get(condition, 'steelblue')
            mouse_means = np.array(cond_data[condition])
            n_m         = mouse_means.shape[0]
            for mm in mouse_means:
                ax_cond.plot(canonical_time, mm, color=color,
                             linewidth=0.6, alpha=0.3)
                _yvals_cond_half.append(float(np.nanmax(mm)))
                _yvals_cond_half.append(float(np.nanmin(mm)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cond_mean = np.nanmean(mouse_means, axis=0)
                cond_sem  = (np.nanstd(mouse_means, axis=0, ddof=1) / np.sqrt(n_m)
                             if n_m > 1 else np.zeros_like(cond_mean))
            _yvals_cond_half.append(float(np.nanmax(cond_mean + cond_sem)))
            _yvals_cond_half.append(float(np.nanmin(cond_mean - cond_sem)))
            ax_cond.plot(canonical_time, cond_mean, color=color, linewidth=2.2,
                         label=f'{condition} (n={n_m} mice)')
            ax_cond.fill_between(canonical_time,
                                 cond_mean - cond_sem,
                                 cond_mean + cond_sem,
                                 color=color, alpha=0.15)

        ax_cond.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zone entry (t=0)')
        ax_cond.axvline(0.65, color='black', linestyle='--', linewidth=1.0, label='Reward delivery (t=0.65 s)')
        ax_cond.set_xlabel('Time from reward zone entry (s)')
        ax_cond.set_ylabel(ylabel)
        ax_cond.set_title(f'{title_prefix} — {half_label} (By Condition)')
        ax_cond.set_xlim(-window_s, window_s)
        if _yvals_cond_half:
            _ymax_cond = float(np.nanmax(_yvals_cond_half))
            _ymin_cond = float(np.nanmin(_yvals_cond_half))
        else:
            _ymax_cond, _ymin_cond = 1.0, 0.0
        _bot_cond_half = _ymin_cond * 1.05 if _ymin_cond < 0 else 0.0
        ax_cond.set_ylim(_bot_cond_half, _ymax_cond * 1.05)
        ax_cond.legend()
        ax_cond.spines['top'].set_visible(False)
        ax_cond.spines['right'].set_visible(False)
        fig_cond.tight_layout()

        return fig_pm, fig_cond

    fig_pm_early, fig_cond_early = _make_half_figs('early')
    fig_pm_late,  fig_cond_late  = _make_half_figs('late')
    return fig_pm_early, fig_pm_late, fig_cond_early, fig_cond_late


def analyze_mouse_data(data_files, markers, starting_conditions, transitions_csv_path=None, selected_plots=None, save_lick_plots=False, output_dir=None):
    # Create dictionaries to map mouse names to markers and starting conditions
    markers = {os.path.basename(file).split("_")[0]: marker for file, marker in zip(data_files, markers)}
    conditions = {os.path.basename(file).split("_")[0]: condition for file, condition in zip(data_files, starting_conditions)}
    if selected_plots is None:
        selected_plots = set(_ALL_PLOT_KEYS)

    # Create output directory for lick detection plots if needed
    if save_lick_plots and output_dir:
        lick_plots_dir = os.path.join(output_dir, 'lick_detection_plots')
        os.makedirs(lick_plots_dir, exist_ok=True)
        print(f"\nSaving lick detection plots to: {lick_plots_dir}")
    
    # Create color mapping based on starting conditions
    # Sort so that the condition→color assignment is deterministic across runs
    unique_conditions = sorted(set(starting_conditions))
    condition_colors = generate_colors(len(unique_conditions))
    condition_color_map = {condition: color for condition, color in zip(unique_conditions, condition_colors)}
    
    speed_fig             = plt.figure(figsize=(12, 6)) if 'speed'              in selected_plots else None
    sensitivity_fig       = plt.figure(figsize=(12, 6)) if 'sensitivity'        in selected_plots else None
    lick_fig              = plt.figure(figsize=(12, 6)) if 'lick_count'         in selected_plots else None
    reward_fig            = plt.figure(figsize=(12, 6)) if 'reward_count'       in selected_plots else None
    avg_reward_fig        = plt.figure(figsize=(12, 6)) if 'avg_reward'         in selected_plots else None
    sex_reward_fig        = plt.figure(figsize=(12, 6)) if 'sex_reward'         in selected_plots else None
    avg_lick_rate_fig     = plt.figure(figsize=(12, 6)) if 'avg_lick_rate'      in selected_plots else None
    sex_lick_rate_fig     = plt.figure(figsize=(12, 6)) if 'sex_lick_rate'      in selected_plots else None
    false_alarm_fig       = plt.figure(figsize=(12, 6)) if 'false_alarms'       in selected_plots else None
    correct_rejection_fig = plt.figure(figsize=(12, 6)) if 'correct_rejections' in selected_plots else None
    specificity_fig       = plt.figure(figsize=(12, 6)) if 'specificity'        in selected_plots else None
    dprime_fig            = plt.figure(figsize=(12, 6)) if 'dprime'             in selected_plots else None
    distance_fig          = plt.figure(figsize=(12, 6)) if 'distance'           in selected_plots else None
    bout_count_fig        = plt.figure(figsize=(12, 6)) if 'bout_count'         in selected_plots else None
    avg_bout_count_fig    = plt.figure(figsize=(12, 6)) if 'avg_bout_count'     in selected_plots else None
    bout_avg_speed_fig    = plt.figure(figsize=(12, 6)) if 'bout_avg_speed'     in selected_plots else None
    bout_avg_dist_fig     = plt.figure(figsize=(12, 6)) if 'bout_avg_dist'      in selected_plots else None
    colors = generate_colors(len(data_files))  # Generate colors based on number of mice
    
    all_results = []
    
    for idx, data_file in enumerate(data_files):
        # Read the combined data file
        df = pd.read_csv(data_file, index_col='timestamp')
        
        print(f"Reading data from: {data_file}")
        
        # Initialize lists to store results
        dates = []
        speeds = []
        total_distances = []  # List for total session distances (cm)
        bout_counts = []  # List for locomotion bout counts per session
        avg_speeds_per_bout = []  # List for avg speed per locomotion bout (cm/s)
        avg_dists_per_bout  = []  # List for avg distance per locomotion bout (mm)
        hits = []  # List for reward events
        misses_list = []  # List for misses (texture changes minus hits)
        sensitivities = []  # List for sensitivity values
        lick_counts = []  # List for daily lick counts
        session_lengths = []  # List for session lengths in minutes
        false_alarms_list = []  # List for false alarm counts
        correct_rejections_list = []  # List for correct rejection counts
        specificities_list = []  # List for specificity values
        dprimes_list = []  # List for d-prime values
        session_file_errors = {}  # {date_str: [list of missing file labels]}
        speed_epoch_windows_all       = []  # list of 2-D arrays (n_events × EPOCH_N_SAMPLES) per session
        cap_epoch_windows_all         = []  # same for capacitive
        speed_epoch_session_means_all   = []  # list of 1-D arrays — one per-session mean trace
        cap_epoch_session_means_all     = []  # same for capacitive
        speed_epoch_session_indices_all = []  # 0-based session position in df for each speed mean
        cap_epoch_session_indices_all   = []  # same for capacitive
        speed_epoch_event_indices_all   = []  # session index repeated for every event row in speed_epoch_matrix
        cap_epoch_event_indices_all     = []  # same for capacitive

        # Process each date's data
        for _sess_idx, (timestamp, row) in enumerate(df.iterrows()):
            date_str = datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
            missing_files = []
            try:
                # Read the treadmill data from the file path
                try:
                    treadmill_data = pd.read_csv(row['treadmill'])
                except Exception:
                    missing_files.append('treadmill')
                    treadmill_data = None

                # Read capacitive data for lick detection
                try:
                    capacitive_data = pd.read_csv(row['capacitive'])
                except Exception:
                    missing_files.append('capacitive')
                    capacitive_data = None

                # Read trial log
                try:
                    trial_log = pd.read_csv(row['trial_log'])
                except Exception:
                    missing_files.append('trial_log')
                    trial_log = None

                if len(missing_files) == 3:
                    # All three files missing — nothing to process for this date
                    session_file_errors[date_str] = missing_files
                    print(f"  [WARN] {date_str}: all files missing — skipping session entirely")
                    continue

                if missing_files:
                    session_file_errors[date_str] = missing_files
                    print(f"  [WARN] {date_str}: missing files: {', '.join(missing_files)} — partial session kept")

                # ── Treadmill-derived metrics ─────────────────────────────
                if treadmill_data is not None:
                    _speed_raw = treadmill_data['speed']
                    # Butterworth low-pass (0.25 Hz, order 3) applied to raw speed
                    _fs = 1.0 / treadmill_data['global_time'].diff().median()
                    _b, _a = butter(3, 0.25 / (_fs / 2.0), btype='low')
                    _speed_filt = filtfilt(_b, _a, _speed_raw)
                    avg_speed = float(np.mean(_speed_filt)) / 10.0
                    _, total_distance = compute_session_distance(treadmill_data)
                    # Bout detection on filtered speed (mm/s → cm/s)
                    _time_arr = treadmill_data['global_time'].values
                    _speed_cm = _speed_filt / 10.0
                    _bouts = detect_locomotion_bouts(_time_arr, _speed_cm)
                    bout_count = len(_bouts)
                    # Per-bout averages (speed in cm/s; dist in mm)
                    if _bouts:
                        _bout_speed_means = []
                        _bout_dist_means  = []
                        for _t0, _t1 in _bouts:
                            _bmask = (_time_arr >= _t0) & (_time_arr <= _t1)
                            if np.any(_bmask):
                                _bout_speed_means.append(float(np.mean(_speed_cm[_bmask])))
                            if 'distance' in treadmill_data.columns:
                                _dist_vals = pd.to_numeric(
                                    treadmill_data['distance'], errors='coerce').values[_bmask]
                                _dist_vals = _dist_vals[~np.isnan(_dist_vals)]
                                if len(_dist_vals) >= 2:
                                    _bout_dist_means.append(float(_dist_vals[-1] - _dist_vals[0]))
                        avg_speed_per_bout = (float(np.mean(_bout_speed_means))
                                             if _bout_speed_means else float('nan'))
                        avg_dist_per_bout  = (float(np.mean(_bout_dist_means))
                                             if _bout_dist_means  else float('nan'))
                    else:
                        avg_speed_per_bout = float('nan')
                        avg_dist_per_bout  = float('nan')
                else:
                    avg_speed = float('nan')
                    total_distance = float('nan')
                    bout_count = float('nan')
                    avg_speed_per_bout = float('nan')
                    avg_dist_per_bout  = float('nan')

                # ── Capacitive-derived metrics ────────────────────────────
                if capacitive_data is not None:
                    session_length_minutes = capacitive_data['elapsed_time'].max() / 60.0

                    cap_df = capacitive_data.copy()
                    cap_df['Time_sec'] = cap_df['elapsed_time']

                    # Compute KDE normalization (with caching)
                    capacitive_filepath = row['capacitive']
                    kde_value = get_cached_kde(capacitive_filepath)
                    if kde_value is None:
                        kde_value = lda.compute_KDE(cap_df, 'capacitive_value')
                        cache_kde_value(capacitive_filepath, kde_value)

                    cap_df = lda.compute_KDE_normalizations(cap_df, 'capacitive_value', kde_value)
                    events_df, threshold_used = lda.detect_events_above_threshold(cap_df, 'capacitive_value', threshold=None)
                    lick_count = events_df['capacitive_value_event'].sum()

                    if save_lick_plots and output_dir:
                        mouse_name_plot = os.path.basename(data_file).split('_')[0]
                        plot_filename = f"{mouse_name_plot}_{date_str}_lick_detection.png"
                        plot_path = os.path.join(lick_plots_dir, plot_filename)
                        fig = lda.plot_summary(
                            cap_df, events_df,
                            column='capacitive_value',
                            kde_value=kde_value,
                            threshold=threshold_used,
                            title=f"{mouse_name_plot} - {date_str} - {lick_count} licks detected",
                            show=False
                        )
                        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                else:
                    session_length_minutes = float('nan')
                    lick_count = float('nan')

                # ── Trial-log-derived metrics ─────────────────────────────
                if trial_log is not None:
                    # Extract ALL stay zone entry times (if column exists; older data may not have it)
                    if 'stay_texture_change_time' in trial_log.columns:
                        stay_zone_times = pd.to_numeric(trial_log['stay_texture_change_time'], errors='coerce').dropna().values

                        # Collect all re-entry times to exclude them
                        re_entry_times_set = set()
                        if 'zone_re_entry_time' in trial_log.columns:
                            for val in trial_log['zone_re_entry_time']:
                                if pd.notna(val) and val != '':
                                    try:
                                        # Handle both single values and arrays
                                        if isinstance(val, str) and val.strip():
                                            import ast
                                            try:
                                                re_entry_list = ast.literal_eval(val)
                                                if isinstance(re_entry_list, (list, tuple)):
                                                    for t in re_entry_list:
                                                        if pd.notna(t):
                                                            re_entry_times_set.add(float(t))
                                                else:
                                                    re_entry_times_set.add(float(re_entry_list))
                                            except:
                                                pass
                                        elif not isinstance(val, str):
                                            re_entry_times_set.add(float(val))
                                    except (ValueError, TypeError):
                                        pass

                        # Filter out stay_zone_times that match re-entries (within 0.05 seconds)
                        valid_zone_times = []
                        for zone_time in stay_zone_times:
                            is_reentry = False
                            for re_entry_time in re_entry_times_set:
                                if abs(zone_time - re_entry_time) <= 0.05:
                                    is_reentry = True
                                    break
                            if not is_reentry:
                                valid_zone_times.append(zone_time)

                        valid_zone_times = np.array(valid_zone_times)
                        total_opportunities = len(valid_zone_times)
                    else:
                        # Fallback for older data without stay_texture_change_time
                        total_opportunities = len(trial_log[trial_log['texture_history'] == 'assets/reward_mean100.jpg'])

                    # Collect all reward_event and hits_event times
                    reward_event_times = pd.to_numeric(trial_log['reward_event'], errors='coerce').dropna().values

                    # Check if hits_event column exists (older data may not have it)
                    if 'hits_event' in trial_log.columns:
                        hits_event_times = pd.to_numeric(trial_log['hits_event'], errors='coerce').dropna().values
                    else:
                        hits_event_times = np.array([])  # Empty array if column doesn't exist

                    # Simple counting: hits = rewards + hits, misses = opportunities - hits
                    if total_opportunities > 0:
                        reward_count = len(reward_event_times) + len(hits_event_times)
                        misses = total_opportunities - reward_count
                        sensitivity = float(reward_count) / float(total_opportunities) if total_opportunities > 0 else 0.0
                    else:
                        reward_count = 0
                        misses = 0
                        sensitivity = float('nan')

                    # Compute puff opportunities and false alarms
                    puff_event_times = pd.to_numeric(trial_log['puff_event'], errors='coerce').dropna().values

                    if 'go_texture_change_time' in trial_log.columns:
                        go_zone_times = np.sort(
                            pd.to_numeric(trial_log['go_texture_change_time'], errors='coerce').dropna().values
                        )
                        puff_opportunities = len(go_zone_times)

                        # Count false alarms: each go zone that had at least one puff event is one false alarm.
                        false_alarm_count = 0
                        for z_idx, zone_start in enumerate(go_zone_times):
                            zone_end = go_zone_times[z_idx + 1] if z_idx + 1 < len(go_zone_times) else np.inf
                            zone_puffs = puff_event_times[
                                (puff_event_times >= zone_start) & (puff_event_times < zone_end)
                            ]
                            if len(zone_puffs) > 0:
                                false_alarm_count += 1
                    else:
                        # Older format: use texture_change_time from punish rows as zone start times
                        punish_rows = trial_log[trial_log['texture_history'] == 'assets/punish_mean100.jpg']
                        puff_opportunities = len(punish_rows)
                        punish_zone_times = np.sort(
                            pd.to_numeric(punish_rows['texture_change_time'], errors='coerce').dropna().values
                        )

                        false_alarm_count = 0
                        for z_idx, zone_start in enumerate(punish_zone_times):
                            zone_end = punish_zone_times[z_idx + 1] if z_idx + 1 < len(punish_zone_times) else np.inf
                            zone_puffs = puff_event_times[
                                (puff_event_times >= zone_start) & (puff_event_times < zone_end)
                            ]
                            if len(zone_puffs) > 0:
                                false_alarm_count += 1

                    # Compute correct rejections and specificity
                    if puff_opportunities > 0:
                        correct_rejection_count = puff_opportunities - false_alarm_count
                        specificity = float(correct_rejection_count) / float(puff_opportunities)
                    else:
                        false_alarm_count = 0
                        correct_rejection_count = 0
                        specificity = float('nan')

                    # Compute d-prime (signal detection theory)
                    # Log-linear correction handles edge cases of hit/FA rate = 0 or 1
                    if total_opportunities > 0 and puff_opportunities > 0:
                        hr_corrected = (reward_count + 0.5) / (total_opportunities + 1.0)
                        fa_corrected = (false_alarm_count + 0.5) / (puff_opportunities + 1.0)
                        dprime = float(norm.ppf(hr_corrected) - norm.ppf(fa_corrected))
                    else:
                        dprime = float('nan')
                else:
                    reward_count = float('nan')
                    misses = float('nan')
                    sensitivity = float('nan')
                    false_alarm_count = float('nan')
                    correct_rejection_count = float('nan')
                    specificity = float('nan')
                    dprime = float('nan')

                # Convert Unix timestamp to datetime and store results
                date = datetime.fromtimestamp(int(timestamp))

                dates.append(date)
                speeds.append(avg_speed)
                total_distances.append(total_distance)
                bout_counts.append(bout_count)
                avg_speeds_per_bout.append(avg_speed_per_bout)
                avg_dists_per_bout.append(avg_dist_per_bout)
                hits.append(reward_count)
                misses_list.append(misses)
                sensitivities.append(sensitivity)
                lick_counts.append(lick_count)
                session_lengths.append(session_length_minutes)
                false_alarms_list.append(false_alarm_count)
                correct_rejections_list.append(correct_rejection_count)
                specificities_list.append(specificity)
                dprimes_list.append(dprime)

                # ── Behavioral epoch extraction (reward zone entry) ──────────────────
                if trial_log is not None and (treadmill_data is not None or capacitive_data is not None):
                    try:
                        _zone_times = _extract_reward_zone_entry_times(trial_log)
                        if _zone_times:
                            if treadmill_data is not None:
                                # Use raw (unfiltered) speed via uniformly_sample_treadmill.
                                _sp_time, _sp_val = uniformly_sample_treadmill(treadmill_data)
                                _sp_mat = _build_epoch_matrix(_sp_time, _sp_val, _zone_times)
                                if _sp_mat is not None:
                                    speed_epoch_windows_all.append(_sp_mat)
                                    with warnings.catch_warnings():
                                        warnings.simplefilter('ignore', RuntimeWarning)
                                        speed_epoch_session_means_all.append(
                                            np.nanmean(_sp_mat, axis=0))
                                    speed_epoch_session_indices_all.append(_sess_idx)
                                    speed_epoch_event_indices_all.extend(
                                        [_sess_idx] * _sp_mat.shape[0])
                            if capacitive_data is not None:
                                _cp_time, _cp_val = uniformly_sample_capacitive(capacitive_data)
                                # Z-score the capacitive signal for this session so that
                                # sessions with different baselines / dynamic ranges are
                                # comparable when epochs are averaged across sessions or mice.
                                _cp_mu  = np.nanmean(_cp_val)
                                _cp_sig = np.nanstd(_cp_val, ddof=1)
                                if _cp_sig > 0:
                                    _cp_val = (_cp_val - _cp_mu) / _cp_sig
                                else:
                                    _cp_val = _cp_val - _cp_mu  # zero-mean, can't scale
                                _cp_mat = _build_epoch_matrix(_cp_time, _cp_val, _zone_times)
                                if _cp_mat is not None:
                                    cap_epoch_windows_all.append(_cp_mat)
                                    with warnings.catch_warnings():
                                        warnings.simplefilter('ignore', RuntimeWarning)
                                        cap_epoch_session_means_all.append(
                                            np.nanmean(_cp_mat, axis=0))
                                    cap_epoch_session_indices_all.append(_sess_idx)
                                    cap_epoch_event_indices_all.extend(
                                        [_sess_idx] * _cp_mat.shape[0])
                    except Exception as _epoch_err:
                        print(f"  [WARN] {date_str}: epoch extraction failed — {_epoch_err}")

            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'average_speed': speeds,
            'total_distance': total_distances,
            'bout_count': bout_counts,
            'avg_speed_per_bout': avg_speeds_per_bout,
            'avg_dist_per_bout':  avg_dists_per_bout,
            'hits': hits,
            'misses': misses_list,
            'sensitivity': sensitivities,
            'lick_count': lick_counts,
            'session_length': session_lengths,
            'false_alarms': false_alarms_list,
            'correct_rejections': correct_rejections_list,
            'specificity': specificities_list,
            'dprime': dprimes_list
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

        # Stack per-session epoch window matrices into one array per signal
        speed_epoch_matrix        = (np.vstack(speed_epoch_windows_all)
                                     if speed_epoch_windows_all       else None)
        cap_epoch_matrix          = (np.vstack(cap_epoch_windows_all)
                                     if cap_epoch_windows_all         else None)
        speed_epoch_session_means   = (np.vstack(speed_epoch_session_means_all)
                                       if speed_epoch_session_means_all else None)
        cap_epoch_session_means     = (np.vstack(cap_epoch_session_means_all)
                                       if cap_epoch_session_means_all   else None)
        speed_epoch_session_indices = (np.array(speed_epoch_session_indices_all, dtype=int)
                                       if speed_epoch_session_indices_all else None)
        cap_epoch_session_indices   = (np.array(cap_epoch_session_indices_all, dtype=int)
                                       if cap_epoch_session_indices_all   else None)
        speed_epoch_event_indices   = (np.array(speed_epoch_event_indices_all, dtype=int)
                                       if speed_epoch_event_indices_all   else None)
        cap_epoch_event_indices     = (np.array(cap_epoch_event_indices_all, dtype=int)
                                       if cap_epoch_event_indices_all     else None)

        # Store results for this mouse
        all_results.append({
            'mouse': mouse_name,
            'dates': dates,
            'speeds': speeds,
            'total_distances': total_distances,
            'bout_counts': bout_counts,
            'avg_speeds_per_bout': avg_speeds_per_bout,
            'avg_dists_per_bout':  avg_dists_per_bout,
            'hits': hits,
            'false_alarms': false_alarms_list,
            'correct_rejections': correct_rejections_list,
            'dprimes': dprimes_list,
            'session_lengths': session_lengths,
            'starting_condition': conditions[mouse_name],
            'df': results_df,
            'session_file_errors': session_file_errors,
            'speed_epoch_matrix':        speed_epoch_matrix,
            'cap_epoch_matrix':          cap_epoch_matrix,
            'speed_epoch_session_means':   speed_epoch_session_means,
            'cap_epoch_session_means':     cap_epoch_session_means,
            'speed_epoch_session_indices': speed_epoch_session_indices,
            'cap_epoch_session_indices':   cap_epoch_session_indices,
            'speed_epoch_event_indices':   speed_epoch_event_indices,
            'cap_epoch_event_indices':     cap_epoch_event_indices,
        })

    # ── Date-aligned per-mouse plots ─────────────────────────────────────────
    # Plotting is deferred to here, after the data-collection loop, so that
    # every mouse is placed on a shared date-based x-axis.  Missing sessions
    # for a given mouse leave a gap rather than compressing the axis leftward.
    all_dates_flat = [d for result in all_results for d in result['df']['date']]
    global_start = min(all_dates_flat).date()
    global_end   = max(all_dates_flat).date()
    total_days   = (global_end - global_start).days + 1
    # max_sessions: number of unique training days (no calendar gaps)
    max_sessions = max(len(result['df']) for result in all_results)

    for result in all_results:
        mouse_name      = result['mouse']
        df_r            = result['df']
        condition_color = condition_color_map[conditions[mouse_name]]
        day_numbers     = list(range(len(df_r)))  # sequential session index, no calendar gaps

        if speed_fig is not None:
            plt.figure(speed_fig.number)
            plt.plot(day_numbers, df_r['average_speed'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if sensitivity_fig is not None:
            plt.figure(sensitivity_fig.number)
            plt.plot(day_numbers, df_r['sensitivity'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if lick_fig is not None:
            plt.figure(lick_fig.number)
            plt.plot(day_numbers, df_r['lick_count'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if reward_fig is not None:
            plt.figure(reward_fig.number)
            plt.plot(day_numbers, df_r['hits'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if false_alarm_fig is not None:
            plt.figure(false_alarm_fig.number)
            plt.plot(day_numbers, df_r['false_alarms'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if correct_rejection_fig is not None:
            plt.figure(correct_rejection_fig.number)
            plt.plot(day_numbers, df_r['correct_rejections'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if specificity_fig is not None:
            plt.figure(specificity_fig.number)
            plt.plot(day_numbers, df_r['specificity'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if dprime_fig is not None:
            plt.figure(dprime_fig.number)
            plt.plot(day_numbers, df_r['dprime'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if distance_fig is not None:
            plt.figure(distance_fig.number)
            # convert mm → m
            dist_m = df_r['total_distance'] / 1000.0
            plt.plot(day_numbers, dist_m,
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if bout_count_fig is not None:
            plt.figure(bout_count_fig.number)
            plt.plot(day_numbers, df_r['bout_count'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if bout_avg_speed_fig is not None:
            plt.figure(bout_avg_speed_fig.number)
            plt.plot(day_numbers, df_r['avg_speed_per_bout'],
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

        if bout_avg_dist_fig is not None:
            plt.figure(bout_avg_dist_fig.number)
            # convert mm → m
            plt.plot(day_numbers, df_r['avg_dist_per_bout'] / 1000.0,
                f'{markers[mouse_name]}-', color=condition_color, markersize=8, label=mouse_name)

    # Tick spacing for individual mouse plots
    max_day = max_sessions
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

    # Configure speed plot
    if speed_fig is not None:
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
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure sensitivity plot
    if sensitivity_fig is not None:
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
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure lick count plot
    if lick_fig is not None:
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
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure reward count plot
    if reward_fig is not None:
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
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure false alarm plot
    if false_alarm_fig is not None:
        plt.figure(false_alarm_fig.number)
        plt.title('False Alarms Over Time')
        plt.xlabel('Training Day')
        plt.ylabel('Number of False Alarms')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure correct rejection plot
    if correct_rejection_fig is not None:
        plt.figure(correct_rejection_fig.number)
        plt.title('Correct Rejections Over Time')
        plt.xlabel('Training Day')
        plt.ylabel('Number of Correct Rejections')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure specificity plot
    if specificity_fig is not None:
        plt.figure(specificity_fig.number)
        plt.title('Specificity Over Time')
        plt.xlabel('Training Day')
        plt.ylabel('Specificity (Correct Rejections / Puff Opportunities)')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure d-prime plot
    if dprime_fig is not None:
        plt.figure(dprime_fig.number)
        plt.title("d' Over Time")
        plt.xlabel('Training Day')
        plt.ylabel("d' (Signal Detection)")
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Configure distance plot
    if distance_fig is not None:
        plt.figure(distance_fig.number)
        plt.title('Total Distance Per Session')
        plt.xlabel('Training Day')
        plt.ylabel('Distance (m)')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    if bout_count_fig is not None:
        plt.figure(bout_count_fig.number)
        plt.title('Locomotion Bout Count Per Session')
        plt.xlabel('Training Day')
        plt.ylabel('Bout Count')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    if bout_avg_speed_fig is not None:
        plt.figure(bout_avg_speed_fig.number)
        plt.title('Average Speed per Locomotion Bout')
        plt.xlabel('Training Day')
        plt.ylabel('Speed per Bout (cm/s)')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    if bout_avg_dist_fig is not None:
        plt.figure(bout_avg_dist_fig.number)
        plt.title('Average Distance per Locomotion Bout')
        plt.xlabel('Training Day')
        plt.ylabel('Distance per Bout (m)')
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max_day - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_spacing))
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Calculate average rewards/minute and SEM across mice
    max_days = max_sessions
    
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

    # Build session-indexed per-mouse rewards-per-minute arrays (no calendar gaps)
    all_rewards_per_min = np.full((len(data_files), max_sessions), np.nan)
    male_rewards_per_min   = []
    female_rewards_per_min = []

    for i, result in enumerate(all_results):
        df_r    = result['df']
        rpm_row = np.full(max_sessions, np.nan)
        for session_idx, (_, row) in enumerate(df_r.iterrows()):
            if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['hits']):
                rpm_row[session_idx] = row['hits'] / row['session_length']
        all_rewards_per_min[i] = rpm_row

        mouse_name = result['mouse']
        if markers[mouse_name] == 's':  # Male
            male_rewards_per_min.append(rpm_row)
        else:  # Female (marker 'o')
            female_rewards_per_min.append(rpm_row)

    if male_rewards_per_min:
        male_rewards_per_min = np.array(male_rewards_per_min)
    if female_rewards_per_min:
        female_rewards_per_min = np.array(female_rewards_per_min)
    
    # Calculate mean and SEM across mice for each day (only over mice that have data on that day)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n_all = np.sum(~np.isnan(all_rewards_per_min), axis=0)
        mean_rewards_per_min = np.where(n_all > 0, np.nanmean(all_rewards_per_min, axis=0), np.nan)
        sem_rewards_per_min = np.where(n_all > 1,
                                       np.nanstd(all_rewards_per_min, axis=0) / np.sqrt(n_all),
                                       0)
    
    # Plot average rewards/minute with SEM
    if avg_reward_fig is not None:
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
    if sex_reward_fig is not None:
        plt.figure(sex_reward_fig.number)
        day_numbers = np.arange(0, max_days)
        
        # Plot male data if available
        if len(male_rewards_per_min) > 0:
            # Check if we have any non-NaN values
            valid_male_data = np.any(~np.isnan(male_rewards_per_min))
            if valid_male_data:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    n_male = np.sum(~np.isnan(male_rewards_per_min), axis=0)
                    mean_male = np.where(n_male > 0, np.nanmean(male_rewards_per_min, axis=0), np.nan)
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
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    n_female = np.sum(~np.isnan(female_rewards_per_min), axis=0)
                    mean_female = np.where(n_female > 0, np.nanmean(female_rewards_per_min, axis=0), np.nan)
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

    # Build session-indexed per-mouse licks-per-minute arrays (no calendar gaps)
    all_licks_per_min = np.full((len(data_files), max_sessions), np.nan)
    male_licks_per_min   = []
    female_licks_per_min = []

    for i, result in enumerate(all_results):
        df_r    = result['df']
        lpm_row = np.full(max_sessions, np.nan)
        for session_idx, (_, row) in enumerate(df_r.iterrows()):
            if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['lick_count']):
                lpm_row[session_idx] = row['lick_count'] / row['session_length']
        all_licks_per_min[i] = lpm_row
        mouse_name = result['mouse']
        if markers[mouse_name] == 's':
            male_licks_per_min.append(lpm_row)
        else:
            female_licks_per_min.append(lpm_row)

    if male_licks_per_min:
        male_licks_per_min = np.array(male_licks_per_min)
    if female_licks_per_min:
        female_licks_per_min = np.array(female_licks_per_min)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n_all_lpm = np.sum(~np.isnan(all_licks_per_min), axis=0)
        mean_licks_per_min = np.where(n_all_lpm > 0, np.nanmean(all_licks_per_min, axis=0), np.nan)
        sem_licks_per_min  = np.where(n_all_lpm > 1,
                                      np.nanstd(all_licks_per_min, axis=0) / np.sqrt(n_all_lpm),
                                      0)

    # Plot average licks/minute with SEM
    if avg_lick_rate_fig is not None:
        plt.figure(avg_lick_rate_fig.number)
        day_numbers = np.arange(0, max_days)
        plt.plot(day_numbers, mean_licks_per_min, '-', color='black', linewidth=2, label='Mean')
        plt.fill_between(day_numbers,
                         mean_licks_per_min - sem_licks_per_min,
                         mean_licks_per_min + sem_licks_per_min,
                         color='gray', alpha=0.3, label='SEM')
        plt.title('Average Lick Rate Across Mice')
        plt.xlabel('Day')
        plt.ylabel('Licks per Minute (Mean \u00b1 SEM)')
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

    # Plot sex-specific average licks/minute with SEM
    if sex_lick_rate_fig is not None:
        plt.figure(sex_lick_rate_fig.number)
        day_numbers = np.arange(0, max_days)
        if len(male_licks_per_min) > 0:
            if np.any(~np.isnan(male_licks_per_min)):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    n_male_lpm = np.sum(~np.isnan(male_licks_per_min), axis=0)
                    mean_male_lpm = np.where(n_male_lpm > 0, np.nanmean(male_licks_per_min, axis=0), np.nan)
                    sem_male_lpm  = np.where(n_male_lpm > 1,
                                             np.nanstd(male_licks_per_min, axis=0) / np.sqrt(n_male_lpm),
                                             0)
                plt.plot(day_numbers, mean_male_lpm, '-', color='green', linewidth=2,
                         label=f'Male (n={len(male_licks_per_min)})')
                plt.fill_between(day_numbers, mean_male_lpm - sem_male_lpm,
                                 mean_male_lpm + sem_male_lpm, color='green', alpha=0.2)
        if len(female_licks_per_min) > 0:
            if np.any(~np.isnan(female_licks_per_min)):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    n_female_lpm = np.sum(~np.isnan(female_licks_per_min), axis=0)
                    mean_female_lpm = np.where(n_female_lpm > 0, np.nanmean(female_licks_per_min, axis=0), np.nan)
                    sem_female_lpm  = np.where(n_female_lpm > 1,
                                               np.nanstd(female_licks_per_min, axis=0) / np.sqrt(n_female_lpm),
                                               0)
                plt.plot(day_numbers, mean_female_lpm, '-', color='purple', linewidth=2,
                         label=f'Female (n={len(female_licks_per_min)})')
                plt.fill_between(day_numbers, mean_female_lpm - sem_female_lpm,
                                 mean_female_lpm + sem_female_lpm, color='purple', alpha=0.2)
        plt.title('Sex-Specific Average Lick Rate')
        plt.xlabel('Training Day')
        plt.ylabel('Licks per Minute (Mean \u00b1 SEM)')
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

    # Build session-indexed per-mouse bout count arrays
    all_bout_counts = np.full((len(data_files), max_sessions), np.nan)

    for i, result in enumerate(all_results):
        df_r = result['df']
        bc_row = np.full(max_sessions, np.nan)
        for session_idx, (_, row) in enumerate(df_r.iterrows()):
            if pd.notna(row['bout_count']):
                bc_row[session_idx] = row['bout_count']
        all_bout_counts[i] = bc_row

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n_all_bc = np.sum(~np.isnan(all_bout_counts), axis=0)
        mean_bout_counts = np.where(n_all_bc > 0, np.nanmean(all_bout_counts, axis=0), np.nan)
        sem_bout_counts  = np.where(n_all_bc > 1,
                                    np.nanstd(all_bout_counts, axis=0) / np.sqrt(n_all_bc),
                                    0)

    if avg_bout_count_fig is not None:
        plt.figure(avg_bout_count_fig.number)
        day_numbers = np.arange(0, max_days)
        plt.plot(day_numbers, mean_bout_counts, '-', color='black', linewidth=2, label='Mean')
        plt.fill_between(day_numbers,
                         mean_bout_counts - sem_bout_counts,
                         mean_bout_counts + sem_bout_counts,
                         color='gray', alpha=0.3, label='SEM')
        plt.title('Average Locomotion Bout Count Across Mice')
        plt.xlabel('Training Day')
        plt.ylabel('Bout Count (Mean \u00b1 SEM)')
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

    # Build session-indexed per-mouse distance arrays (mm → m)
    all_distances_m = np.full((len(data_files), max_sessions), np.nan)
    male_distances_m   = []
    female_distances_m = []

    for i, result in enumerate(all_results):
        df_r = result['df']
        dist_row = np.full(max_sessions, np.nan)
        for session_idx, (_, row) in enumerate(df_r.iterrows()):
            if pd.notna(row['total_distance']):
                dist_row[session_idx] = row['total_distance'] / 1000.0  # mm → m
        all_distances_m[i] = dist_row
        mouse_name = result['mouse']
        if markers[mouse_name] == 's':
            male_distances_m.append(dist_row)
        else:
            female_distances_m.append(dist_row)

    if male_distances_m:
        male_distances_m = np.array(male_distances_m)
    if female_distances_m:
        female_distances_m = np.array(female_distances_m)

    # Sex-specific distance plot
    sex_distance_fig = plt.figure(figsize=(12, 6)) if 'sex_distance' in selected_plots else None
    if sex_distance_fig is not None:
        plt.figure(sex_distance_fig.number)
        day_numbers = np.arange(0, max_days)
        if len(male_distances_m) > 0 and np.any(~np.isnan(male_distances_m)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_m = np.sum(~np.isnan(male_distances_m), axis=0)
                mean_m = np.where(n_m > 0, np.nanmean(male_distances_m, axis=0), np.nan)
                sem_m  = np.where(n_m > 1, np.nanstd(male_distances_m, axis=0) / np.sqrt(n_m), 0)
            plt.plot(day_numbers, mean_m, '-', color='green', linewidth=2,
                     label=f'Male (n={len(male_distances_m)})')
            plt.fill_between(day_numbers, mean_m - sem_m, mean_m + sem_m, color='green', alpha=0.2)
        if len(female_distances_m) > 0 and np.any(~np.isnan(female_distances_m)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_f = np.sum(~np.isnan(female_distances_m), axis=0)
                mean_f = np.where(n_f > 0, np.nanmean(female_distances_m, axis=0), np.nan)
                sem_f  = np.where(n_f > 1, np.nanstd(female_distances_m, axis=0) / np.sqrt(n_f), 0)
            plt.plot(day_numbers, mean_f, '-', color='purple', linewidth=2,
                     label=f'Female (n={len(female_distances_m)})')
            plt.fill_between(day_numbers, mean_f - sem_f, mean_f + sem_f, color='purple', alpha=0.2)
        plt.title('Sex-Specific Average Distance Per Session')
        plt.xlabel('Training Day')
        plt.ylabel('Distance (m, Mean \u00b1 SEM)')
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

    # Condition-based distance plot
    condition_distance_fig = plt.figure(figsize=(12, 6)) if 'condition_distance' in selected_plots else None
    if condition_distance_fig is not None:
        condition_dist_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_dist_groups:
                condition_dist_groups[condition] = []
            df_r = result['df']
            dist_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['total_distance']):
                    dist_array[session_idx] = row['total_distance'] / 1000.0  # mm → m
            condition_dist_groups[condition].append(dist_array)

        day_numbers = np.arange(0, max_sessions)
        for condition, dist_list in condition_dist_groups.items():
            color = condition_color_map[condition]
            padded = np.array(dist_list)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded), axis=0)
                mean_d = np.where(n_mice > 0, np.nanmean(padded, axis=0), np.nan)
                sem_d  = np.where(n_mice > 1, np.nanstd(padded, axis=0) / np.sqrt(n_mice), 0)
            plt.plot(day_numbers, mean_d, '-', color=color, linewidth=2,
                     label=f'{condition} (n={len(dist_list)})')
            plt.fill_between(day_numbers, mean_d - sem_d, mean_d + sem_d, color=color, alpha=0.2)
        plt.title('Average Distance Per Session by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Distance (m, Mean \u00b1 SEM)')
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

    # Create a new figure for condition-based analysis
    condition_reward_fig = plt.figure(figsize=(12, 6)) if 'condition_reward' in selected_plots else None
    if condition_reward_fig is not None:
        # Group mice by starting condition (session-indexed arrays, no calendar gaps)
        condition_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_groups:
                condition_groups[condition] = []
            df_r = result['df']
            rpm_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['hits']):
                    rpm_array[session_idx] = row['hits'] / row['session_length']
            condition_groups[condition].append(rpm_array)

        # Plot each condition's data
        day_numbers = np.arange(0, max_sessions)
        for condition, rewards_list in condition_groups.items():
            color = condition_color_map[condition]
            padded_rewards = np.array(rewards_list)

            # Calculate mean and SEM (only over mice that have data on that day)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded_rewards), axis=0)
                mean_rewards = np.where(n_mice > 0, np.nanmean(padded_rewards, axis=0), np.nan)
                sem_rewards = np.where(n_mice > 1,
                                       np.nanstd(padded_rewards, axis=0) / np.sqrt(n_mice),
                                       0)

            # Plot the data
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

    # Create a new figure for condition-based speed analysis
    condition_speed_fig = plt.figure(figsize=(12, 6)) if 'condition_speed' in selected_plots else None
    if condition_speed_fig is not None:
        # Group mice by starting condition for speed (session-indexed arrays, no calendar gaps)
        condition_speed_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_speed_groups:
                condition_speed_groups[condition] = []
            df_r = result['df']
            speed_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                speed_array[session_idx] = row['average_speed']
            condition_speed_groups[condition].append(speed_array)

        # Plot each condition's speed data
        day_numbers = np.arange(0, max_sessions)
        for condition, speed_list in condition_speed_groups.items():
            color = condition_color_map[condition]
            padded_speeds = np.array(speed_list)

            # Calculate mean and SEM (only over mice that have data on that day)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded_speeds), axis=0)
                mean_speeds = np.where(n_mice > 0, np.nanmean(padded_speeds, axis=0), np.nan)
                sem_speeds = np.where(n_mice > 1,
                                      np.nanstd(padded_speeds, axis=0) / np.sqrt(n_mice),
                                      0)

            # Plot the data
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

    # Create a new figure for condition-based bout count analysis
    condition_bout_count_fig = plt.figure(figsize=(12, 6)) if 'condition_bout_count' in selected_plots else None
    if condition_bout_count_fig is not None:
        condition_bout_count_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_bout_count_groups:
                condition_bout_count_groups[condition] = []
            df_r = result['df']
            bc_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['bout_count']):
                    bc_array[session_idx] = row['bout_count']
            condition_bout_count_groups[condition].append(bc_array)

        day_numbers = np.arange(0, max_sessions)
        for condition, bc_list in condition_bout_count_groups.items():
            color = condition_color_map[condition]
            padded_bcs = np.array(bc_list)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded_bcs), axis=0)
                mean_bcs = np.where(n_mice > 0, np.nanmean(padded_bcs, axis=0), np.nan)
                sem_bcs  = np.where(n_mice > 1,
                                    np.nanstd(padded_bcs, axis=0) / np.sqrt(n_mice),
                                    0)
            plt.plot(day_numbers, mean_bcs, '-', color=color, linewidth=2,
                     label=f'{condition} (n={len(bc_list)})')
            plt.fill_between(day_numbers, mean_bcs - sem_bcs, mean_bcs + sem_bcs,
                             color=color, alpha=0.2)

        plt.title('Average Bout Count by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Bout Count (Mean \u00b1 SEM)')
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

    # Create a new figure for condition-based average speed per bout analysis
    condition_bout_avg_speed_fig = plt.figure(figsize=(12, 6)) if 'condition_bout_avg_speed' in selected_plots else None
    if condition_bout_avg_speed_fig is not None:
        condition_bas_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_bas_groups:
                condition_bas_groups[condition] = []
            df_r = result['df']
            bas_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['avg_speed_per_bout']):
                    bas_array[session_idx] = row['avg_speed_per_bout']
            condition_bas_groups[condition].append(bas_array)

        day_numbers = np.arange(0, max_sessions)
        for condition, bas_list in condition_bas_groups.items():
            color = condition_color_map[condition]
            padded = np.array(bas_list)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded), axis=0)
                mean_bas = np.where(n_mice > 0, np.nanmean(padded, axis=0), np.nan)
                sem_bas  = np.where(n_mice > 1, np.nanstd(padded, axis=0) / np.sqrt(n_mice), 0)
            plt.plot(day_numbers, mean_bas, '-', color=color, linewidth=2,
                     label=f'{condition} (n={len(bas_list)})')
            plt.fill_between(day_numbers, mean_bas - sem_bas, mean_bas + sem_bas,
                             color=color, alpha=0.2)

        plt.title('Average Speed per Locomotion Bout by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Speed per Bout (cm/s, Mean \u00b1 SEM)')
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

    # Create a new figure for condition-based average distance per bout analysis
    condition_bout_avg_dist_fig = plt.figure(figsize=(12, 6)) if 'condition_bout_avg_dist' in selected_plots else None
    if condition_bout_avg_dist_fig is not None:
        condition_bad_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_bad_groups:
                condition_bad_groups[condition] = []
            df_r = result['df']
            bad_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['avg_dist_per_bout']):
                    bad_array[session_idx] = row['avg_dist_per_bout'] / 1000.0  # mm → m
            condition_bad_groups[condition].append(bad_array)

        day_numbers = np.arange(0, max_sessions)
        for condition, bad_list in condition_bad_groups.items():
            color = condition_color_map[condition]
            padded = np.array(bad_list)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded), axis=0)
                mean_bad = np.where(n_mice > 0, np.nanmean(padded, axis=0), np.nan)
                sem_bad  = np.where(n_mice > 1, np.nanstd(padded, axis=0) / np.sqrt(n_mice), 0)
            plt.plot(day_numbers, mean_bad, '-', color=color, linewidth=2,
                     label=f'{condition} (n={len(bad_list)})')
            plt.fill_between(day_numbers, mean_bad - sem_bad, mean_bad + sem_bad,
                             color=color, alpha=0.2)

        plt.title('Average Distance per Locomotion Bout by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Distance per Bout (m, Mean \u00b1 SEM)')
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

    # Create a new figure for condition-based lick count analysis
    condition_lick_fig = plt.figure(figsize=(12, 6)) if 'condition_lick' in selected_plots else None
    if condition_lick_fig is not None:
        # Group mice by starting condition for lick count (session-indexed arrays, no calendar gaps)
        condition_lick_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_lick_groups:
                condition_lick_groups[condition] = []
            df_r = result['df']
            lick_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                lick_array[session_idx] = row['lick_count']
            condition_lick_groups[condition].append(lick_array)

        # Plot each condition's lick count data
        day_numbers = np.arange(0, max_sessions)
        for condition, lick_list in condition_lick_groups.items():
            color = condition_color_map[condition]
            padded_licks = np.array(lick_list)

            # Calculate mean and SEM (only over mice that have data on that day)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded_licks), axis=0)
                mean_licks = np.where(n_mice > 0, np.nanmean(padded_licks, axis=0), np.nan)
                sem_licks = np.where(n_mice > 1,
                                     np.nanstd(padded_licks, axis=0) / np.sqrt(n_mice),
                                     0)

            # Plot the data
            plt.plot(day_numbers, mean_licks, '-', color=color, linewidth=2,
                    label=f'{condition} (n={len(lick_list)})')
            plt.fill_between(day_numbers, mean_licks - sem_licks, mean_licks + sem_licks,
                            color=color, alpha=0.2)

        # Configure condition-based lick count plot
        plt.title('Average Lick Count by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Number of Licks (Mean \u00b1 SEM)')
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

    # Create a new figure for condition-based lick rate analysis
    condition_lick_rate_fig = plt.figure(figsize=(12, 6)) if 'condition_lick_rate' in selected_plots else None
    if condition_lick_rate_fig is not None:
        condition_lick_rate_groups = {}
        for result in all_results:
            condition = result['starting_condition']
            if condition not in condition_lick_rate_groups:
                condition_lick_rate_groups[condition] = []
            df_r = result['df']
            lpm_array = np.full(max_sessions, np.nan)
            for session_idx, (_, row) in enumerate(df_r.iterrows()):
                if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['lick_count']):
                    lpm_array[session_idx] = row['lick_count'] / row['session_length']
            condition_lick_rate_groups[condition].append(lpm_array)

        day_numbers = np.arange(0, max_sessions)
        for condition, lpm_list in condition_lick_rate_groups.items():
            color = condition_color_map[condition]
            padded_lpms = np.array(lpm_list)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                n_mice = np.sum(~np.isnan(padded_lpms), axis=0)
                mean_lpms = np.where(n_mice > 0, np.nanmean(padded_lpms, axis=0), np.nan)
                sem_lpms  = np.where(n_mice > 1,
                                     np.nanstd(padded_lpms, axis=0) / np.sqrt(n_mice),
                                     0)
            plt.plot(day_numbers, mean_lpms, '-', color=color, linewidth=2,
                     label=f'{condition} (n={len(lpm_list)})')
            plt.fill_between(day_numbers, mean_lpms - sem_lpms, mean_lpms + sem_lpms,
                             color=color, alpha=0.2)

        plt.title('Average Lick Rate by Starting Condition')
        plt.xlabel('Training Day')
        plt.ylabel('Licks per Minute (Mean \u00b1 SEM)')
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

    # Create collapsed condition bar plot (one average rpm per mouse, collapsed across all days)
    condition_bar_fig = None
    if 'condition_bar' in selected_plots:
        condition_bar_fig, ax_bar = plt.subplots(figsize=(8, 6))

        condition_mouse_rpms: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_rpms = []
            for _, row in df_r.iterrows():
                if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['hits']):
                    session_rpms.append(row['hits'] / row['session_length'])
            if session_rpms:
                if condition not in condition_mouse_rpms:
                    condition_mouse_rpms[condition] = []
                condition_mouse_rpms[condition].append((result['mouse'], float(np.mean(session_rpms))))

        conditions_sorted_bar = sorted(condition_mouse_rpms.keys())
        x_pos_bar = np.arange(len(conditions_sorted_bar))

        rng_bar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_bar):
            entries = condition_mouse_rpms[condition]
            mouse_rpms = [v for _, v in entries]
            mean_rpm = float(np.mean(mouse_rpms))
            sem_rpm  = float(np.std(mouse_rpms, ddof=1) / np.sqrt(len(mouse_rpms))) if len(mouse_rpms) > 1 else 0.0
            color = condition_color_map[condition]
            ax_bar.bar(ci, mean_rpm, width=0.5, color=color, alpha=0.8,
                       yerr=sem_rpm, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_bar.random(len(mouse_rpms)) - 0.5) * 0.22
            for j, (mouse_name_bar, rpm_val) in enumerate(entries):
                ax_bar.plot(ci + jitter[j], rpm_val, 'o',
                            color='white', markeredgecolor=color,
                            markeredgewidth=1.8, markersize=7, zorder=3)

        ax_bar.set_xticks(x_pos_bar)
        ax_bar.set_xticklabels(conditions_sorted_bar)
        ax_bar.set_title('Average Reward Rate by Starting Condition\n(collapsed across all sessions)')
        ax_bar.set_xlabel('Starting Condition')
        ax_bar.set_ylabel('Rewards per Minute (Mean \u00b1 SEM)')
        ax_bar.set_ylim(bottom=0)
        ax_bar.tick_params(axis='both', direction='in')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        condition_bar_fig.tight_layout()

    # Create collapsed condition bar plot for speed
    condition_speed_bar_fig = None
    if 'condition_speed_bar' in selected_plots:
        condition_speed_bar_fig, ax_sbar = plt.subplots(figsize=(8, 6))

        condition_mouse_speeds: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_speeds = pd.to_numeric(df_r['average_speed'], errors='coerce').dropna().tolist()
            if session_speeds:
                if condition not in condition_mouse_speeds:
                    condition_mouse_speeds[condition] = []
                condition_mouse_speeds[condition].append((result['mouse'], float(np.mean(session_speeds))))

        conditions_sorted_sbar = sorted(condition_mouse_speeds.keys())
        x_pos_sbar = np.arange(len(conditions_sorted_sbar))

        rng_sbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_sbar):
            entries = condition_mouse_speeds[condition]
            mouse_speeds = [v for _, v in entries]
            mean_spd = float(np.mean(mouse_speeds))
            sem_spd  = float(np.std(mouse_speeds, ddof=1) / np.sqrt(len(mouse_speeds))) if len(mouse_speeds) > 1 else 0.0
            color = condition_color_map[condition]
            ax_sbar.bar(ci, mean_spd, width=0.5, color=color, alpha=0.8,
                        yerr=sem_spd, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_sbar.random(len(mouse_speeds)) - 0.5) * 0.22
            for j, (mouse_name_sbar, spd_val) in enumerate(entries):
                ax_sbar.plot(ci + jitter[j], spd_val, 'o',
                             color='white', markeredgecolor=color,
                             markeredgewidth=1.8, markersize=7, zorder=3)

        ax_sbar.set_xticks(x_pos_sbar)
        ax_sbar.set_xticklabels(conditions_sorted_sbar)
        ax_sbar.set_title('Average Speed by Starting Condition\n(collapsed across all sessions)')
        ax_sbar.set_xlabel('Starting Condition')
        ax_sbar.set_ylabel('Average Speed cm/s (Mean \u00b1 SEM)')
        ax_sbar.set_ylim(bottom=0)
        ax_sbar.tick_params(axis='both', direction='in')
        ax_sbar.spines['top'].set_visible(False)
        ax_sbar.spines['right'].set_visible(False)
        condition_speed_bar_fig.tight_layout()

    # Create collapsed condition bar plot for bout count
    condition_bout_count_bar_fig = None
    if 'condition_bout_count_bar' in selected_plots:
        condition_bout_count_bar_fig, ax_bcbar = plt.subplots(figsize=(8, 6))

        condition_mouse_bcs: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_bcs = pd.to_numeric(df_r['bout_count'], errors='coerce').dropna().tolist()
            if session_bcs:
                if condition not in condition_mouse_bcs:
                    condition_mouse_bcs[condition] = []
                condition_mouse_bcs[condition].append((result['mouse'], float(np.mean(session_bcs))))

        conditions_sorted_bcbar = sorted(condition_mouse_bcs.keys())
        x_pos_bcbar = np.arange(len(conditions_sorted_bcbar))

        rng_bcbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_bcbar):
            entries = condition_mouse_bcs[condition]
            mouse_bcs = [v for _, v in entries]
            mean_bc = float(np.mean(mouse_bcs))
            sem_bc  = float(np.std(mouse_bcs, ddof=1) / np.sqrt(len(mouse_bcs))) if len(mouse_bcs) > 1 else 0.0
            color = condition_color_map[condition]
            ax_bcbar.bar(ci, mean_bc, width=0.5, color=color, alpha=0.8,
                         yerr=sem_bc, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_bcbar.random(len(mouse_bcs)) - 0.5) * 0.22
            for j, (_, bc_val) in enumerate(entries):
                ax_bcbar.plot(ci + jitter[j], bc_val, 'o',
                              color='white', markeredgecolor=color,
                              markeredgewidth=1.8, markersize=7, zorder=3)

        ax_bcbar.set_xticks(x_pos_bcbar)
        ax_bcbar.set_xticklabels(conditions_sorted_bcbar)
        ax_bcbar.set_title('Average Bout Count by Starting Condition\n(collapsed across all sessions)')
        ax_bcbar.set_xlabel('Starting Condition')
        ax_bcbar.set_ylabel('Bout Count (Mean \u00b1 SEM)')
        ax_bcbar.set_ylim(bottom=0)
        ax_bcbar.tick_params(axis='both', direction='in')
        ax_bcbar.spines['top'].set_visible(False)
        ax_bcbar.spines['right'].set_visible(False)
        condition_bout_count_bar_fig.tight_layout()

    # Create collapsed condition bar plot for average speed per bout
    condition_bout_avg_speed_bar_fig = None
    if 'condition_bout_avg_speed_bar' in selected_plots:
        condition_bout_avg_speed_bar_fig, ax_basbar = plt.subplots(figsize=(8, 6))

        condition_mouse_bas: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            vals = pd.to_numeric(df_r['avg_speed_per_bout'], errors='coerce').dropna().tolist()
            if vals:
                if condition not in condition_mouse_bas:
                    condition_mouse_bas[condition] = []
                condition_mouse_bas[condition].append((result['mouse'], float(np.mean(vals))))

        conditions_sorted_basbar = sorted(condition_mouse_bas.keys())
        x_pos_basbar = np.arange(len(conditions_sorted_basbar))
        rng_basbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_basbar):
            entries = condition_mouse_bas[condition]
            mouse_vals = [v for _, v in entries]
            mean_v = float(np.mean(mouse_vals))
            sem_v  = float(np.std(mouse_vals, ddof=1) / np.sqrt(len(mouse_vals))) if len(mouse_vals) > 1 else 0.0
            color = condition_color_map[condition]
            ax_basbar.bar(ci, mean_v, width=0.5, color=color, alpha=0.8,
                          yerr=sem_v, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_basbar.random(len(mouse_vals)) - 0.5) * 0.22
            for j, (_, val) in enumerate(entries):
                ax_basbar.plot(ci + jitter[j], val, 'o',
                               color='white', markeredgecolor=color,
                               markeredgewidth=1.8, markersize=7, zorder=3)
        ax_basbar.set_xticks(x_pos_basbar)
        ax_basbar.set_xticklabels(conditions_sorted_basbar)
        ax_basbar.set_title('Average Speed per Locomotion Bout by Starting Condition\n(collapsed across all sessions)')
        ax_basbar.set_xlabel('Starting Condition')
        ax_basbar.set_ylabel('Speed per Bout (cm/s, Mean \u00b1 SEM)')
        ax_basbar.set_ylim(bottom=0)
        ax_basbar.tick_params(axis='both', direction='in')
        ax_basbar.spines['top'].set_visible(False)
        ax_basbar.spines['right'].set_visible(False)
        condition_bout_avg_speed_bar_fig.tight_layout()

    # Create collapsed condition bar plot for average distance per bout
    condition_bout_avg_dist_bar_fig = None
    if 'condition_bout_avg_dist_bar' in selected_plots:
        condition_bout_avg_dist_bar_fig, ax_badbar = plt.subplots(figsize=(8, 6))

        condition_mouse_bad: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            vals = pd.to_numeric(df_r['avg_dist_per_bout'], errors='coerce').dropna().tolist()
            if vals:
                if condition not in condition_mouse_bad:
                    condition_mouse_bad[condition] = []
                condition_mouse_bad[condition].append(
                    (result['mouse'], float(np.mean(vals)) / 1000.0))  # mm → m

        conditions_sorted_badbar = sorted(condition_mouse_bad.keys())
        x_pos_badbar = np.arange(len(conditions_sorted_badbar))
        rng_badbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_badbar):
            entries = condition_mouse_bad[condition]
            mouse_vals = [v for _, v in entries]
            mean_v = float(np.mean(mouse_vals))
            sem_v  = float(np.std(mouse_vals, ddof=1) / np.sqrt(len(mouse_vals))) if len(mouse_vals) > 1 else 0.0
            color = condition_color_map[condition]
            ax_badbar.bar(ci, mean_v, width=0.5, color=color, alpha=0.8,
                          yerr=sem_v, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_badbar.random(len(mouse_vals)) - 0.5) * 0.22
            for j, (_, val) in enumerate(entries):
                ax_badbar.plot(ci + jitter[j], val, 'o',
                               color='white', markeredgecolor=color,
                               markeredgewidth=1.8, markersize=7, zorder=3)
        ax_badbar.set_xticks(x_pos_badbar)
        ax_badbar.set_xticklabels(conditions_sorted_badbar)
        ax_badbar.set_title('Average Distance per Locomotion Bout by Starting Condition\n(collapsed across all sessions)')
        ax_badbar.set_xlabel('Starting Condition')
        ax_badbar.set_ylabel('Distance per Bout (m, Mean \u00b1 SEM)')
        ax_badbar.set_ylim(bottom=0)
        ax_badbar.tick_params(axis='both', direction='in')
        ax_badbar.spines['top'].set_visible(False)
        ax_badbar.spines['right'].set_visible(False)
        condition_bout_avg_dist_bar_fig.tight_layout()

    # Create collapsed condition bar plot for lick rate
    condition_lick_bar_fig = None
    if 'condition_lick_bar' in selected_plots:
        condition_lick_bar_fig, ax_lbar = plt.subplots(figsize=(8, 6))

        condition_mouse_lpms: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_lpms = []
            for _, row in df_r.iterrows():
                if pd.notna(row['session_length']) and row['session_length'] > 0 and pd.notna(row['lick_count']):
                    session_lpms.append(row['lick_count'] / row['session_length'])
            if session_lpms:
                if condition not in condition_mouse_lpms:
                    condition_mouse_lpms[condition] = []
                condition_mouse_lpms[condition].append((result['mouse'], float(np.mean(session_lpms))))

        conditions_sorted_lbar = sorted(condition_mouse_lpms.keys())
        x_pos_lbar = np.arange(len(conditions_sorted_lbar))

        rng_lbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_lbar):
            entries = condition_mouse_lpms[condition]
            mouse_lpms = [v for _, v in entries]
            mean_lpm = float(np.mean(mouse_lpms))
            sem_lpm  = float(np.std(mouse_lpms, ddof=1) / np.sqrt(len(mouse_lpms))) if len(mouse_lpms) > 1 else 0.0
            color = condition_color_map[condition]
            ax_lbar.bar(ci, mean_lpm, width=0.5, color=color, alpha=0.8,
                        yerr=sem_lpm, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_lbar.random(len(mouse_lpms)) - 0.5) * 0.22
            for j, (mouse_name_lbar, lpm_val) in enumerate(entries):
                ax_lbar.plot(ci + jitter[j], lpm_val, 'o',
                             color='white', markeredgecolor=color,
                             markeredgewidth=1.8, markersize=7, zorder=3)

        ax_lbar.set_xticks(x_pos_lbar)
        ax_lbar.set_xticklabels(conditions_sorted_lbar)
        ax_lbar.set_title('Average Lick Rate by Starting Condition\n(collapsed across all sessions)')
        ax_lbar.set_xlabel('Starting Condition')
        ax_lbar.set_ylabel('Licks per Minute (Mean \u00b1 SEM)')
        ax_lbar.set_ylim(bottom=0)
        ax_lbar.tick_params(axis='both', direction='in')
        ax_lbar.spines['top'].set_visible(False)
        ax_lbar.spines['right'].set_visible(False)
        condition_lick_bar_fig.tight_layout()

    # Create collapsed condition bar plot for average distance per session (mm → m)
    condition_distance_bar_fig = None
    if 'condition_distance_bar' in selected_plots:
        condition_distance_bar_fig, ax_dbar = plt.subplots(figsize=(8, 6))

        condition_mouse_dist_avg: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_dists = pd.to_numeric(df_r['total_distance'], errors='coerce').dropna().tolist()
            if session_dists:
                if condition not in condition_mouse_dist_avg:
                    condition_mouse_dist_avg[condition] = []
                condition_mouse_dist_avg[condition].append(
                    (result['mouse'], float(np.mean(session_dists)) / 1000.0))  # mm → m

        conditions_sorted_dbar = sorted(condition_mouse_dist_avg.keys())
        x_pos_dbar = np.arange(len(conditions_sorted_dbar))

        rng_dbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_dbar):
            entries = condition_mouse_dist_avg[condition]
            mouse_dists = [v for _, v in entries]
            mean_d = float(np.mean(mouse_dists))
            sem_d  = float(np.std(mouse_dists, ddof=1) / np.sqrt(len(mouse_dists))) if len(mouse_dists) > 1 else 0.0
            color = condition_color_map[condition]
            ax_dbar.bar(ci, mean_d, width=0.5, color=color, alpha=0.8,
                        yerr=sem_d, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_dbar.random(len(mouse_dists)) - 0.5) * 0.22
            for j, (_, dist_val) in enumerate(entries):
                ax_dbar.plot(ci + jitter[j], dist_val, 'o',
                             color='white', markeredgecolor=color,
                             markeredgewidth=1.8, markersize=7, zorder=3)

        ax_dbar.set_xticks(x_pos_dbar)
        ax_dbar.set_xticklabels(conditions_sorted_dbar)
        ax_dbar.set_title('Average Distance Per Session by Starting Condition\n(collapsed across all sessions)')
        ax_dbar.set_xlabel('Starting Condition')
        ax_dbar.set_ylabel('Average Distance per Session (m, Mean \u00b1 SEM)')
        ax_dbar.set_ylim(bottom=0)
        ax_dbar.tick_params(axis='both', direction='in')
        ax_dbar.spines['top'].set_visible(False)
        ax_dbar.spines['right'].set_visible(False)
        condition_distance_bar_fig.tight_layout()

    # Create collapsed condition bar plot for total distance (sum across all sessions, mm → m)
    total_distance_bar_fig = None
    if 'total_distance_bar' in selected_plots:
        total_distance_bar_fig, ax_tbar = plt.subplots(figsize=(8, 6))

        condition_mouse_dist_total: dict[str, list] = {}
        for result in all_results:
            condition = result['starting_condition']
            df_r = result['df']
            session_dists = pd.to_numeric(df_r['total_distance'], errors='coerce').dropna().tolist()
            if session_dists:
                if condition not in condition_mouse_dist_total:
                    condition_mouse_dist_total[condition] = []
                condition_mouse_dist_total[condition].append(
                    (result['mouse'], float(np.sum(session_dists)) / 1000.0))  # mm → m

        conditions_sorted_tbar = sorted(condition_mouse_dist_total.keys())
        x_pos_tbar = np.arange(len(conditions_sorted_tbar))

        rng_tbar = np.random.default_rng(seed=42)
        for ci, condition in enumerate(conditions_sorted_tbar):
            entries = condition_mouse_dist_total[condition]
            mouse_totals = [v for _, v in entries]
            mean_t = float(np.mean(mouse_totals))
            sem_t  = float(np.std(mouse_totals, ddof=1) / np.sqrt(len(mouse_totals))) if len(mouse_totals) > 1 else 0.0
            color = condition_color_map[condition]
            ax_tbar.bar(ci, mean_t, width=0.5, color=color, alpha=0.8,
                        yerr=sem_t, capsize=7, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            jitter = (rng_tbar.random(len(mouse_totals)) - 0.5) * 0.22
            for j, (_, total_val) in enumerate(entries):
                ax_tbar.plot(ci + jitter[j], total_val, 'o',
                             color='white', markeredgecolor=color,
                             markeredgewidth=1.8, markersize=7, zorder=3)

        ax_tbar.set_xticks(x_pos_tbar)
        ax_tbar.set_xticklabels(conditions_sorted_tbar)
        ax_tbar.set_title('Total Distance Traveled per Mouse by Starting Condition\n(summed across all sessions)')
        ax_tbar.set_xlabel('Starting Condition')
        ax_tbar.set_ylabel('Total Distance (m, Mean \u00b1 SEM)')
        ax_tbar.set_ylim(bottom=0)
        ax_tbar.tick_params(axis='both', direction='in')
        ax_tbar.spines['top'].set_visible(False)
        ax_tbar.spines['right'].set_visible(False)
        total_distance_bar_fig.tight_layout()

    # ── Behavioral epoch plots ────────────────────────────────────────────────
    epoch_speed_per_mouse_fig            = epoch_speed_cond_fig            = None
    epoch_cap_per_mouse_fig              = epoch_cap_cond_fig              = None
    epoch_speed_sess_per_mouse_fig       = epoch_speed_sess_cond_fig       = None
    epoch_cap_sess_per_mouse_fig         = epoch_cap_sess_cond_fig         = None
    epoch_speed_early_per_mouse_fig    = epoch_speed_late_per_mouse_fig    = None
    epoch_speed_early_cond_fig         = epoch_speed_late_cond_fig         = None
    epoch_cap_early_per_mouse_fig      = epoch_cap_late_per_mouse_fig      = None
    epoch_cap_early_cond_fig           = epoch_cap_late_cond_fig           = None
    epoch_speed_early_ev_per_mouse_fig = epoch_speed_late_ev_per_mouse_fig = None
    epoch_speed_early_ev_cond_fig      = epoch_speed_late_ev_cond_fig      = None
    epoch_cap_early_ev_per_mouse_fig   = epoch_cap_late_ev_per_mouse_fig   = None
    epoch_cap_early_ev_cond_fig        = epoch_cap_late_ev_cond_fig        = None
    _epoch_keys = {'epoch_reward_speed', 'epoch_reward_cap',
                   'epoch_reward_speed_sess', 'epoch_reward_cap_sess',
                   'epoch_reward_speed_early_late', 'epoch_reward_cap_early_late',
                   'epoch_reward_speed_early_late_ev', 'epoch_reward_cap_early_late_ev'}
    if _epoch_keys & set(selected_plots):
        _any_speed = any(r.get('speed_epoch_matrix') is not None for r in all_results)
        _any_cap   = any(r.get('cap_epoch_matrix')   is not None for r in all_results)

        if 'epoch_reward_speed' in selected_plots and _any_speed:
            epoch_speed_per_mouse_fig, epoch_speed_cond_fig = _plot_epoch_panels(
                all_results, 'speed_epoch_matrix',
                ylabel='Treadmill Speed (cm/s)',
                title_prefix='Speed Aligned to Reward Zone Entry',
                condition_color_map=condition_color_map,
                hierarchy='event',
            )

        if 'epoch_reward_cap' in selected_plots and _any_cap:
            epoch_cap_per_mouse_fig, epoch_cap_cond_fig = _plot_epoch_panels(
                all_results, 'cap_epoch_matrix',
                ylabel='Capacitive Value (z-score)',
                title_prefix='Capacitive Value Aligned to Reward Zone Entry',
                condition_color_map=condition_color_map,
                hierarchy='event',
            )

        if 'epoch_reward_speed_sess' in selected_plots and _any_speed:
            epoch_speed_sess_per_mouse_fig, epoch_speed_sess_cond_fig = _plot_epoch_panels(
                all_results, 'speed_epoch_session_means',
                ylabel='Treadmill Speed (cm/s)',
                title_prefix='Speed Aligned to Reward Zone Entry',
                condition_color_map=condition_color_map,
                hierarchy='session',
            )

        if 'epoch_reward_cap_sess' in selected_plots and _any_cap:
            epoch_cap_sess_per_mouse_fig, epoch_cap_sess_cond_fig = _plot_epoch_panels(
                all_results, 'cap_epoch_session_means',
                ylabel='Capacitive Value (z-score)',
                title_prefix='Capacitive Value Aligned to Reward Zone Entry',
                condition_color_map=condition_color_map,
                hierarchy='session',
            )

        if 'epoch_reward_speed_early_late' in selected_plots and _any_speed:
            (epoch_speed_early_per_mouse_fig, epoch_speed_late_per_mouse_fig,
             epoch_speed_early_cond_fig,      epoch_speed_late_cond_fig) = \
                _plot_epoch_early_late_panels(
                    all_results, 'speed_epoch_session_means',
                    ylabel='Treadmill Speed (cm/s)',
                    title_prefix='Speed Aligned to Reward Zone Entry',
                    condition_color_map=condition_color_map,
                )

        if 'epoch_reward_cap_early_late' in selected_plots and _any_cap:
            (epoch_cap_early_per_mouse_fig, epoch_cap_late_per_mouse_fig,
             epoch_cap_early_cond_fig,      epoch_cap_late_cond_fig) = \
                _plot_epoch_early_late_panels(
                    all_results, 'cap_epoch_session_means',
                    ylabel='Capacitive Value (z-score)',
                    title_prefix='Capacitive Value Aligned to Reward Zone Entry',
                    condition_color_map=condition_color_map,
                )

        if 'epoch_reward_speed_early_late_ev' in selected_plots and _any_speed:
            (epoch_speed_early_ev_per_mouse_fig, epoch_speed_late_ev_per_mouse_fig,
             epoch_speed_early_ev_cond_fig,      epoch_speed_late_ev_cond_fig) = \
                _plot_epoch_early_late_panels(
                    all_results, 'speed_epoch_matrix',
                    ylabel='Treadmill Speed (cm/s)',
                    title_prefix='Speed Aligned to Reward Zone Entry',
                    condition_color_map=condition_color_map,
                    indices_key='speed_epoch_event_indices',
                    row_unit='events',
                )

        if 'epoch_reward_cap_early_late_ev' in selected_plots and _any_cap:
            (epoch_cap_early_ev_per_mouse_fig, epoch_cap_late_ev_per_mouse_fig,
             epoch_cap_early_ev_cond_fig,      epoch_cap_late_ev_cond_fig) = \
                _plot_epoch_early_late_panels(
                    all_results, 'cap_epoch_matrix',
                    ylabel='Capacitive Value (z-score)',
                    title_prefix='Capacitive Value Aligned to Reward Zone Entry',
                    condition_color_map=condition_color_map,
                    indices_key='cap_epoch_event_indices',
                    row_unit='events',
                )

    # Create the level-based analysis plots
    level_reward_fig = level_speed_collapsed_fig = level_speed_condition_fig = None
    level_lick_collapsed_fig = level_lick_condition_fig = None
    level_dist_collapsed_fig = level_dist_condition_fig = level_dist_condition_excl_last_fig = None
    level_bout_collapsed_fig = level_bout_condition_fig = None
    level_bout_avg_speed_collapsed_fig = level_bout_avg_speed_condition_fig = None
    level_bout_avg_dist_collapsed_fig = level_bout_avg_dist_condition_fig = None
    _level_stats_data = None
    if transitions_csv_path or any(k in selected_plots for k in ('levels', 'level_speed', 'level_speed_condition',
                                                                   'level_lick', 'level_lick_condition',
                                                                   'level_dist', 'level_dist_condition',
                                                                   'level_dist_condition_excl_last',
                                                                   'level_bout', 'level_bout_condition',
                                                                   'level_bout_avg_speed', 'level_bout_avg_speed_condition',
                                                                   'level_bout_avg_dist', 'level_bout_avg_dist_condition')):
        level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, \
        level_lick_collapsed_fig, level_lick_condition_fig, \
        level_dist_collapsed_fig, level_dist_condition_fig, \
        level_dist_condition_excl_last_fig, \
        level_bout_collapsed_fig, level_bout_condition_fig, \
        level_bout_avg_speed_collapsed_fig, level_bout_avg_speed_condition_fig, \
        level_bout_avg_dist_collapsed_fig, level_bout_avg_dist_condition_fig, \
        _level_stats_data = analyze_levels(
            data_files, transitions_csv_path, animal_conditions=conditions,
            selected_plots=selected_plots,
        )

    # ── Missing data report ───────────────────────────────────────────────────
    all_session_dates = set(
        d.date() for result in all_results for d in result['df']['date']
    )

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MISSING DATA REPORT")
    report_lines.append(f"Global date range: {global_start}  →  {global_end}  ({total_days} calendar days)")
    report_lines.append(f"Session dates across all mice: {len(all_session_dates)}")
    report_lines.append("=" * 70)

    for result in all_results:
        mouse_name  = result['mouse']
        mouse_dates = set(d.date() for d in result['df']['date'])
        file_errors = result.get('session_file_errors', {})
        error_dates = set(datetime.strptime(ds, '%Y-%m-%d').date() for ds in file_errors)
        fully_missing = sorted(all_session_dates - mouse_dates - error_dates)
        present       = sorted(mouse_dates)

        report_lines.append(f"\nMouse: {mouse_name}")
        report_lines.append(f"  Sessions present   : {len(present)}")
        for d in present:
            day_offset = (d - global_start).days
            report_lines.append(f"    Day {day_offset:>3}  {d}")

        if file_errors:
            report_lines.append(f"  Incomplete sessions (missing file): {len(file_errors)}")
            for date_str, missing_types in sorted(file_errors.items()):
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                day_offset = (d - global_start).days
                report_lines.append(f"    Day {day_offset:>3}  {date_str}  <-- MISSING FILE(S): {', '.join(missing_types)}")

        if fully_missing:
            report_lines.append(f"  Fully absent dates : {len(fully_missing)}")
            for d in fully_missing:
                day_offset = (d - global_start).days
                report_lines.append(f"    Day {day_offset:>3}  {d}  <-- NO SESSION")

        if not file_errors and not fully_missing:
            report_lines.append(f"  Missing sessions   : none")

    report_lines.append("\n" + "=" * 70)
    report_text = "\n".join(report_lines)

    print("\n" + report_text)

    report_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(data_files[0]))
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"missing_data_report_{timestamp_str}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nMissing data report saved to: {report_path}")

    return speed_fig, sensitivity_fig, lick_fig, reward_fig, false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig, avg_reward_fig, sex_reward_fig, distance_fig, bout_count_fig, avg_bout_count_fig, bout_avg_speed_fig, bout_avg_dist_fig, sex_distance_fig, condition_distance_fig, condition_distance_bar_fig, total_distance_bar_fig, avg_lick_rate_fig, sex_lick_rate_fig, condition_reward_fig, condition_speed_fig, condition_bout_count_fig, condition_bout_avg_speed_fig, condition_bout_avg_dist_fig, condition_lick_fig, condition_lick_rate_fig, condition_bar_fig, condition_speed_bar_fig, condition_bout_count_bar_fig, condition_bout_avg_speed_bar_fig, condition_bout_avg_dist_bar_fig, condition_lick_bar_fig, level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig, level_bout_collapsed_fig, level_bout_condition_fig, level_bout_avg_speed_collapsed_fig, level_bout_avg_speed_condition_fig, level_bout_avg_dist_collapsed_fig, level_bout_avg_dist_condition_fig, epoch_speed_per_mouse_fig, epoch_speed_cond_fig, epoch_cap_per_mouse_fig, epoch_cap_cond_fig, epoch_speed_sess_per_mouse_fig, epoch_speed_sess_cond_fig, epoch_cap_sess_per_mouse_fig, epoch_cap_sess_cond_fig, epoch_speed_early_per_mouse_fig, epoch_speed_late_per_mouse_fig, epoch_speed_early_cond_fig, epoch_speed_late_cond_fig, epoch_cap_early_per_mouse_fig, epoch_cap_late_per_mouse_fig, epoch_cap_early_cond_fig, epoch_cap_late_cond_fig, epoch_speed_early_ev_per_mouse_fig, epoch_speed_late_ev_per_mouse_fig, epoch_speed_early_ev_cond_fig, epoch_speed_late_ev_cond_fig, epoch_cap_early_ev_per_mouse_fig, epoch_cap_late_ev_per_mouse_fig, epoch_cap_early_ev_cond_fig, epoch_cap_late_ev_cond_fig, all_results, _level_stats_data

def _ask_mode(root):
    """Ask whether to generate plots or run the descriptive stats report.
    Returns 'plots', 'stats', or None if the dialog is dismissed."""
    result = [None]
    dialog = tk.Toplevel(root)
    dialog.title('Select Mode')
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text='What would you like to do?',
             font=('Arial', 11, 'bold')).pack(padx=20, pady=(16, 8))

    def _choose(mode):
        result[0] = mode
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(padx=20, pady=(4, 16))
    tk.Button(btn_frame, text='Generate Plots', width=20,
              command=lambda: _choose('plots')).pack(side='left', padx=6)
    tk.Button(btn_frame, text='Descriptive Stats Report', width=24,
              command=lambda: _choose('stats')).pack(side='left', padx=6)

    dialog.update_idletasks()
    dialog.geometry(
        f"+{root.winfo_screenwidth() // 2 - dialog.winfo_reqwidth() // 2}"
        f"+{root.winfo_screenheight() // 2 - dialog.winfo_reqheight() // 2}"
    )
    root.wait_window(dialog)
    return result[0]


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
    
    if not file_paths:
        print("No file selected. Exiting...")
        return

    # ── Mode selection ────────────────────────────────────────────────────────
    mode = _ask_mode(root)
    if mode is None:
        print("No mode selected. Exiting...")
        return

    # ── Extract markers and starting conditions (shared by both modes) ────────
    markers = []
    starting_conditions = []
    for file_path in file_paths:
        mouse_name = os.path.basename(file_path).split("_")[0]
        if mouse_name in animal_info:
            sex = animal_info[mouse_name]['sex']
            marker = 's' if sex == 'male' else 'o'
            markers.append(marker)
            starting_conditions.append(animal_info[mouse_name]['starting_condition'])
            print(f"{mouse_name}: sex={sex}, marker={marker}, condition={animal_info[mouse_name]['starting_condition']}")
        else:
            print(f"Warning: {mouse_name} not found in master CSV file. Skipping...")
            continue

    # ── STATS mode ────────────────────────────────────────────────────────────
    if mode == 'stats':
        transitions_csv_path = filedialog.askopenfilename(
            title='Select transitions CSV for level stats — cancel to skip level sheets',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            initialdir=os.path.dirname(file_paths[0]),
        ) or None
        if transitions_csv_path:
            print(f"Transitions CSV: {os.path.basename(transitions_csv_path)}")
        else:
            print("No transitions CSV selected — level sheets will be omitted.")

        output_dir = filedialog.askdirectory(
            title='Select output folder for the stats report',
            initialdir=os.path.dirname(file_paths[0]),
        ) or None

        *_, all_results, _level_stats_data = analyze_mouse_data(
            file_paths, markers, starting_conditions,
            transitions_csv_path=transitions_csv_path,
            selected_plots=frozenset(),
        )
        generate_descriptive_stats_report(all_results, _level_stats_data, output_dir=output_dir)
        return

    # ── PLOTS mode ────────────────────────────────────────────────────────────
    selected_plots = _ask_plot_selection(root)
    if not selected_plots:
        print("No plots selected. Exiting...")
        return

    # Select transitions CSV only if a level plot was requested
    transitions_csv_path = None
    if any(k in selected_plots for k in ('levels', 'level_speed', 'level_speed_condition',
                                          'level_lick', 'level_lick_condition',
                                          'level_dist', 'level_dist_condition',
                                          'level_dist_condition_excl_last',
                                          'level_bout', 'level_bout_condition',
                                          'level_bout_avg_speed', 'level_bout_avg_speed_condition',
                                          'level_bout_avg_dist', 'level_bout_avg_dist_condition')):
        transitions_csv_path = filedialog.askopenfilename(
            title='Select transitions CSV (from level_sorter.py) — cancel to skip level plot',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            initialdir=os.path.dirname(file_paths[0]),
        ) or None
        if transitions_csv_path:
            print(f"Transitions CSV: {os.path.basename(transitions_csv_path)}")
        else:
            print("No transitions CSV selected — level plot will be empty.")

    # Analyze data and plot results
    speed_fig, sensitivity_fig, lick_fig, reward_fig, false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig, avg_reward_fig, sex_reward_fig, distance_fig, bout_count_fig, avg_bout_count_fig, bout_avg_speed_fig, bout_avg_dist_fig, sex_distance_fig, condition_distance_fig, condition_distance_bar_fig, total_distance_bar_fig, avg_lick_rate_fig, sex_lick_rate_fig, condition_reward_fig, condition_speed_fig, condition_bout_count_fig, condition_bout_avg_speed_fig, condition_bout_avg_dist_fig, condition_lick_fig, condition_lick_rate_fig, condition_bar_fig, condition_speed_bar_fig, condition_bout_count_bar_fig, condition_bout_avg_speed_bar_fig, condition_bout_avg_dist_bar_fig, condition_lick_bar_fig, level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig, level_bout_collapsed_fig, level_bout_condition_fig, level_bout_avg_speed_collapsed_fig, level_bout_avg_speed_condition_fig, level_bout_avg_dist_collapsed_fig, level_bout_avg_dist_condition_fig, epoch_speed_per_mouse_fig, epoch_speed_cond_fig, epoch_cap_per_mouse_fig, epoch_cap_cond_fig, epoch_speed_sess_per_mouse_fig, epoch_speed_sess_cond_fig, epoch_cap_sess_per_mouse_fig, epoch_cap_sess_cond_fig, epoch_speed_early_per_mouse_fig, epoch_speed_late_per_mouse_fig, epoch_speed_early_cond_fig, epoch_speed_late_cond_fig, epoch_cap_early_per_mouse_fig, epoch_cap_late_per_mouse_fig, epoch_cap_early_cond_fig, epoch_cap_late_cond_fig, epoch_speed_early_ev_per_mouse_fig, epoch_speed_late_ev_per_mouse_fig, epoch_speed_early_ev_cond_fig, epoch_speed_late_ev_cond_fig, epoch_cap_early_ev_per_mouse_fig, epoch_cap_late_ev_per_mouse_fig, epoch_cap_early_ev_cond_fig, epoch_cap_late_ev_cond_fig, all_results, _level_stats_data = analyze_mouse_data(
        file_paths, markers, starting_conditions,
        transitions_csv_path=transitions_csv_path,
        selected_plots=selected_plots,
    )

    # All generated figures (None entries are skipped)
    all_figs = [f for f in [
        speed_fig, sensitivity_fig, lick_fig, reward_fig,
        false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig,
        avg_reward_fig, sex_reward_fig,
        distance_fig, bout_count_fig, avg_bout_count_fig, bout_avg_speed_fig, bout_avg_dist_fig, sex_distance_fig, condition_distance_fig,
        condition_distance_bar_fig, total_distance_bar_fig,
        avg_lick_rate_fig, sex_lick_rate_fig,
        condition_reward_fig, condition_speed_fig, condition_bout_count_fig, condition_bout_avg_speed_fig, condition_bout_avg_dist_fig, condition_lick_fig, condition_lick_rate_fig,
        condition_bar_fig, condition_speed_bar_fig, condition_bout_count_bar_fig, condition_bout_avg_speed_bar_fig, condition_bout_avg_dist_bar_fig, condition_lick_bar_fig,
        level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig,
        level_lick_collapsed_fig, level_lick_condition_fig,
        level_dist_collapsed_fig, level_dist_condition_fig,
        level_dist_condition_excl_last_fig,
        level_bout_collapsed_fig, level_bout_condition_fig,
        level_bout_avg_speed_collapsed_fig, level_bout_avg_speed_condition_fig,
        level_bout_avg_dist_collapsed_fig, level_bout_avg_dist_condition_fig,
        epoch_speed_per_mouse_fig, epoch_speed_cond_fig,
        epoch_cap_per_mouse_fig, epoch_cap_cond_fig,
        epoch_speed_sess_per_mouse_fig, epoch_speed_sess_cond_fig,
        epoch_cap_sess_per_mouse_fig, epoch_cap_sess_cond_fig,
        epoch_speed_early_per_mouse_fig, epoch_speed_late_per_mouse_fig,
        epoch_speed_early_cond_fig, epoch_speed_late_cond_fig,
        epoch_cap_early_per_mouse_fig, epoch_cap_late_per_mouse_fig,
        epoch_cap_early_cond_fig, epoch_cap_late_cond_fig,
        epoch_speed_early_ev_per_mouse_fig, epoch_speed_late_ev_per_mouse_fig,
        epoch_speed_early_ev_cond_fig, epoch_speed_late_ev_cond_fig,
        epoch_cap_early_ev_per_mouse_fig, epoch_cap_late_ev_per_mouse_fig,
        epoch_cap_early_ev_cond_fig, epoch_cap_late_ev_cond_fig,
    ] if f is not None]

    # Configure all figures (add legend only when labeled artists exist, then tight layout)
    for fig in all_figs:
        plt.figure(fig.number)
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            if len(file_paths) > 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.subplots_adjust(right=0.85)
            else:
                plt.legend()
        plt.tight_layout()

    # Display all plots
    for fig in all_figs:
        fig.show()
    plt.show()

    # Save all selected plots automatically
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'

    # Plot configurations to save (skip None figures)
    plot_configs = [
        (speed_fig,             'speed',               'Speed plot'),
        (sensitivity_fig,       'sensitivity',         'Sensitivity plot'),
        (lick_fig,              'lick_count',          'Lick count plot'),
        (reward_fig,            'reward_count',        'Reward count plot'),
        (false_alarm_fig,       'false_alarms',        'False alarms plot'),
        (correct_rejection_fig, 'correct_rejections',  'Correct rejections plot'),
        (specificity_fig,       'specificity',         'Specificity plot'),
        (dprime_fig,            'dprime',              "d' plot"),
        (avg_reward_fig,        'avg_reward',          'Average rewards plot'),
        (sex_reward_fig,        'sex_reward',          'Sex-specific average rewards plot'),
        (distance_fig,          'distance',            'Distance per session plot'),
        (bout_count_fig,         'bout_count',          'Locomotion bout count plot'),
        (avg_bout_count_fig,     'avg_bout_count',      'Average bout count across all mice plot'),
        (bout_avg_speed_fig,     'bout_avg_speed',      'Average speed per locomotion bout plot'),
        (bout_avg_dist_fig,      'bout_avg_dist',       'Average distance per locomotion bout plot'),
        (sex_distance_fig,      'sex_distance',        'Sex-specific distance per session plot'),
        (condition_distance_fig,'condition_distance',  'Condition-based distance per session plot'),
        (condition_distance_bar_fig, 'condition_distance_bar', 'Condition distance collapsed bar chart'),
        (total_distance_bar_fig,     'total_distance_bar',     'Total distance per mouse collapsed bar chart'),
        (avg_lick_rate_fig,     'avg_lick_rate',       'Average lick rate plot'),
        (sex_lick_rate_fig,     'sex_lick_rate',       'Sex-specific lick rate plot'),
        (condition_reward_fig,  'condition_reward',    'Condition-based average rewards plot'),
        (condition_speed_fig,   'condition_speed',     'Condition-based average speed plot'),
        (condition_bout_count_fig, 'condition_bout_count', 'Condition-based bout count plot'),
        (condition_bout_avg_speed_fig, 'condition_bout_avg_speed', 'Condition-based avg speed per bout plot'),
        (condition_bout_avg_dist_fig,  'condition_bout_avg_dist',  'Condition-based avg distance per bout plot'),
        (condition_lick_fig,    'condition_lick',      'Condition-based average lick count plot'),
        (condition_lick_rate_fig, 'condition_lick_rate', 'Condition-based lick rate plot'),
        (condition_bar_fig,      'condition_bar',       'Condition collapsed bar chart'),
        (condition_speed_bar_fig,'condition_speed_bar', 'Condition speed collapsed bar chart'),
        (condition_bout_count_bar_fig, 'condition_bout_count_bar', 'Condition bout count collapsed bar chart'),
        (condition_bout_avg_speed_bar_fig, 'condition_bout_avg_speed_bar', 'Condition avg speed per bout bar chart'),
        (condition_bout_avg_dist_bar_fig,  'condition_bout_avg_dist_bar',  'Condition avg distance per bout bar chart'),
        (condition_lick_bar_fig, 'condition_lick_bar',  'Condition lick rate collapsed bar chart'),
        (level_reward_fig,              'level_reward',          'Level-based average rewards plot'),
        (level_speed_collapsed_fig,     'level_speed',           'Level-based average speed — collapsed'),
        (level_speed_condition_fig,     'level_speed_condition', 'Level-based average speed — by condition'),
        (level_lick_collapsed_fig,      'level_lick',            'Level-based average lick rate — collapsed'),
        (level_lick_condition_fig,      'level_lick_condition',  'Level-based average lick rate — by condition'),
        (level_dist_collapsed_fig,      'level_dist',            'Level-based distance — collapsed'),
        (level_dist_condition_fig,      'level_dist_condition',  'Level-based distance — by condition'),
        (level_dist_condition_excl_last_fig, 'level_dist_condition_excl_last', 'Level-based distance — by condition, last level excluded'),
        (level_bout_collapsed_fig,   'level_bout',           'Level-based bout count — collapsed'),
        (level_bout_condition_fig,   'level_bout_condition', 'Level-based bout count — by condition'),
        (level_bout_avg_speed_collapsed_fig,  'level_bout_avg_speed',           'Level-based avg speed per bout — collapsed'),
        (level_bout_avg_speed_condition_fig,  'level_bout_avg_speed_condition', 'Level-based avg speed per bout — by condition'),
        (level_bout_avg_dist_collapsed_fig,   'level_bout_avg_dist',            'Level-based avg distance per bout — collapsed'),
        (level_bout_avg_dist_condition_fig,   'level_bout_avg_dist_condition',  'Level-based avg distance per bout — by condition'),
        (epoch_speed_per_mouse_fig,      'epoch_reward_speed_per_mouse',           'Speed epoch (event) — per mouse'),
        (epoch_speed_cond_fig,           'epoch_reward_speed_condition',           'Speed epoch (event) — by condition'),
        (epoch_cap_per_mouse_fig,        'epoch_reward_cap_per_mouse',             'Capacitive epoch (event) — per mouse'),
        (epoch_cap_cond_fig,             'epoch_reward_cap_condition',             'Capacitive epoch (event) — by condition'),
        (epoch_speed_sess_per_mouse_fig, 'epoch_reward_speed_sess_per_mouse',      'Speed epoch (session) — per mouse'),
        (epoch_speed_sess_cond_fig,      'epoch_reward_speed_sess_condition',      'Speed epoch (session) — by condition'),
        (epoch_cap_sess_per_mouse_fig,   'epoch_reward_cap_sess_per_mouse',        'Capacitive epoch (session) — per mouse'),
        (epoch_cap_sess_cond_fig,        'epoch_reward_cap_sess_condition',        'Capacitive epoch (session) — by condition'),
        (epoch_speed_early_per_mouse_fig, 'epoch_reward_speed_early_per_mouse',     'Speed epoch (early sessions) — per mouse'),
        (epoch_speed_early_cond_fig,      'epoch_reward_speed_early_condition',     'Speed epoch (early sessions) — by condition'),
        (epoch_speed_late_per_mouse_fig,  'epoch_reward_speed_late_per_mouse',      'Speed epoch (late sessions) — per mouse'),
        (epoch_speed_late_cond_fig,       'epoch_reward_speed_late_condition',      'Speed epoch (late sessions) — by condition'),
        (epoch_cap_early_per_mouse_fig,   'epoch_reward_cap_early_per_mouse',       'Capacitive epoch (early sessions) — per mouse'),
        (epoch_cap_early_cond_fig,        'epoch_reward_cap_early_condition',       'Capacitive epoch (early sessions) — by condition'),
        (epoch_cap_late_per_mouse_fig,    'epoch_reward_cap_late_per_mouse',        'Capacitive epoch (late sessions) — per mouse'),
        (epoch_cap_late_cond_fig,         'epoch_reward_cap_late_condition',        'Capacitive epoch (late sessions) — by condition'),
        (epoch_speed_early_ev_per_mouse_fig, 'epoch_reward_speed_early_ev_per_mouse', 'Speed epoch ev (early sessions) — per mouse'),
        (epoch_speed_early_ev_cond_fig,      'epoch_reward_speed_early_ev_condition', 'Speed epoch ev (early sessions) — by condition'),
        (epoch_speed_late_ev_per_mouse_fig,  'epoch_reward_speed_late_ev_per_mouse',  'Speed epoch ev (late sessions) — per mouse'),
        (epoch_speed_late_ev_cond_fig,       'epoch_reward_speed_late_ev_condition',  'Speed epoch ev (late sessions) — by condition'),
        (epoch_cap_early_ev_per_mouse_fig,   'epoch_reward_cap_early_ev_per_mouse',   'Capacitive epoch ev (early sessions) — per mouse'),
        (epoch_cap_early_ev_cond_fig,        'epoch_reward_cap_early_ev_condition',   'Capacitive epoch ev (early sessions) — by condition'),
        (epoch_cap_late_ev_per_mouse_fig,    'epoch_reward_cap_late_ev_per_mouse',    'Capacitive epoch ev (late sessions) — per mouse'),
        (epoch_cap_late_ev_cond_fig,         'epoch_reward_cap_late_ev_condition',    'Capacitive epoch ev (late sessions) — by condition'),
    ]

    for fig, name, title in plot_configs:
        if fig is None:
            continue
        save_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            title=f"Save {title} as",
            initialfile=f"mouse_{name}_comparison_{len(file_paths)}mice.svg"
        )
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', format='svg')
            print(f"{title} saved to: {save_path}")
if __name__ == "__main__":
    main()