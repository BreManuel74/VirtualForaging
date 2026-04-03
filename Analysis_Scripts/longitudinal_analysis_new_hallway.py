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

# Add Analysis_Scripts to path to import lick detection algorithm
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import lick_detection_algorithm as lda
from scipy.stats import norm

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
        return plt.figure(figsize=(15, 8)), None, None, None, None, None, None, None
    try:
        transitions_df = pd.read_csv(transitions_csv_path)
        transitions_df['date'] = pd.to_datetime(transitions_df['date'])
    except Exception as e:
        print(f"  [ERROR] Cannot read transitions CSV: {e}")
        return plt.figure(figsize=(15, 8)), None, None, None, None, None, None, None

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
                                               'lick_count': 0, 'dist_sum': 0.0}
                animal_level_accum[key]['rewards']      += count
                animal_level_accum[key]['duration_min'] += duration_min
                if treadmill_df is not None and len(treadmill_df) > 0:
                    mask = (treadmill_df['global_time'] >= start_t) & (treadmill_df['global_time'] < end_t)
                    lvl_speeds = treadmill_df.loc[mask, 'speed'].values / 10.0
                    animal_level_accum[key]['speed_sum']   += float(np.sum(lvl_speeds))
                    animal_level_accum[key]['speed_count'] += len(lvl_speeds)
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
    for (animal_id, level), accum in animal_level_accum.items():
        condition = accum['condition']
        if accum['duration_min'] > 0:
            rpm = accum['rewards'] / accum['duration_min']
            condition_level_data.setdefault(condition, {}).setdefault(level, []).append(rpm)
        if accum['speed_count'] > 0:
            mean_spd = accum['speed_sum'] / accum['speed_count']
            condition_level_speed.setdefault(condition, {}).setdefault(level, []).append(mean_spd)
            collapsed_level_speed.setdefault(level, []).append(mean_spd)
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

    return level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig

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

        # Process each date's data
        for timestamp, row in df.iterrows():
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
                    avg_speed = treadmill_data['speed'].mean() / 10.0
                    _, total_distance = compute_session_distance(treadmill_data)
                else:
                    avg_speed = float('nan')
                    total_distance = float('nan')

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
                hits.append(reward_count)
                misses_list.append(misses)
                sensitivities.append(sensitivity)
                lick_counts.append(lick_count)
                session_lengths.append(session_length_minutes)
                false_alarms_list.append(false_alarm_count)
                correct_rejections_list.append(correct_rejection_count)
                specificities_list.append(specificity)
                dprimes_list.append(dprime)

            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'average_speed': speeds,
            'total_distance': total_distances,
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
        
        # Store results for this mouse
        all_results.append({
            'mouse': mouse_name,
            'dates': dates,
            'speeds': speeds,
            'total_distances': total_distances,
            'hits': hits,
            'false_alarms': false_alarms_list,
            'correct_rejections': correct_rejections_list,
            'dprimes': dprimes_list,
            'session_lengths': session_lengths,
            'starting_condition': conditions[mouse_name],
            'df': results_df,
            'session_file_errors': session_file_errors,
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

    # Create the level-based analysis plots
    level_reward_fig = level_speed_collapsed_fig = level_speed_condition_fig = None
    level_lick_collapsed_fig = level_lick_condition_fig = None
    level_dist_collapsed_fig = level_dist_condition_fig = level_dist_condition_excl_last_fig = None
    if any(k in selected_plots for k in ('levels', 'level_speed', 'level_speed_condition',
                                          'level_lick', 'level_lick_condition',
                                          'level_dist', 'level_dist_condition',
                                          'level_dist_condition_excl_last')):
        level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, \
        level_lick_collapsed_fig, level_lick_condition_fig, \
        level_dist_collapsed_fig, level_dist_condition_fig, \
        level_dist_condition_excl_last_fig = analyze_levels(
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

    return speed_fig, sensitivity_fig, lick_fig, reward_fig, false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig, avg_reward_fig, sex_reward_fig, distance_fig, sex_distance_fig, condition_distance_fig, condition_distance_bar_fig, total_distance_bar_fig, avg_lick_rate_fig, sex_lick_rate_fig, condition_reward_fig, condition_speed_fig, condition_lick_fig, condition_lick_rate_fig, condition_bar_fig, condition_speed_bar_fig, condition_lick_bar_fig, level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig, all_results

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
        # Ask user which plots to generate
        selected_plots = _ask_plot_selection(root)
        if not selected_plots:
            print("No plots selected. Exiting...")
            return

        # Select transitions CSV only if a level plot was requested
        transitions_csv_path = None
        if any(k in selected_plots for k in ('levels', 'level_speed', 'level_speed_condition',
                                              'level_lick', 'level_lick_condition',
                                              'level_dist', 'level_dist_condition',
                                              'level_dist_condition_excl_last')):
            transitions_csv_path = filedialog.askopenfilename(
                title='Select transitions CSV (from level_sorter.py) — cancel to skip level plot',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
                initialdir=os.path.dirname(file_paths[0]),
            ) or None
            if transitions_csv_path:
                print(f"Transitions CSV: {os.path.basename(transitions_csv_path)}")
            else:
                print("No transitions CSV selected — level plot will be empty.")

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
        
        # Analyze data and plot results
        speed_fig, sensitivity_fig, lick_fig, reward_fig, false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig, avg_reward_fig, sex_reward_fig, distance_fig, sex_distance_fig, condition_distance_fig, condition_distance_bar_fig, total_distance_bar_fig, avg_lick_rate_fig, sex_lick_rate_fig, condition_reward_fig, condition_speed_fig, condition_lick_fig, condition_lick_rate_fig, condition_bar_fig, condition_speed_bar_fig, condition_lick_bar_fig, level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig, level_lick_collapsed_fig, level_lick_condition_fig, level_dist_collapsed_fig, level_dist_condition_fig, level_dist_condition_excl_last_fig, all_results = analyze_mouse_data(
            file_paths, markers, starting_conditions,
            transitions_csv_path=transitions_csv_path,
            selected_plots=selected_plots,
        )

        # All generated figures (None entries are skipped)
        all_figs = [f for f in [
            speed_fig, sensitivity_fig, lick_fig, reward_fig,
            false_alarm_fig, correct_rejection_fig, specificity_fig, dprime_fig,
            avg_reward_fig, sex_reward_fig,
            distance_fig, sex_distance_fig, condition_distance_fig,
            condition_distance_bar_fig, total_distance_bar_fig,
            avg_lick_rate_fig, sex_lick_rate_fig,
            condition_reward_fig, condition_speed_fig, condition_lick_fig, condition_lick_rate_fig,
            condition_bar_fig, condition_speed_bar_fig, condition_lick_bar_fig,
            level_reward_fig, level_speed_collapsed_fig, level_speed_condition_fig,
            level_lick_collapsed_fig, level_lick_condition_fig,
            level_dist_collapsed_fig, level_dist_condition_fig,
            level_dist_condition_excl_last_fig,
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
            (sex_distance_fig,      'sex_distance',        'Sex-specific distance per session plot'),
            (condition_distance_fig,'condition_distance',  'Condition-based distance per session plot'),
            (condition_distance_bar_fig, 'condition_distance_bar', 'Condition distance collapsed bar chart'),
            (total_distance_bar_fig,     'total_distance_bar',     'Total distance per mouse collapsed bar chart'),
            (avg_lick_rate_fig,     'avg_lick_rate',       'Average lick rate plot'),
            (sex_lick_rate_fig,     'sex_lick_rate',       'Sex-specific lick rate plot'),
            (condition_reward_fig,  'condition_reward',    'Condition-based average rewards plot'),
            (condition_speed_fig,   'condition_speed',     'Condition-based average speed plot'),
            (condition_lick_fig,    'condition_lick',      'Condition-based average lick count plot'),
            (condition_lick_rate_fig, 'condition_lick_rate', 'Condition-based lick rate plot'),
            (condition_bar_fig,      'condition_bar',       'Condition collapsed bar chart'),
            (condition_speed_bar_fig,'condition_speed_bar', 'Condition speed collapsed bar chart'),
            (condition_lick_bar_fig, 'condition_lick_bar',  'Condition lick rate collapsed bar chart'),
            (level_reward_fig,              'level_reward',          'Level-based average rewards plot'),
            (level_speed_collapsed_fig,     'level_speed',           'Level-based average speed — collapsed'),
            (level_speed_condition_fig,     'level_speed_condition', 'Level-based average speed — by condition'),
            (level_lick_collapsed_fig,      'level_lick',            'Level-based average lick rate — collapsed'),
            (level_lick_condition_fig,      'level_lick_condition',  'Level-based average lick rate — by condition'),
            (level_dist_collapsed_fig,      'level_dist',            'Level-based distance — collapsed'),
            (level_dist_condition_fig,      'level_dist_condition',  'Level-based distance — by condition'),
            (level_dist_condition_excl_last_fig, 'level_dist_condition_excl_last', 'Level-based distance — by condition, last level excluded'),
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
    else:
        print("No file selected. Exiting...")
if __name__ == "__main__":
    main()