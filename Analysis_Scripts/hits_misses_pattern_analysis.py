"""
Pattern Analysis for Reward Hits and Misses

Analyzes temporal patterns in reward zone success/failure:
- Clustering & Randomness (Runs test)
- Conditional Probabilities
- Autocorrelation
- Time-in-Session Effects

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
from scipy import stats

# Configure matplotlib
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
    capacitive_files = [f for f in os.listdir(folder_path) if 'capacitive.csv' in f]
    treadmill_files = [f for f in os.listdir(folder_path) if 'treadmill.csv' in f]
    
    if not trial_log_files or not capacitive_files:
        print("Error: Missing required files (trial_log.csv or capacitive.csv)")
        return None
    
    print(f"\nLoading files:")
    print(f"  - {trial_log_files[0]}")
    print(f"  - {capacitive_files[0]}")
    if treadmill_files:
        print(f"  - {treadmill_files[0]}")
    
    data = {
        'trial_log': pd.read_csv(os.path.join(folder_path, trial_log_files[0]), engine='python'),
        'capacitive': pd.read_csv(os.path.join(folder_path, capacitive_files[0]), comment='/', engine='python')
    }
    
    if treadmill_files:
        data['treadmill'] = pd.read_csv(os.path.join(folder_path, treadmill_files[0]), comment='/', engine='python')
    
    print("Files loaded successfully.\n")
    return data


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
    
    Returns:
        Tuple of (hits_list, misses_list, sequence, zone_times, is_hit_sequence)
    """
    hits = []
    misses = []
    sequence = []  # List of (zone_entry_time, is_hit)
    
    reward_event_times = [r_time for _, r_time in reward_events]
    
    for trial_idx, zone_entry_time in all_reward_zones:
        matching_rewards = [
            r_time for r_time in reward_event_times
            if zone_entry_time <= r_time <= (zone_entry_time + match_window)
        ]
        
        is_hit = len(matching_rewards) > 0
        
        if is_hit:
            hits.append(zone_entry_time)
        else:
            misses.append(zone_entry_time)
        
        sequence.append((zone_entry_time, is_hit))
    
    # Create arrays for analysis
    zone_times = np.array([s[0] for s in sequence])
    is_hit_sequence = np.array([s[1] for s in sequence], dtype=int)
    
    return hits, misses, sequence, zone_times, is_hit_sequence


# ============================================================================
# ANALYSIS 1: RUNS TEST (Clustering & Randomness)
# ============================================================================

def runs_test(sequence):
    """
    Perform runs test to detect non-randomness in binary sequence
    
    A 'run' is a sequence of consecutive identical outcomes
    Example: [1,1,0,0,0,1] has 3 runs: [1,1], [0,0,0], [1]
    
    Returns:
        dict with runs count, expected runs, z-score, p-value
    """
    n = len(sequence)
    n_hits = np.sum(sequence)
    n_misses = n - n_hits
    
    if n_hits == 0 or n_misses == 0:
        return {
            'n_runs': np.nan,
            'expected_runs': np.nan,
            'z_score': np.nan,
            'p_value': np.nan,
            'interpretation': 'All outcomes identical - no variation'
        }
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i-1]:
            runs += 1
    
    # Expected number of runs under null hypothesis (random)
    expected_runs = (2 * n_hits * n_misses) / n + 1
    
    # Variance of runs
    var_runs = (2 * n_hits * n_misses * (2 * n_hits * n_misses - n)) / (n**2 * (n - 1))
    
    # Z-score (for large samples, runs follows normal distribution)
    if var_runs > 0:
        z_score = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    else:
        z_score = np.nan
        p_value = np.nan
    
    # Interpretation
    if p_value < 0.05:
        if runs < expected_runs:
            interpretation = 'CLUSTERED (fewer runs than expected, p<0.05)'
        else:
            interpretation = 'TOO ALTERNATING (more runs than expected, p<0.05)'
    else:
        interpretation = 'RANDOM (consistent with random pattern)'
    
    return {
        'n_runs': runs,
        'expected_runs': expected_runs,
        'z_score': z_score,
        'p_value': p_value,
        'interpretation': interpretation
    }


# ============================================================================
# ANALYSIS 2: CONDITIONAL PROBABILITIES
# ============================================================================

def calculate_conditional_probabilities(sequence):
    """
    Calculate conditional probabilities: P(Hit | previous outcome)
    
    Returns:
        dict with probabilities and counts
    """
    n = len(sequence)
    
    if n < 2:
        return None
    
    # Count transitions
    hit_after_hit = 0
    hit_after_miss = 0
    total_after_hit = 0
    total_after_miss = 0
    
    for i in range(1, n):
        prev = sequence[i-1]
        curr = sequence[i]
        
        if prev == 1:  # Previous was hit
            total_after_hit += 1
            if curr == 1:
                hit_after_hit += 1
        else:  # Previous was miss
            total_after_miss += 1
            if curr == 1:
                hit_after_miss += 1
    
    # Calculate probabilities
    p_hit_after_hit = hit_after_hit / total_after_hit if total_after_hit > 0 else np.nan
    p_hit_after_miss = hit_after_miss / total_after_miss if total_after_miss > 0 else np.nan
    
    # Overall hit rate (baseline)
    overall_hit_rate = np.mean(sequence)
    
    return {
        'p_hit_after_hit': p_hit_after_hit,
        'p_hit_after_miss': p_hit_after_miss,
        'overall_hit_rate': overall_hit_rate,
        'n_after_hit': total_after_hit,
        'n_after_miss': total_after_miss,
        'hit_after_hit_count': hit_after_hit,
        'hit_after_miss_count': hit_after_miss
    }


# ============================================================================
# ANALYSIS 3: AUTOCORRELATION
# ============================================================================

def calculate_autocorrelation(sequence, max_lag=10):
    """
    Calculate autocorrelation at different lags
    
    Autocorrelation measures correlation between sequence and lagged version
    Lag 1: correlation between outcome[i] and outcome[i-1]
    Lag 2: correlation between outcome[i] and outcome[i-2], etc.
    
    Returns:
        dict with lags and autocorrelation values
    """
    n = len(sequence)
    mean_seq = np.mean(sequence)
    var_seq = np.var(sequence)
    
    if var_seq == 0:
        return None
    
    lags = range(1, min(max_lag + 1, n // 2))
    autocorr_values = []
    
    for lag in lags:
        # Calculate correlation between sequence and lagged sequence
        seq1 = sequence[lag:]
        seq2 = sequence[:-lag]
        
        autocorr = np.corrcoef(seq1, seq2)[0, 1]
        autocorr_values.append(autocorr)
    
    return {
        'lags': list(lags),
        'autocorr': autocorr_values
    }


# ============================================================================
# ANALYSIS 4: TIME-IN-SESSION EFFECTS
# ============================================================================

def calculate_session_progression(sequence, zone_times, n_bins=10):
    """
    Calculate hit rate as function of time in session
    
    Divides session into bins and calculates hit rate for each
    
    Returns:
        dict with bin centers and hit rates
    """
    n = len(sequence)
    
    if n == 0:
        return None
    
    # Trial number analysis (divide into equal-sized bins)
    bin_size = n // n_bins
    trial_bins = []
    trial_hit_rates = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
        
        if end_idx > start_idx:
            bin_sequence = sequence[start_idx:end_idx]
            hit_rate = np.mean(bin_sequence)
            trial_bins.append((start_idx + end_idx) / 2)  # Bin center
            trial_hit_rates.append(hit_rate)
    
    # Time-based analysis (divide session time into bins)
    time_min = zone_times.min()
    time_max = zone_times.max()
    time_bins_edges = np.linspace(time_min, time_max, n_bins + 1)
    time_bin_centers = []
    time_hit_rates = []
    
    for i in range(n_bins):
        t_start = time_bins_edges[i]
        t_end = time_bins_edges[i + 1]
        
        in_bin = (zone_times >= t_start) & (zone_times <= t_end)
        if np.sum(in_bin) > 0:
            bin_sequence = sequence[in_bin]
            hit_rate = np.mean(bin_sequence)
            time_bin_centers.append((t_start + t_end) / 2)
            time_hit_rates.append(hit_rate)
    
    # Calculate trend (linear regression)
    trial_slope, trial_intercept, trial_r, trial_p, _ = stats.linregress(
        trial_bins, trial_hit_rates
    )
    
    time_slope, time_intercept, time_r, time_p, _ = stats.linregress(
        time_bin_centers, time_hit_rates
    ) if len(time_bin_centers) > 1 else (np.nan, np.nan, np.nan, np.nan, np.nan)
    
    return {
        'trial_bins': trial_bins,
        'trial_hit_rates': trial_hit_rates,
        'trial_slope': trial_slope,
        'trial_p_value': trial_p,
        'time_bins': time_bin_centers,
        'time_hit_rates': time_hit_rates,
        'time_slope': time_slope,
        'time_p_value': time_p
    }


# ============================================================================
# ANALYSIS 5: FIRST MISS AFTER HIT - SPEED ANALYSIS
# ============================================================================

def identify_first_miss_after_hit(sequence):
    """
    Identify all instances where a miss occurs immediately after a hit
    
    Args:
        sequence: List of (zone_entry_time, is_hit) tuples
    
    Returns:
        List of tuples (miss_zone_time, miss_index) for first misses after hits
    """
    first_misses_after_hit = []
    
    for i in range(1, len(sequence)):
        prev_time, prev_is_hit = sequence[i-1]
        curr_time, curr_is_hit = sequence[i]
        
        # Check if current is a miss and previous was a hit
        if prev_is_hit and not curr_is_hit:
            first_misses_after_hit.append((curr_time, i))
    
    return first_misses_after_hit


def identify_last_hit_before_miss(sequence):
    """
    Identify all instances where a hit occurs immediately before a miss
    
    Args:
        sequence: List of (zone_entry_time, is_hit) tuples
    
    Returns:
        List of tuples (hit_zone_time, hit_index) for last hits before misses
    """
    last_hits_before_miss = []
    
    for i in range(len(sequence) - 1):
        curr_time, curr_is_hit = sequence[i]
        next_time, next_is_hit = sequence[i+1]
        
        # Check if current is a hit and next is a miss
        if curr_is_hit and not next_is_hit:
            last_hits_before_miss.append((curr_time, i))
    
    return last_hits_before_miss


def identify_last_miss_before_hit(sequence):
    """
    Identify all instances where a miss occurs immediately before a hit
    
    Args:
        sequence: List of (zone_entry_time, is_hit) tuples
    
    Returns:
        List of tuples (miss_zone_time, miss_index) for last misses before hits
    """
    last_misses_before_hit = []
    
    for i in range(len(sequence) - 1):
        curr_time, curr_is_hit = sequence[i]
        next_time, next_is_hit = sequence[i+1]
        
        # Check if current is a miss and next is a hit
        if not curr_is_hit and next_is_hit:
            last_misses_before_hit.append((curr_time, i))
    
    return last_misses_before_hit


def prepare_treadmill_data(treadmill_df):
    """
    Prepare treadmill data by interpolating to regular time intervals
    (Similar to daily_analysis.py SpeedAnalysis class)
    
    Returns:
        pandas Series with interpolated speed values indexed by time
    """
    # Remove rows with NaN or invalid values
    # Note: treadmill uses 'global_time' not 'elapsed_time'
    treadmill_clean = treadmill_df.dropna(subset=['global_time', 'speed'])
    treadmill_clean = treadmill_clean[treadmill_clean['global_time'] >= 0]
    
    if len(treadmill_clean) < 2:
        return None
    
    # Convert speed from mm/s to cm/s (same as daily_analysis.py)
    treadmill_clean = treadmill_clean.copy()
    treadmill_clean['speed'] = treadmill_clean['speed'] / 10
    
    # Sort by time
    treadmill_clean = treadmill_clean.sort_values('global_time')
    
    # Create interpolated series (interpolate to 10Hz, same as daily_analysis)
    time_min = treadmill_clean['global_time'].min()
    time_max = treadmill_clean['global_time'].max()
    time_interp = np.arange(time_min, time_max, 0.1)  # 10 Hz
    
    speed_interp = np.interp(time_interp, 
                            treadmill_clean['global_time'].values,
                            treadmill_clean['speed'].values)
    
    return pd.Series(speed_interp, index=time_interp)


def analyze_first_miss_after_hit_speeds(first_misses_after_hit, treadmill_interp, window=2.0):
    """
    Analyze speed before and after entering reward zone for first misses after hits
    Uses the same 2-second window as daily_analysis.py for no-reward zones
    
    Args:
        first_misses_after_hit: List of (miss_zone_time, miss_index) tuples
        treadmill_interp: Interpolated treadmill data (pandas Series)
        window: Time window in seconds (default 2.0)
    
    Returns:
        dict with speed metrics and raw data for visualization
    """
    if treadmill_interp is None or len(first_misses_after_hit) == 0:
        return None
    
    speed_times = treadmill_interp.index
    
    speeds_before = []
    speeds_after = []
    miss_times = []
    
    for miss_time, miss_idx in first_misses_after_hit:
        # Speed in 2-second window BEFORE zone entry: [miss_time - 2, miss_time)
        before_mask = (speed_times >= (miss_time - window)) & (speed_times < miss_time)
        if np.any(before_mask):
            speed_before = treadmill_interp[before_mask].mean()
            speeds_before.append(speed_before)
        else:
            speeds_before.append(np.nan)
        
        # Speed in 2-second window AFTER zone entry: [miss_time, miss_time + 2)
        after_mask = (speed_times >= miss_time) & (speed_times < (miss_time + window))
        if np.any(after_mask):
            speed_after = treadmill_interp[after_mask].mean()
            speeds_after.append(speed_after)
        else:
            speeds_after.append(np.nan)
        
        miss_times.append(miss_time)
    
    # Remove NaN pairs
    valid_indices = [i for i in range(len(speeds_before)) 
                    if not np.isnan(speeds_before[i]) and not np.isnan(speeds_after[i])]
    
    speeds_before_valid = [speeds_before[i] for i in valid_indices]
    speeds_after_valid = [speeds_after[i] for i in valid_indices]
    miss_times_valid = [miss_times[i] for i in valid_indices]
    
    # Calculate statistics
    avg_speed_before = np.mean(speeds_before_valid) if speeds_before_valid else np.nan
    avg_speed_after = np.mean(speeds_after_valid) if speeds_after_valid else np.nan
    
    # Perform paired t-test
    t_stat = np.nan
    p_value = np.nan
    if len(speeds_before_valid) >= 2:
        t_stat, p_value = stats.ttest_rel(speeds_after_valid, speeds_before_valid)
    
    return {
        'n_first_misses': len(first_misses_after_hit),
        'n_valid_speed_pairs': len(speeds_before_valid),
        'avg_speed_before': avg_speed_before,
        'avg_speed_after': avg_speed_after,
        'speeds_before_list': speeds_before_valid,
        'speeds_after_list': speeds_after_valid,
        'miss_times': miss_times_valid,
        't_statistic': t_stat,
        'p_value': p_value,
        'window_seconds': window
    }


def analyze_last_hit_before_miss_speeds(last_hits_before_miss, treadmill_interp, window=2.0):
    """
    Analyze speed before and after entering reward zone for last hits before misses
    Uses the same 2-second window as daily_analysis.py
    
    Args:
        last_hits_before_miss: List of (hit_zone_time, hit_index) tuples
        treadmill_interp: Interpolated treadmill data (pandas Series)
        window: Time window in seconds (default 2.0)
    
    Returns:
        dict with speed metrics and raw data for visualization
    """
    if treadmill_interp is None or len(last_hits_before_miss) == 0:
        return None
    
    speed_times = treadmill_interp.index
    
    speeds_before = []
    speeds_after = []
    hit_times = []
    
    for hit_time, hit_idx in last_hits_before_miss:
        # Speed in 2-second window BEFORE zone entry: [hit_time - 2, hit_time)
        before_mask = (speed_times >= (hit_time - window)) & (speed_times < hit_time)
        if np.any(before_mask):
            speed_before = treadmill_interp[before_mask].mean()
            speeds_before.append(speed_before)
        else:
            speeds_before.append(np.nan)
        
        # Speed in 2-second window AFTER zone entry: [hit_time, hit_time + 2)
        after_mask = (speed_times >= hit_time) & (speed_times < (hit_time + window))
        if np.any(after_mask):
            speed_after = treadmill_interp[after_mask].mean()
            speeds_after.append(speed_after)
        else:
            speeds_after.append(np.nan)
        
        hit_times.append(hit_time)
    
    # Remove NaN pairs
    valid_indices = [i for i in range(len(speeds_before)) 
                    if not np.isnan(speeds_before[i]) and not np.isnan(speeds_after[i])]
    
    speeds_before_valid = [speeds_before[i] for i in valid_indices]
    speeds_after_valid = [speeds_after[i] for i in valid_indices]
    hit_times_valid = [hit_times[i] for i in valid_indices]
    
    # Calculate statistics
    avg_speed_before = np.mean(speeds_before_valid) if speeds_before_valid else np.nan
    avg_speed_after = np.mean(speeds_after_valid) if speeds_after_valid else np.nan
    
    # Perform paired t-test
    t_stat = np.nan
    p_value = np.nan
    if len(speeds_before_valid) >= 2:
        t_stat, p_value = stats.ttest_rel(speeds_after_valid, speeds_before_valid)
    
    return {
        'n_last_hits': len(last_hits_before_miss),
        'n_valid_speed_pairs': len(speeds_before_valid),
        'avg_speed_before': avg_speed_before,
        'avg_speed_after': avg_speed_after,
        'speeds_before_list': speeds_before_valid,
        'speeds_after_list': speeds_after_valid,
        'hit_times': hit_times_valid,
        't_statistic': t_stat,
        'p_value': p_value,
        'window_seconds': window
    }


def analyze_last_miss_before_hit_speeds(last_misses_before_hit, treadmill_interp, window=2.0):
    """
    Analyze speed before and after entering reward zone for last misses before hits
    Uses the same 2-second window as daily_analysis.py
    
    Args:
        last_misses_before_hit: List of (miss_zone_time, miss_index) tuples
        treadmill_interp: Interpolated treadmill data (pandas Series)
        window: Time window in seconds (default 2.0)
    
    Returns:
        dict with speed metrics and raw data for visualization
    """
    if treadmill_interp is None or len(last_misses_before_hit) == 0:
        return None
    
    speed_times = treadmill_interp.index
    
    speeds_before = []
    speeds_after = []
    miss_times = []
    
    for miss_time, miss_idx in last_misses_before_hit:
        # Speed in 2-second window BEFORE zone entry: [miss_time - 2, miss_time)
        before_mask = (speed_times >= (miss_time - window)) & (speed_times < miss_time)
        if np.any(before_mask):
            speed_before = treadmill_interp[before_mask].mean()
            speeds_before.append(speed_before)
        else:
            speeds_before.append(np.nan)
        
        # Speed in 2-second window AFTER zone entry: [miss_time, miss_time + 2)
        after_mask = (speed_times >= miss_time) & (speed_times < (miss_time + window))
        if np.any(after_mask):
            speed_after = treadmill_interp[after_mask].mean()
            speeds_after.append(speed_after)
        else:
            speeds_after.append(np.nan)
        
        miss_times.append(miss_time)
    
    # Remove NaN pairs
    valid_indices = [i for i in range(len(speeds_before)) 
                    if not np.isnan(speeds_before[i]) and not np.isnan(speeds_after[i])]
    
    speeds_before_valid = [speeds_before[i] for i in valid_indices]
    speeds_after_valid = [speeds_after[i] for i in valid_indices]
    miss_times_valid = [miss_times[i] for i in valid_indices]
    
    # Calculate statistics
    avg_speed_before = np.mean(speeds_before_valid) if speeds_before_valid else np.nan
    avg_speed_after = np.mean(speeds_after_valid) if speeds_after_valid else np.nan
    
    # Perform paired t-test
    t_stat = np.nan
    p_value = np.nan
    if len(speeds_before_valid) >= 2:
        t_stat, p_value = stats.ttest_rel(speeds_after_valid, speeds_before_valid)
    
    return {
        'n_last_misses': len(last_misses_before_hit),
        'n_valid_speed_pairs': len(speeds_before_valid),
        'avg_speed_before': avg_speed_before,
        'avg_speed_after': avg_speed_after,
        'speeds_before_list': speeds_before_valid,
        'speeds_after_list': speeds_after_valid,
        'miss_times': miss_times_valid,
        't_statistic': t_stat,
        'p_value': p_value,
        'window_seconds': window
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_all_analyses(runs_result, cond_prob_result, autocorr_result, 
                      session_prog_result, is_hit_sequence, output_folder):
    """Create comprehensive figure with all analysis results"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Plot 1: Runs Test Results (Top Left)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    if runs_result and not np.isnan(runs_result['n_runs']):
        n_runs = runs_result['n_runs']
        expected = runs_result['expected_runs']
        
        ax1.bar(['Observed\nRuns', 'Expected\nRuns (random)'], 
                [n_runs, expected], 
                color=['#2ca02c' if n_runs >= expected else '#d62728', '#808080'],
                alpha=0.7, edgecolor='black')
        
        ax1.set_ylabel('Number of Runs', fontsize=11)
        ax1.set_title('Runs Test for Clustering/Randomness', fontsize=12, fontweight='bold')
        
        # Add text annotation
        interp_text = runs_result['interpretation']
        p_val_text = f"p = {runs_result['p_value']:.4f}" if not np.isnan(runs_result['p_value']) else "p = N/A"
        ax1.text(0.5, 0.95, f"{interp_text}\n{p_val_text}", 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for runs test', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Runs Test for Clustering/Randomness', fontsize=12, fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 2: Conditional Probabilities (Top Right)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if cond_prob_result:
        p_after_hit = cond_prob_result['p_hit_after_hit']
        p_after_miss = cond_prob_result['p_hit_after_miss']
        overall = cond_prob_result['overall_hit_rate']
        
        x_pos = [0, 1, 2]
        probs = [p_after_hit, p_after_miss, overall]
        colors = ['#2ca02c', '#9467bd', '#808080']
        labels = ['P(Hit | prev Hit)', 'P(Hit | prev Miss)', 'Overall Hit Rate']
        
        bars = ax2.bar(x_pos, probs, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel('Probability', fontsize=11)
        ax2.set_title('Conditional Probabilities', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Conditional Probabilities', fontsize=12, fontweight='bold')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 3: Autocorrelation (Middle Left)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    if autocorr_result and len(autocorr_result['autocorr']) > 0:
        lags = autocorr_result['lags']
        autocorr = autocorr_result['autocorr']
        
        ax3.bar(lags, autocorr, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add confidence interval lines (approximate 95% CI for white noise)
        n = len(is_hit_sequence)
        ci = 1.96 / np.sqrt(n)
        ax3.axhline(y=ci, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% CI')
        ax3.axhline(y=-ci, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax3.set_xlabel('Lag', fontsize=11)
        ax3.set_ylabel('Autocorrelation', fontsize=11)
        ax3.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
        ax3.set_ylim(-1, 1)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 4: Hit Rate vs Trial Number (Middle Right)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if session_prog_result:
        trial_bins = session_prog_result['trial_bins']
        trial_rates = session_prog_result['trial_hit_rates']
        slope = session_prog_result['trial_slope']
        p_val = session_prog_result['trial_p_value']
        
        ax4.plot(trial_bins, trial_rates, 'o-', color='#ff7f0e', 
                linewidth=2, markersize=8, alpha=0.7, label='Hit Rate')
        
        # Add trend line
        x_fit = np.array([min(trial_bins), max(trial_bins)])
        y_fit = session_prog_result['trial_slope'] * x_fit + session_prog_result['trial_p_value']
        
        ax4.set_xlabel('Trial Number', fontsize=11)
        ax4.set_ylabel('Hit Rate', fontsize=11)
        ax4.set_title('Hit Rate Across Session (Trial Number)', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1)
        
        # Add trend annotation
        trend_text = f"Slope = {slope:.4f}, p = {p_val:.4f}"
        if p_val < 0.05:
            trend_text += "\n(Significant trend)"
        ax4.text(0.05, 0.95, trend_text, transform=ax4.transAxes, 
                va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Hit Rate Across Session (Trial Number)', fontsize=12, fontweight='bold')
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 5: Hit Rate vs Session Time (Bottom Left)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    if session_prog_result:
        time_bins = session_prog_result['time_bins']
        time_rates = session_prog_result['time_hit_rates']
        
        ax5.plot(time_bins, time_rates, 'o-', color='#2ca02c', 
                linewidth=2, markersize=8, alpha=0.7, label='Hit Rate')
        
        ax5.set_xlabel('Session Time (s)', fontsize=11)
        ax5.set_ylabel('Hit Rate', fontsize=11)
        ax5.set_title('Hit Rate Across Session (Time)', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 1)
        
        # Add trend annotation
        slope = session_prog_result['time_slope']
        p_val = session_prog_result['time_p_value']
        if not np.isnan(slope):
            trend_text = f"Slope = {slope:.6f}, p = {p_val:.4f}"
            if p_val < 0.05:
                trend_text += "\n(Significant trend)"
            ax5.text(0.05, 0.95, trend_text, transform=ax5.transAxes, 
                    va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Hit Rate Across Session (Time)', fontsize=12, fontweight='bold')
    
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 6: Summary Statistics (Bottom Right)
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Compile summary text
    summary_lines = ["SUMMARY STATISTICS\n" + "="*40]
    
    n_total = len(is_hit_sequence)
    n_hits = np.sum(is_hit_sequence)
    n_misses = n_total - n_hits
    
    summary_lines.append(f"Total reward zones: {n_total}")
    summary_lines.append(f"Hits: {n_hits} ({n_hits/n_total*100:.1f}%)")
    summary_lines.append(f"Misses: {n_misses} ({n_misses/n_total*100:.1f}%)")
    summary_lines.append("")
    
    if runs_result:
        summary_lines.append("CLUSTERING TEST:")
        summary_lines.append(f"  {runs_result['interpretation']}")
        summary_lines.append("")
    
    if cond_prob_result:
        summary_lines.append("CONDITIONAL PROBABILITIES:")
        summary_lines.append(f"  P(Hit | prev Hit) = {cond_prob_result['p_hit_after_hit']:.3f}")
        summary_lines.append(f"  P(Hit | prev Miss) = {cond_prob_result['p_hit_after_miss']:.3f}")
        diff = cond_prob_result['p_hit_after_hit'] - cond_prob_result['p_hit_after_miss']
        summary_lines.append(f"  Difference = {diff:.3f}")
        summary_lines.append("")
    
    if autocorr_result and len(autocorr_result['autocorr']) > 0:
        lag1_autocorr = autocorr_result['autocorr'][0]
        summary_lines.append("AUTOCORRELATION:")
        summary_lines.append(f"  Lag-1 autocorr = {lag1_autocorr:.3f}")
        summary_lines.append("")
    
    if session_prog_result:
        summary_lines.append("SESSION TRENDS:")
        p_trial = session_prog_result['trial_p_value']
        summary_lines.append(f"  Trial trend p-value = {p_trial:.4f}")
        if p_trial < 0.05:
            slope = session_prog_result['trial_slope']
            direction = "IMPROVING" if slope > 0 else "DECLINING"
            summary_lines.append(f"  → {direction} over session")
    
    summary_text = '\n'.join(summary_lines)
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, 
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, "hits_misses_pattern_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved comprehensive analysis plot: {output_path}")
    
    plt.show()


def plot_last_hit_before_miss_speeds(speed_result, output_folder):
    """
    Create visualization for last-hit-before-miss speed analysis
    
    Args:
        speed_result: Dictionary from analyze_last_hit_before_miss_speeds()
        output_folder: Path to save the figure
    """
    if speed_result is None or speed_result['n_valid_speed_pairs'] == 0:
        print("No valid speed data for last-hit-before-miss analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================================================
    # Plot 1: Bar Chart Comparison (Before vs After)
    # ========================================================================
    speeds_before = speed_result['speeds_before_list']
    speeds_after = speed_result['speeds_after_list']
    avg_before = speed_result['avg_speed_before']
    avg_after = speed_result['avg_speed_after']
    p_value = speed_result['p_value']
    window = speed_result['window_seconds']
    
    x_pos = [0, 1]
    means = [avg_before, avg_after]
    labels = [f'Before Zone Entry\n({window}s window)', f'After Zone Entry\n({window}s window)']
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Average Speed (cm/s)', fontsize=11)
    ax1.set_title('Speed Before/After Entering Reward Zone\n(Last Hit Before Miss)', 
                  fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add significance annotation
    if not np.isnan(p_value):
        # Determine significance stars
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        # Draw line and significance marker
        max_y = max(means) * 1.15
        ax1.plot([0, 1], [max_y, max_y], 'k-', linewidth=1.5)
        ax1.text(0.5, max_y + 0.5, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add p-value text box
        p_text = f'p = {p_value:.4f}' if p_value >= 0.0001 else f'p < 0.0001'
        ax1.text(0.98, 0.98, p_text, transform=ax1.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 2: Paired Data Points (Individual Hits)
    # ========================================================================
    n_pairs = len(speeds_before)
    
    for i in range(n_pairs):
        ax2.plot([0, 1], [speeds_before[i], speeds_after[i]], 
                'o-', color='gray', alpha=0.3, linewidth=1, markersize=4)
    
    # Overlay mean values
    ax2.plot([0, 1], [avg_before, avg_after], 
            'o-', color='red', linewidth=3, markersize=12, 
            label=f'Mean (n={n_pairs})', zorder=10)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'Before\n({window}s)', f'After\n({window}s)'], fontsize=10)
    ax2.set_ylabel('Speed (cm/s)', fontsize=11)
    ax2.set_title('Individual Last-Hit-Before-Miss Events', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add summary statistics
    change = avg_after - avg_before
    pct_change = (change / avg_before * 100) if avg_before != 0 else 0
    
    stats_text = f"N hits: {speed_result['n_last_hits']}\n"
    stats_text += f"Valid pairs: {n_pairs}\n"
    stats_text += f"Mean change: {change:+.1f} cm/s ({pct_change:+.1f}%)\n"
    stats_text += f"t-statistic: {speed_result['t_statistic']:.3f}"
    
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, ha='left', va='bottom',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, "last_hit_before_miss_speed_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved last-hit-before-miss speed analysis plot: {output_path}")
    
    plt.show()


def plot_last_miss_before_hit_speeds(speed_result, output_folder):
    """
    Create visualization for last-miss-before-hit speed analysis
    
    Args:
        speed_result: Dictionary from analyze_last_miss_before_hit_speeds()
        output_folder: Path to save the figure
    """
    if speed_result is None or speed_result['n_valid_speed_pairs'] == 0:
        print("No valid speed data for last-miss-before-hit analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================================================
    # Plot 1: Bar Chart Comparison (Before vs After)
    # ========================================================================
    speeds_before = speed_result['speeds_before_list']
    speeds_after = speed_result['speeds_after_list']
    avg_before = speed_result['avg_speed_before']
    avg_after = speed_result['avg_speed_after']
    p_value = speed_result['p_value']
    window = speed_result['window_seconds']
    
    x_pos = [0, 1]
    means = [avg_before, avg_after]
    labels = [f'Before Zone Entry\n({window}s window)', f'After Zone Entry\n({window}s window)']
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Average Speed (cm/s)', fontsize=11)
    ax1.set_title('Speed Before/After Entering Reward Zone\n(Last Miss Before Hit)', 
                  fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add significance annotation
    if not np.isnan(p_value):
        # Determine significance stars
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        # Draw line and significance marker
        max_y = max(means) * 1.15
        ax1.plot([0, 1], [max_y, max_y], 'k-', linewidth=1.5)
        ax1.text(0.5, max_y + 0.5, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add p-value text box
        p_text = f'p = {p_value:.4f}' if p_value >= 0.0001 else f'p < 0.0001'
        ax1.text(0.98, 0.98, p_text, transform=ax1.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 2: Paired Data Points (Individual Misses)
    # ========================================================================
    n_pairs = len(speeds_before)
    
    for i in range(n_pairs):
        ax2.plot([0, 1], [speeds_before[i], speeds_after[i]], 
                'o-', color='gray', alpha=0.3, linewidth=1, markersize=4)
    
    # Overlay mean values
    ax2.plot([0, 1], [avg_before, avg_after], 
            'o-', color='red', linewidth=3, markersize=12, 
            label=f'Mean (n={n_pairs})', zorder=10)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'Before\n({window}s)', f'After\n({window}s)'], fontsize=10)
    ax2.set_ylabel('Speed (cm/s)', fontsize=11)
    ax2.set_title('Individual Last-Miss-Before-Hit Events', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add summary statistics
    change = avg_after - avg_before
    pct_change = (change / avg_before * 100) if avg_before != 0 else 0
    
    stats_text = f"N misses: {speed_result['n_last_misses']}\n"
    stats_text += f"Valid pairs: {n_pairs}\n"
    stats_text += f"Mean change: {change:+.1f} cm/s ({pct_change:+.1f}%)\n"
    stats_text += f"t-statistic: {speed_result['t_statistic']:.3f}"
    
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, ha='left', va='bottom',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, "last_miss_before_hit_speed_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved last-miss-before-hit speed analysis plot: {output_path}")
    
    plt.show()


def plot_first_miss_after_hit_speeds(speed_result, output_folder):
    """
    Create visualization for first-miss-after-hit speed analysis
    
    Args:
        speed_result: Dictionary from analyze_first_miss_after_hit_speeds()
        output_folder: Path to save the figure
    """
    if speed_result is None or speed_result['n_valid_speed_pairs'] == 0:
        print("No valid speed data for first-miss-after-hit analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================================================
    # Plot 1: Bar Chart Comparison (Before vs After)
    # ========================================================================
    speeds_before = speed_result['speeds_before_list']
    speeds_after = speed_result['speeds_after_list']
    avg_before = speed_result['avg_speed_before']
    avg_after = speed_result['avg_speed_after']
    p_value = speed_result['p_value']
    window = speed_result['window_seconds']
    
    x_pos = [0, 1]
    means = [avg_before, avg_after]
    labels = [f'Before Zone Entry\\n({window}s window)', f'After Zone Entry\\n({window}s window)']
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax1.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Average Speed (cm/s)', fontsize=11)
    ax1.set_title('Speed Before/After Entering Reward Zone\\n(First Miss After Hit)', 
                  fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add significance annotation
    if not np.isnan(p_value):
        # Determine significance stars
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        # Draw line and significance marker
        max_y = max(means) * 1.15
        ax1.plot([0, 1], [max_y, max_y], 'k-', linewidth=1.5)
        ax1.text(0.5, max_y + 0.5, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add p-value text box
        p_text = f'p = {p_value:.4f}' if p_value >= 0.0001 else f'p < 0.0001'
        ax1.text(0.98, 0.98, p_text, transform=ax1.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 2: Paired Data Points (Individual Misses)
    # ========================================================================
    n_pairs = len(speeds_before)
    
    for i in range(n_pairs):
        ax2.plot([0, 1], [speeds_before[i], speeds_after[i]], 
                'o-', color='gray', alpha=0.3, linewidth=1, markersize=4)
    
    # Overlay mean values
    ax2.plot([0, 1], [avg_before, avg_after], 
            'o-', color='red', linewidth=3, markersize=12, 
            label=f'Mean (n={n_pairs})', zorder=10)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'Before\\n({window}s)', f'After\\n({window}s)'], fontsize=10)
    ax2.set_ylabel('Speed (cm/s)', fontsize=11)
    ax2.set_title('Individual First-Miss-After-Hit Events', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add summary statistics
    change = avg_after - avg_before
    pct_change = (change / avg_before * 100) if avg_before != 0 else 0
    
    stats_text = f"N misses: {speed_result['n_first_misses']}\\n"
    stats_text += f"Valid pairs: {n_pairs}\\n"
    stats_text += f"Mean change: {change:+.1f} cm/s ({pct_change:+.1f}%)\\n"
    stats_text += f"t-statistic: {speed_result['t_statistic']:.3f}"
    
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, ha='left', va='bottom',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, "first_miss_after_hit_speed_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\\nSaved first-miss-after-hit speed analysis plot: {output_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("REWARD HITS/MISSES PATTERN ANALYSIS")
    print("=" * 70)
    
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
    
    # Extract reward zones and events
    print("\nExtracting reward zone entries...")
    all_reward_zones = extract_reward_zone_entries(data['trial_log'])
    print(f"Found {len(all_reward_zones)} reward zone entries")
    
    print("Extracting reward events...")
    reward_events = extract_reward_events(data['trial_log'])
    print(f"Found {len(reward_events)} reward delivery events")
    
    # Classify hits and misses
    print("\nClassifying hits and misses...")
    hits, misses, sequence, zone_times, is_hit_sequence = classify_hits_and_misses(
        all_reward_zones, reward_events
    )
    print(f"  Hits: {len(hits)} ({len(hits)/len(all_reward_zones)*100:.1f}%)")
    print(f"  Misses: {len(misses)} ({len(misses)/len(all_reward_zones)*100:.1f}%)")
    
    # Run analyses
    print("\n" + "="*70)
    print("RUNNING PATTERN ANALYSES")
    print("="*70)
    
    print("\n1. Runs Test (Clustering & Randomness)...")
    runs_result = runs_test(is_hit_sequence)
    if runs_result:
        print(f"   Observed runs: {runs_result['n_runs']}")
        print(f"   Expected runs: {runs_result['expected_runs']:.2f}")
        print(f"   p-value: {runs_result['p_value']:.4f}")
        print(f"   → {runs_result['interpretation']}")
    
    print("\n2. Conditional Probabilities...")
    cond_prob_result = calculate_conditional_probabilities(is_hit_sequence)
    if cond_prob_result:
        print(f"   P(Hit | previous Hit) = {cond_prob_result['p_hit_after_hit']:.3f}")
        print(f"   P(Hit | previous Miss) = {cond_prob_result['p_hit_after_miss']:.3f}")
        print(f"   Overall hit rate = {cond_prob_result['overall_hit_rate']:.3f}")
    
    print("\n3. Autocorrelation...")
    autocorr_result = calculate_autocorrelation(is_hit_sequence, max_lag=10)
    if autocorr_result:
        print(f"   Lag-1 autocorrelation: {autocorr_result['autocorr'][0]:.3f}")
        print(f"   Calculated for lags 1-{len(autocorr_result['lags'])}")
    
    print("\n4. Time-in-Session Effects...")
    session_prog_result = calculate_session_progression(is_hit_sequence, zone_times, n_bins=10)
    if session_prog_result:
        print(f"   Trial-based slope: {session_prog_result['trial_slope']:.6f}")
        print(f"   Trial-based p-value: {session_prog_result['trial_p_value']:.4f}")
        if session_prog_result['trial_p_value'] < 0.05:
            direction = "IMPROVING" if session_prog_result['trial_slope'] > 0 else "DECLINING"
            print(f"   → Performance {direction} across session (p<0.05)")
        else:
            print(f"   → No significant trend across session")
    
    # ========================================================================
    # NEW: First Miss After Hit - Speed Analysis
    # ========================================================================
    print("\n5. First Miss After Hit - Speed Analysis...")
    speed_result_miss = None
    
    if 'treadmill' in data:
        # Prepare treadmill data
        print("   Preparing treadmill data...")
        treadmill_interp = prepare_treadmill_data(data['treadmill'])
        
        if treadmill_interp is not None:
            # Identify first misses after hits
            first_misses = identify_first_miss_after_hit(sequence)
            print(f"   Found {len(first_misses)} first-miss-after-hit events")
            
            if len(first_misses) > 0:
                # Analyze speeds
                speed_result_miss = analyze_first_miss_after_hit_speeds(
                    first_misses, treadmill_interp, window=2.0
                )
                
                if speed_result_miss and speed_result_miss['n_valid_speed_pairs'] > 0:
                    print(f"   Valid speed pairs: {speed_result_miss['n_valid_speed_pairs']}")
                    print(f"   Avg speed before zone: {speed_result_miss['avg_speed_before']:.2f} cm/s")
                    print(f"   Avg speed after zone: {speed_result_miss['avg_speed_after']:.2f} cm/s")
                    speed_change = speed_result_miss['avg_speed_after'] - speed_result_miss['avg_speed_before']
                    print(f"   Change: {speed_change:+.2f} cm/s")
                    
                    if not np.isnan(speed_result_miss['p_value']):
                        print(f"   t-statistic: {speed_result_miss['t_statistic']:.3f}")
                        print(f"   p-value: {speed_result_miss['p_value']:.4f}")
                        
                        if speed_result_miss['p_value'] < 0.05:
                            if speed_change > 0:
                                print(f"   → Mice INCREASE speed after entering miss zones (p<0.05)")
                            else:
                                print(f"   → Mice DECREASE speed after entering miss zones (p<0.05)")
                        else:
                            print(f"   → No significant speed change in miss zones")
                else:
                    print("   No valid speed data available")
            else:
                print("   No first-miss-after-hit events found")
        else:
            print("   Could not prepare treadmill data")
    else:
        print("   Treadmill data not available - skipping speed analysis")
    
    # ========================================================================
    # NEW: Last Hit Before Miss - Speed Analysis
    # ========================================================================
    print("\n6. Last Hit Before Miss - Speed Analysis...")
    speed_result_hit = None
    
    if 'treadmill' in data and treadmill_interp is not None:
        # Identify last hits before misses
        last_hits = identify_last_hit_before_miss(sequence)
        print(f"   Found {len(last_hits)} last-hit-before-miss events")
        
        if len(last_hits) > 0:
            # Analyze speeds
            speed_result_hit = analyze_last_hit_before_miss_speeds(
                last_hits, treadmill_interp, window=2.0
            )
            
            if speed_result_hit and speed_result_hit['n_valid_speed_pairs'] > 0:
                print(f"   Valid speed pairs: {speed_result_hit['n_valid_speed_pairs']}")
                print(f"   Avg speed before zone: {speed_result_hit['avg_speed_before']:.2f} cm/s")
                print(f"   Avg speed after zone: {speed_result_hit['avg_speed_after']:.2f} cm/s")
                speed_change = speed_result_hit['avg_speed_after'] - speed_result_hit['avg_speed_before']
                print(f"   Change: {speed_change:+.2f} cm/s")
                
                if not np.isnan(speed_result_hit['p_value']):
                    print(f"   t-statistic: {speed_result_hit['t_statistic']:.3f}")
                    print(f"   p-value: {speed_result_hit['p_value']:.4f}")
                    
                    if speed_result_hit['p_value'] < 0.05:
                        if speed_change > 0:
                            print(f"   → Mice INCREASE speed after entering hit zones (p<0.05)")
                        else:
                            print(f"   → Mice DECREASE speed after entering hit zones (p<0.05)")
                    else:
                        print(f"   → No significant speed change in hit zones")
            else:
                print("   No valid speed data available")
        else:
            print("   No last-hit-before-miss events found")
    
    # ========================================================================
    # NEW: Last Miss Before Hit - Speed Analysis
    # ========================================================================
    print("\n7. Last Miss Before Hit - Speed Analysis...")
    speed_result_miss_before = None
    
    if 'treadmill' in data and treadmill_interp is not None:
        # Identify last misses before hits
        last_misses = identify_last_miss_before_hit(sequence)
        print(f"   Found {len(last_misses)} last-miss-before-hit events")
        
        if len(last_misses) > 0:
            # Analyze speeds
            speed_result_miss_before = analyze_last_miss_before_hit_speeds(
                last_misses, treadmill_interp, window=2.0
            )
            
            if speed_result_miss_before and speed_result_miss_before['n_valid_speed_pairs'] > 0:
                print(f"   Valid speed pairs: {speed_result_miss_before['n_valid_speed_pairs']}")
                print(f"   Avg speed before zone: {speed_result_miss_before['avg_speed_before']:.2f} cm/s")
                print(f"   Avg speed after zone: {speed_result_miss_before['avg_speed_after']:.2f} cm/s")
                speed_change = speed_result_miss_before['avg_speed_after'] - speed_result_miss_before['avg_speed_before']
                print(f"   Change: {speed_change:+.2f} cm/s")
                
                if not np.isnan(speed_result_miss_before['p_value']):
                    print(f"   t-statistic: {speed_result_miss_before['t_statistic']:.3f}")
                    print(f"   p-value: {speed_result_miss_before['p_value']:.4f}")
                    
                    if speed_result_miss_before['p_value'] < 0.05:
                        if speed_change > 0:
                            print(f"   → Mice INCREASE speed after entering miss zones (p<0.05)")
                        else:
                            print(f"   → Mice DECREASE speed after entering miss zones (p<0.05)")
                    else:
                        print(f"   → No significant speed change in miss zones")
            else:
                print("   No valid speed data available")
        else:
            print("   No last-miss-before-hit events found")
    
    # Create comprehensive visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_all_analyses(runs_result, cond_prob_result, autocorr_result, 
                     session_prog_result, is_hit_sequence, output_folder)
    
    # Plot speed analyses if available
    if speed_result_miss is not None and speed_result_miss['n_valid_speed_pairs'] > 0:
        plot_first_miss_after_hit_speeds(speed_result_miss, output_folder)
    
    if speed_result_hit is not None and speed_result_hit['n_valid_speed_pairs'] > 0:
        plot_last_hit_before_miss_speeds(speed_result_hit, output_folder)
    
    if speed_result_miss_before is not None and speed_result_miss_before['n_valid_speed_pairs'] > 0:
        plot_last_miss_before_hit_speeds(speed_result_miss_before, output_folder)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
