"""
Analysis of Distance Between Reward Events

Examines whether reward events occur at consistent distance intervals:
- Inter-reward distances
- Distribution and consistency
- Temporal vs spatial spacing

Author: Brenna Manuel
Created: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from scipy import stats

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


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


def uniformly_sample_treadmill_distance(treadmill_df):
    """
    Uniformly sample treadmill distance at 50 Hz.

    Returns cumulative distance traveled in meters, sampled at the
    treadmill's native 50 Hz rate (not tied to the capacitive timeline).
    """
    # Find first non-zero distance to use as baseline
    non_zero_distances = treadmill_df['distance'][treadmill_df['distance'] != 0]
    start_distance = non_zero_distances.iloc[0] if len(non_zero_distances) > 0 else 0

    # Calculate distance moved from start (mm)
    distance_moved = treadmill_df['distance'] - start_distance

    # Uniformly sample at 50 Hz over treadmill's own time range
    time_min = treadmill_df['global_time'].min()
    time_max = treadmill_df['global_time'].max()
    uniform_time = np.arange(time_min, time_max, 1.0 / 50.0)

    uniform_distance = np.interp(
        uniform_time,
        treadmill_df['global_time'].values,
        distance_moved.values
    ) / 1000.0  # Convert mm to meters

    return pd.Series(uniform_distance, index=uniform_time)


def extract_reward_events_with_distance(trial_log_df, distance_interp):
    """
    Extract reward events and their cumulative distances.

    Both reward_event (active zone delivery) and hits_event (inactive zone correct stop)
    are treated as synonymous hits and included.

    Returns:
        List of (reward_time, cumulative_distance) tuples
    """
    event_times = []

    # Active zone reward deliveries
    for trial_idx in range(len(trial_log_df)):
        reward_time = pd.to_numeric(trial_log_df.loc[trial_idx, 'reward_event'], errors='coerce')
        if pd.notna(reward_time) and reward_time > 0:
            event_times.append(reward_time)

    # Inactive zone correct stops (also hits)
    if 'hits_event' in trial_log_df.columns:
        for trial_idx in range(len(trial_log_df)):
            hits_time = pd.to_numeric(trial_log_df.loc[trial_idx, 'hits_event'], errors='coerce')
            if pd.notna(hits_time) and hits_time > 0:
                event_times.append(hits_time)

    reward_events_with_dist = []
    for event_time in event_times:
        # Find cumulative distance at event time
        if event_time in distance_interp.index:
            cum_distance = distance_interp.loc[event_time]
        else:
            # Find nearest time point
            idx = distance_interp.index.searchsorted(event_time)
            if idx < len(distance_interp):
                cum_distance = distance_interp.iloc[idx]
            else:
                cum_distance = distance_interp.iloc[-1]

        reward_events_with_dist.append((event_time, cum_distance))

    # Sort by time
    reward_events_with_dist.sort(key=lambda x: x[0])
    
    return reward_events_with_dist


def calculate_inter_reward_distances(reward_events_with_dist):
    """
    Calculate distances between consecutive rewards
    
    Returns:
        dict with inter-reward distances and statistics
    """
    if len(reward_events_with_dist) < 2:
        return None
    
    times = [r[0] for r in reward_events_with_dist]
    distances = [r[1] for r in reward_events_with_dist]
    
    # Calculate inter-reward distances
    inter_distances = []
    for i in range(1, len(distances)):
        inter_dist = distances[i] - distances[i-1]
        inter_distances.append(inter_dist)
    
    # Calculate inter-reward times
    inter_times = []
    for i in range(1, len(times)):
        inter_time = times[i] - times[i-1]
        inter_times.append(inter_time)
    
    # Calculate statistics
    mean_dist = np.mean(inter_distances)
    std_dist = np.std(inter_distances)
    cv_dist = std_dist / mean_dist if mean_dist > 0 else np.nan  # Coefficient of variation
    
    mean_time = np.mean(inter_times)
    std_time = np.std(inter_times)
    cv_time = std_time / mean_time if mean_time > 0 else np.nan
    
    return {
        'inter_distances': np.array(inter_distances),
        'inter_times': np.array(inter_times),
        'cumulative_distances': np.array(distances),
        'reward_times': np.array(times),
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'cv_distance': cv_dist,
        'min_distance': np.min(inter_distances),
        'max_distance': np.max(inter_distances),
        'mean_time': mean_time,
        'std_time': std_time,
        'cv_time': cv_time,
        'n_rewards': len(reward_events_with_dist)
    }


def plot_reward_distance_analysis(results, output_folder):
    """Create comprehensive plot of reward spacing analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Plot 1: Inter-Reward Distances Over Time (Top Left)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    reward_numbers = np.arange(1, len(results['inter_distances']) + 1)
    ax1.plot(reward_numbers, results['inter_distances'], 'o-', 
            color='#1f77b4', linewidth=2, markersize=8, alpha=0.7)
    
    # Add mean line
    mean_dist = results['mean_distance']
    ax1.axhline(y=mean_dist, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_dist:.2f} m')
    
    # Add ±1 SD bands
    std_dist = results['std_distance']
    ax1.axhline(y=mean_dist + std_dist, color='red', linestyle=':', 
               linewidth=1, alpha=0.5, label=f'±1 SD')
    ax1.axhline(y=mean_dist - std_dist, color='red', linestyle=':', 
               linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Reward Interval Number', fontsize=12)
    ax1.set_ylabel('Distance Between Rewards (m)', fontsize=12)
    ax1.set_title('Inter-Reward Distances Across Session', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(bottom=0)
    
    # ========================================================================
    # Plot 2: Distribution of Inter-Reward Distances (Top Right)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    n_bins = min(20, len(results['inter_distances']) // 2 + 1)
    counts, bins, patches = ax2.hist(results['inter_distances'], bins=n_bins, 
                                     color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # Add mean line
    ax2.axvline(x=mean_dist, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_dist:.2f} m')
    
    ax2.set_xlabel('Distance Between Rewards (m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Inter-Reward Distances', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 3: Cumulative Distance vs Reward Number (Bottom Left)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    reward_numbers_cum = np.arange(1, len(results['cumulative_distances']) + 1)
    ax3.plot(reward_numbers_cum, results['cumulative_distances'], 'o-', 
            color='#ff7f0e', linewidth=2, markersize=8, alpha=0.7)
    
    # Add linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        reward_numbers_cum, results['cumulative_distances']
    )
    
    fit_line = slope * reward_numbers_cum + intercept
    ax3.plot(reward_numbers_cum, fit_line, '--', color='red', linewidth=2, 
            label=f'Linear fit (slope={slope:.2f} m/reward)')
    
    ax3.set_xlabel('Reward Number', fontsize=12)
    ax3.set_ylabel('Cumulative Distance (m)', fontsize=12)
    ax3.set_title('Cumulative Distance vs Reward Number', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(bottom=0)
    
    # ========================================================================
    # Plot 4: Summary Statistics (Bottom Right)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Compile summary text
    summary_lines = ["REWARD SPACING ANALYSIS\n" + "="*45]
    summary_lines.append("")
    summary_lines.append(f"Total rewards: {results['n_rewards']}")
    summary_lines.append(f"Number of intervals: {len(results['inter_distances'])}")
    summary_lines.append(f"Total distance covered: {results['cumulative_distances'][-1]:.2f} m")
    summary_lines.append("")
    
    summary_lines.append("DISTANCE SPACING:")
    summary_lines.append(f"  Mean distance: {results['mean_distance']:.2f} m")
    summary_lines.append(f"  Std deviation: {results['std_distance']:.2f} m")
    summary_lines.append(f"  Coefficient of variation: {results['cv_distance']:.2%}")
    summary_lines.append(f"  Min distance: {results['min_distance']:.2f} m")
    summary_lines.append(f"  Max distance: {results['max_distance']:.2f} m")
    summary_lines.append(f"  Range: {results['max_distance'] - results['min_distance']:.2f} m")
    summary_lines.append("")
    
    summary_lines.append("TIME SPACING:")
    summary_lines.append(f"  Mean time: {results['mean_time']:.2f} s")
    summary_lines.append(f"  Std deviation: {results['std_time']:.2f} s")
    summary_lines.append(f"  Coefficient of variation: {results['cv_time']:.2%}")
    summary_lines.append("")
    
    # Consistency interpretation
    cv_dist = results['cv_distance']
    if cv_dist < 0.15:
        consistency = "HIGHLY CONSISTENT"
    elif cv_dist < 0.30:
        consistency = "MODERATELY CONSISTENT"
    elif cv_dist < 0.50:
        consistency = "SOMEWHAT VARIABLE"
    else:
        consistency = "HIGHLY VARIABLE"
    
    summary_lines.append("CONSISTENCY ASSESSMENT:")
    summary_lines.append(f"  Distance spacing: {consistency}")
    summary_lines.append(f"  (CV < 15% = highly consistent)")
    summary_lines.append("")
    
    # Compare distance vs time consistency
    if results['cv_distance'] < results['cv_time']:
        summary_lines.append("→ Rewards spaced more consistently by")
        summary_lines.append("  DISTANCE than by TIME")
    else:
        summary_lines.append("→ Rewards spaced more consistently by")
        summary_lines.append("  TIME than by DISTANCE")
    
    summary_text = '\n'.join(summary_lines)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
            va='top', ha='left', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, "reward_distance_spacing_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved analysis plot: {output_path}")
    
    plt.show()


def main():
    """Main execution function"""
    print("=" * 70)
    print("REWARD DISTANCE SPACING ANALYSIS")
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
    
    # Sample treadmill distance at native 50 Hz
    print("Sampling treadmill distance at 50 Hz...")
    distance_interp = uniformly_sample_treadmill_distance(data['treadmill'])
    print(f"Total distance range: 0 to {distance_interp.max():.2f} meters")
    
    # Extract reward events with distances
    print("\nExtracting reward events...")
    reward_events_with_dist = extract_reward_events_with_distance(
        data['trial_log'], distance_interp
    )
    print(f"Found {len(reward_events_with_dist)} reward events")
    
    if len(reward_events_with_dist) < 2:
        print("\nError: Need at least 2 reward events to analyze spacing")
        return
    
    # Calculate inter-reward distances
    print("\nCalculating inter-reward distances...")
    results = calculate_inter_reward_distances(reward_events_with_dist)
    
    if results is None:
        print("Error: Could not calculate inter-reward distances")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("DISTANCE SPACING SUMMARY")
    print("="*70)
    print(f"\nTotal rewards: {results['n_rewards']}")
    print(f"Number of intervals analyzed: {len(results['inter_distances'])}")
    print(f"\nDistance Spacing:")
    print(f"  Mean: {results['mean_distance']:.2f} ± {results['std_distance']:.2f} m")
    print(f"  Range: {results['min_distance']:.2f} to {results['max_distance']:.2f} m")
    print(f"  Coefficient of Variation: {results['cv_distance']:.2%}")
    
    print(f"\nTime Spacing:")
    print(f"  Mean: {results['mean_time']:.2f} ± {results['std_time']:.2f} s")
    print(f"  Coefficient of Variation: {results['cv_time']:.2%}")
    
    print(f"\nConsistency Assessment:")
    if results['cv_distance'] < 0.15:
        print("  Distance spacing is HIGHLY CONSISTENT (CV < 15%)")
    elif results['cv_distance'] < 0.30:
        print("  Distance spacing is MODERATELY CONSISTENT (CV 15-30%)")
    elif results['cv_distance'] < 0.50:
        print("  Distance spacing is SOMEWHAT VARIABLE (CV 30-50%)")
    else:
        print("  Distance spacing is HIGHLY VARIABLE (CV > 50%)")
    
    if results['cv_distance'] < results['cv_time']:
        print("\n  → Rewards are spaced more consistently by DISTANCE than by TIME")
    else:
        print("\n  → Rewards are spaced more consistently by TIME than by DISTANCE")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_reward_distance_analysis(results, output_folder)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
