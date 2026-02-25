"""
Analysis of Time Between Reward Events

Examines whether reward events occur at consistent time intervals:
- Inter-reward times
- Distribution and consistency
- Temporal spacing patterns

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
    """Load trial log CSV file"""
    trial_log_files = [f for f in os.listdir(folder_path) if 'trial_log.csv' in f]
    
    if not trial_log_files:
        print("Error: Missing required file (trial_log.csv)")
        return None
    
    print(f"\nLoading files:")
    print(f"  - {trial_log_files[0]}")
    
    data = {
        'trial_log': pd.read_csv(os.path.join(folder_path, trial_log_files[0]), engine='python')
    }
    
    print("Files loaded successfully.\n")
    return data


def extract_reward_events(trial_log_df):
    """
    Extract reward event times
    
    Returns:
        List of reward times sorted chronologically
    """
    reward_times = []
    
    for trial_idx in range(len(trial_log_df)):
        reward_time = pd.to_numeric(trial_log_df.loc[trial_idx, 'reward_event'], errors='coerce')
        if pd.notna(reward_time) and reward_time > 0:
            reward_times.append(reward_time)
    
    # Sort by time
    reward_times.sort()
    
    return reward_times


def calculate_inter_reward_times(reward_times):
    """
    Calculate times between consecutive rewards
    
    Returns:
        dict with inter-reward times and statistics
    """
    if len(reward_times) < 2:
        return None
    
    # Calculate inter-reward times
    inter_times = []
    for i in range(1, len(reward_times)):
        inter_time = reward_times[i] - reward_times[i-1]
        inter_times.append(inter_time)
    
    # Calculate statistics
    mean_time = np.mean(inter_times)
    std_time = np.std(inter_times)
    cv_time = std_time / mean_time if mean_time > 0 else np.nan  # Coefficient of variation
    
    return {
        'inter_times': np.array(inter_times),
        'reward_times': np.array(reward_times),
        'mean_time': mean_time,
        'std_time': std_time,
        'cv_time': cv_time,
        'min_time': np.min(inter_times),
        'max_time': np.max(inter_times),
        'median_time': np.median(inter_times),
        'n_rewards': len(reward_times),
        'total_session_time': reward_times[-1] - reward_times[0]
    }


def plot_reward_time_analysis(results, output_folder):
    """Create comprehensive plot of reward time spacing analysis"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Plot 1: Inter-Reward Times Over Session (Top Left)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    reward_numbers = np.arange(1, len(results['inter_times']) + 1)
    ax1.plot(reward_numbers, results['inter_times'], 'o-', 
            color='#1f77b4', linewidth=2, markersize=8, alpha=0.7)
    
    # Add mean line
    mean_time = results['mean_time']
    ax1.axhline(y=mean_time, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_time:.2f} s')
    
    # Add ±1 SD bands
    std_time = results['std_time']
    ax1.axhline(y=mean_time + std_time, color='red', linestyle=':', 
               linewidth=1, alpha=0.5, label=f'±1 SD')
    ax1.axhline(y=mean_time - std_time, color='red', linestyle=':', 
               linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Reward Interval Number', fontsize=12)
    ax1.set_ylabel('Time Between Rewards (s)', fontsize=12)
    ax1.set_title('Inter-Reward Times Across Session', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(bottom=0)
    
    # ========================================================================
    # Plot 2: Distribution of Inter-Reward Times (Top Right)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    n_bins = min(20, len(results['inter_times']) // 2 + 1)
    counts, bins, patches = ax2.hist(results['inter_times'], bins=n_bins, 
                                     color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # Add mean and median lines
    ax2.axvline(x=mean_time, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_time:.2f} s')
    ax2.axvline(x=results['median_time'], color='orange', linestyle='--', linewidth=2,
               label=f'Median = {results["median_time"]:.2f} s')
    
    ax2.set_xlabel('Time Between Rewards (s)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Inter-Reward Times', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ========================================================================
    # Plot 3: Cumulative Session Time vs Reward Number (Bottom Left)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Cumulative session time for each reward
    cumulative_times = results['reward_times'] - results['reward_times'][0]
    reward_numbers_cum = np.arange(1, len(cumulative_times) + 1)
    
    ax3.plot(reward_numbers_cum, cumulative_times, 'o-', 
            color='#ff7f0e', linewidth=2, markersize=8, alpha=0.7)
    
    # Add linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        reward_numbers_cum, cumulative_times
    )
    
    fit_line = slope * reward_numbers_cum + intercept
    ax3.plot(reward_numbers_cum, fit_line, '--', color='red', linewidth=2, 
            label=f'Linear fit (slope={slope:.2f} s/reward, R²={r_value**2:.3f})')
    
    ax3.set_xlabel('Reward Number', fontsize=12)
    ax3.set_ylabel('Cumulative Session Time (s)', fontsize=12)
    ax3.set_title('Session Time vs Reward Number', fontsize=13, fontweight='bold')
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
    summary_lines = ["REWARD TIME SPACING ANALYSIS\n" + "="*45]
    summary_lines.append("")
    summary_lines.append(f"Total rewards: {results['n_rewards']}")
    summary_lines.append(f"Number of intervals: {len(results['inter_times'])}")
    summary_lines.append(f"Total session time: {results['total_session_time']:.2f} s")
    summary_lines.append(f"                    ({results['total_session_time']/60:.2f} min)")
    summary_lines.append("")
    
    summary_lines.append("TIME SPACING STATISTICS:")
    summary_lines.append(f"  Mean time: {results['mean_time']:.2f} s")
    summary_lines.append(f"  Median time: {results['median_time']:.2f} s")
    summary_lines.append(f"  Std deviation: {results['std_time']:.2f} s")
    summary_lines.append(f"  Coefficient of variation: {results['cv_time']:.2%}")
    summary_lines.append(f"  Min time: {results['min_time']:.2f} s")
    summary_lines.append(f"  Max time: {results['max_time']:.2f} s")
    summary_lines.append(f"  Range: {results['max_time'] - results['min_time']:.2f} s")
    summary_lines.append("")
    
    # Calculate reward rate
    reward_rate = results['n_rewards'] / (results['total_session_time'] / 60)  # rewards per minute
    summary_lines.append("REWARD RATE:")
    summary_lines.append(f"  Average: {reward_rate:.2f} rewards/min")
    summary_lines.append(f"  Expected interval: {60/reward_rate:.2f} s")
    summary_lines.append("")
    
    # Consistency interpretation
    cv_time = results['cv_time']
    if cv_time < 0.15:
        consistency = "HIGHLY CONSISTENT"
    elif cv_time < 0.30:
        consistency = "MODERATELY CONSISTENT"
    elif cv_time < 0.50:
        consistency = "SOMEWHAT VARIABLE"
    else:
        consistency = "HIGHLY VARIABLE"
    
    summary_lines.append("CONSISTENCY ASSESSMENT:")
    summary_lines.append(f"  Time spacing: {consistency}")
    summary_lines.append(f"  (CV < 15% = highly consistent)")
    summary_lines.append("")
    
    # Interpretation
    if cv_time < 0.30:
        summary_lines.append("→ Rewards occur at relatively")
        summary_lines.append("  REGULAR time intervals")
    else:
        summary_lines.append("→ Rewards occur at IRREGULAR")
        summary_lines.append("  time intervals")
        summary_lines.append("")
        summary_lines.append("  High variability suggests")
        summary_lines.append("  performance-based (not fixed-")
        summary_lines.append("  interval) reward schedule")
    
    summary_text = '\n'.join(summary_lines)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
            va='top', ha='left', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, "reward_time_spacing_analysis.svg")
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"\nSaved analysis plot: {output_path}")
    
    plt.show()


def main():
    """Main execution function"""
    print("=" * 70)
    print("REWARD TIME SPACING ANALYSIS")
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
    
    # Extract reward events
    print("Extracting reward events...")
    reward_times = extract_reward_events(data['trial_log'])
    print(f"Found {len(reward_times)} reward events")
    
    if len(reward_times) < 2:
        print("\nError: Need at least 2 reward events to analyze spacing")
        return
    
    # Calculate inter-reward times
    print("\nCalculating inter-reward times...")
    results = calculate_inter_reward_times(reward_times)
    
    if results is None:
        print("Error: Could not calculate inter-reward times")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("TIME SPACING SUMMARY")
    print("="*70)
    print(f"\nTotal rewards: {results['n_rewards']}")
    print(f"Number of intervals analyzed: {len(results['inter_times'])}")
    print(f"Total session time: {results['total_session_time']:.2f} s ({results['total_session_time']/60:.2f} min)")
    
    print(f"\nTime Spacing Statistics:")
    print(f"  Mean: {results['mean_time']:.2f} ± {results['std_time']:.2f} s")
    print(f"  Median: {results['median_time']:.2f} s")
    print(f"  Range: {results['min_time']:.2f} to {results['max_time']:.2f} s")
    print(f"  Coefficient of Variation: {results['cv_time']:.2%}")
    
    reward_rate = results['n_rewards'] / (results['total_session_time'] / 60)
    print(f"\nReward Rate:")
    print(f"  {reward_rate:.2f} rewards/min")
    print(f"  (Expected interval: {60/reward_rate:.2f} s)")
    
    print(f"\nConsistency Assessment:")
    if results['cv_time'] < 0.15:
        print("  Time spacing is HIGHLY CONSISTENT (CV < 15%)")
    elif results['cv_time'] < 0.30:
        print("  Time spacing is MODERATELY CONSISTENT (CV 15-30%)")
    elif results['cv_time'] < 0.50:
        print("  Time spacing is SOMEWHAT VARIABLE (CV 30-50%)")
    else:
        print("  Time spacing is HIGHLY VARIABLE (CV > 50%)")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_reward_time_analysis(results, output_folder)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
