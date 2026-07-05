"""
Test Script for Lick Detection Algorithm

This script allows you to test the lick detection algorithm on any capacitive CSV file.
It will load the data, run the full detection pipeline, and display visualizations
along with summary statistics.

Usage:
    python test_lick_detection.py path/to/your/file.csv
    
    Or run without arguments and you'll be prompted to enter the file path.

Expected CSV format:
    - Must have columns: arduino_timestamp, elapsed_time, and capacitive_value
    - Time column will be derived (elapsed_time or arduino_timestamp/1000)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import lick_detection_algorithm as lda


def load_and_prepare_data(csv_path):
    """Load CSV and prepare Time_sec column."""
    print(f"\nLoading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return None
    
    # Check for required columns
    required_cols = ['capacitive_value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        return None
    
    # Create Time_sec column
    if 'Time_sec' not in df.columns:
        if 'elapsed_time' in df.columns:
            df['Time_sec'] = df['elapsed_time']
            print("✓ Using 'elapsed_time' as Time_sec")
        elif 'arduino_timestamp' in df.columns:
            df['Time_sec'] = df['arduino_timestamp'] / 1000.0
            print("✓ Using 'arduino_timestamp'/1000 as Time_sec")
        else:
            print("✗ No time column found (need 'elapsed_time' or 'arduino_timestamp')")
            return None
    
    print(f"✓ Data prepared: {df['Time_sec'].min():.2f}s to {df['Time_sec'].max():.2f}s")
    return df


def run_lick_detection(df, threshold=None, ili_cutoff=0.3):
    """Run the complete lick detection pipeline."""
    print("\n" + "="*60)
    print("RUNNING LICK DETECTION ALGORITHM")
    print("="*60)
    
    # Step 1: Compute KDE
    print("\n[1/5] Computing KDE baseline...")
    kde_value = lda.compute_KDE(df, 'capacitive_value')
    if kde_value is None:
        print("✗ KDE computation failed")
        return None, None, None, None
    print(f"✓ KDE baseline: {kde_value:.4f}")
    
    # Step 2: Compute KDE normalizations
    print("\n[2/5] Computing KDE normalizations...")
    df_normalized = lda.compute_KDE_normalizations(df, 'capacitive_value', kde_value)
    print(f"✓ Added 'capacitive_value_deviation' column")
    
    # Step 3: Detect events (with dynamic threshold calculation if threshold is None)
    print(f"\n[3/5] Detecting events...")
    events_df, threshold = lda.detect_events_above_threshold(df_normalized, 'capacitive_value', threshold=threshold)
    n_events = events_df['capacitive_value_event'].sum()
    print(f"✓ Detected {n_events} lick events")
    
    # Step 4: Compute inter-lick intervals
    print("\n[4/5] Computing inter-lick intervals...")
    ili_array = lda.compute_inter_lick_intervals(events_df, 'capacitive_value')
    if len(ili_array) > 0:
        print(f"✓ {len(ili_array)} intervals computed")
        print(f"  Mean ILI: {ili_array.mean():.3f}s")
        print(f"  Median ILI: {ili_array.median():.3f}s" if hasattr(ili_array, 'median') else f"  Median ILI: {pd.Series(ili_array).median():.3f}s")
        print(f"  Range: {ili_array.min():.3f}s - {ili_array.max():.3f}s")
    else:
        print(f"✓ No inter-lick intervals (fewer than 2 events)")
    
    # Step 5: Compute lick bouts
    print(f"\n[5/5] Computing lick bouts (ILI cutoff={ili_cutoff}s)...")
    bout_dict = lda.compute_lick_bouts(events_df, 'capacitive_value', ili_cutoff=ili_cutoff)
    n_bouts = bout_dict['bout_count']
    print(f"✓ Detected {n_bouts} lick bouts")
    
    if n_bouts > 0:
        print(f"  Mean bout size: {bout_dict['bout_sizes'].mean():.1f} licks")
        print(f"  Mean bout duration: {bout_dict['bout_durations'].mean():.3f}s")
        print(f"  Total licks in bouts: {bout_dict['bout_sizes'].sum()}")
    
    return df_normalized, events_df, kde_value, bout_dict, threshold


def print_summary_statistics(df_normalized, events_df, bout_dict, threshold, ili_cutoff):
    """Print comprehensive summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Data duration
    duration = df_normalized['Time_sec'].max() - df_normalized['Time_sec'].min()
    print(f"\nSession Duration: {duration:.2f}s ({duration/60:.2f} min)")
    
    # Event statistics
    n_events = events_df['capacitive_value_event'].sum()
    lick_rate = n_events / duration if duration > 0 else 0
    print(f"\nLick Events:")
    print(f"  Total: {n_events}")
    print(f"  Rate: {lick_rate:.2f} licks/s ({lick_rate*60:.1f} licks/min)")
    print(f"  Detection threshold: {threshold}")
    
    # Bout statistics
    n_bouts = bout_dict['bout_count']
    if n_bouts > 0:
        print(f"\nLick Bouts:")
        print(f"  Total: {n_bouts}")
        print(f"  ILI cutoff: {ili_cutoff}s")
        print(f"  Mean size: {bout_dict['bout_sizes'].mean():.1f} ± {bout_dict['bout_sizes'].std():.1f} licks")
        print(f"  Mean duration: {bout_dict['bout_durations'].mean():.3f} ± {bout_dict['bout_durations'].std():.3f}s")
        print(f"  Size range: {bout_dict['bout_sizes'].min():.0f} - {bout_dict['bout_sizes'].max():.0f} licks")
    
    # ILI statistics
    ili_array = lda.compute_inter_lick_intervals(events_df, 'capacitive_value')
    if len(ili_array) > 0:
        print(f"\nInter-Lick Intervals:")
        print(f"  Mean: {ili_array.mean():.3f}s")
        print(f"  Median: {pd.Series(ili_array).median():.3f}s")
        print(f"  Std Dev: {ili_array.std():.3f}s")
        print(f"  Range: {ili_array.min():.3f}s - {ili_array.max():.3f}s")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("LICK DETECTION ALGORITHM - TEST SCRIPT")
    print("="*60)
    
    # Get CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\nEnter path to CSV file: ").strip().strip('"').strip("'")
    
    if not os.path.exists(csv_path):
        print(f"\n✗ File not found: {csv_path}")
        return
    
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    if df is None:
        return
    
    # Get parameters (with defaults)
    print("\n" + "-"*60)
    threshold_input = input("Enter detection threshold [default=dynamic (KDE valley)]: ").strip()
    threshold = float(threshold_input) if threshold_input else None
    
    ili_cutoff_input = input("Enter ILI cutoff for bouts in seconds [default=0.3]: ").strip()
    ili_cutoff = float(ili_cutoff_input) if ili_cutoff_input else 0.3
    
    # Run lick detection
    df_normalized, events_df, kde_value, bout_dict, threshold = run_lick_detection(
        df, threshold=threshold, ili_cutoff=ili_cutoff
    )
    
    if events_df is None:
        print("\n✗ Lick detection failed")
        return
    
    # Print summary statistics
    print_summary_statistics(df_normalized, events_df, bout_dict, threshold, ili_cutoff)
    
    # Generate visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    
    filename = os.path.basename(csv_path)
    fig = lda.plot_summary(
        df_normalized, 
        events_df, 
        column='capacitive_value',
        kde_value=kde_value,
        threshold=threshold,
        bout_dict=bout_dict,
        title=f'Lick Detection Analysis: {filename}',
        show=True
    )
    
    print("\n✓ Visualization displayed")

    # Plot histogram + KDE density overlay for threshold diagnostics
    print("\n" + "="*60)
    print("CAPACITIVE VALUE HISTOGRAM (KDE VALLEY DIAGNOSTIC)")
    print("="*60)
    deviation_col = 'capacitive_value_deviation'
    deviations = df_normalized[deviation_col].dropna()
    # Always use the full data range so rare single-lick events are visible
    x_max = deviations.max() * 1.02
    x_eval = np.linspace(0, x_max, 1000)

    fig_hist, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig_hist.suptitle(f'KDE Valley Threshold Diagnostic: {filename}', fontsize=13)

    # --- Top panel: histogram ---
    ax_top.hist(deviations, bins=100, color='steelblue', edgecolor='white',
                linewidth=0.3, density=True, alpha=0.7, label='Deviation histogram')
    ax_top.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Valley threshold: {threshold:.4f}')
    ax_top.set_ylabel('Density', fontsize=11)
    ax_top.set_title('Deviation Distribution (density-normalised)', fontsize=11)
    ax_top.legend(fontsize=10)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # --- Bottom panel: KDE density curve with FWHM boundary and valley marked ---
    clean = deviations.values[np.isfinite(deviations.values) & (deviations.values >= 0)]
    try:
        kde_curve = stats.gaussian_kde(clean, bw_method='scott')
        density_vals = kde_curve(x_eval)
        ax_bot.plot(x_eval, density_vals, color='darkorange', linewidth=2, label='KDE density')

        from scipy.signal import find_peaks as _fp
        MIN_DEVIATION_GAP = 10.0  # must match _kde_valley_search default
        peaks_idx, _ = _fp(density_vals)
        if len(peaks_idx) > 0:
            noise_peak = peaks_idx[0]
            noise_peak_x = x_eval[noise_peak]
            half_max = density_vals[noise_peak] / 2.0

            ax_bot.axvline(noise_peak_x, color='green', linestyle=':', linewidth=1.5,
                           label=f'Noise peak: {noise_peak_x:.4f}')
            ax_bot.axhline(half_max, color='gray', linestyle=':', linewidth=1,
                           alpha=0.6, label='Half-max (FWHM level)')

            # Min deviation gap boundary (10 deviation units past noise peak)
            gap_x = noise_peak_x + MIN_DEVIATION_GAP
            ax_bot.axvline(gap_x, color='darkorchid', linestyle=':', linewidth=1.5,
                           alpha=0.8, label=f'Min gap boundary (+{MIN_DEVIATION_GAP:.0f}): {gap_x:.4f}')

            # Find FWHM right edge
            right_half = density_vals[noise_peak:]
            below_half = np.where(right_half < half_max)[0]
            if len(below_half) > 0:
                fwhm_right_idx = noise_peak + below_half[0]
                fwhm_right_x = x_eval[fwhm_right_idx]
                ax_bot.axvline(fwhm_right_x, color='goldenrod', linestyle='--', linewidth=1.5,
                               label=f'FWHM noise edge: {fwhm_right_x:.4f}')

                # Fallback = max(FWHM, gap) — mirrors the algorithm fix
                fallback_x = max(fwhm_right_x, gap_x)

                # Find deepest valley past max(FWHM, gap)
                search_start_x = fallback_x
                search_mask = x_eval >= search_start_x
                if search_mask.any():
                    search_start_idx = np.where(search_mask)[0][0]
                    post_search = density_vals[search_start_idx:]
                    v_rel, _ = _fp(-post_search)
                    if len(v_rel) > 0:
                        deepest_rel = v_rel[np.argmin(post_search[v_rel])]
                        valley_idx = search_start_idx + deepest_rel
                        # Only accept if signal peak follows
                        post_v_peaks, _ = _fp(density_vals[valley_idx:])
                        if len(post_v_peaks) > 0:
                            ax_bot.axvline(x_eval[valley_idx], color='red', linestyle='--', linewidth=2,
                                           label=f'Valley (threshold): {x_eval[valley_idx]:.4f}')
                            ax_bot.scatter([x_eval[valley_idx]], [density_vals[valley_idx]],
                                           color='red', s=80, zorder=5)
                        else:
                            ax_bot.axvline(fallback_x, color='red', linestyle='--', linewidth=2,
                                           label=f'Fallback threshold: {fallback_x:.4f}')
                    else:
                        # No valley found — fallback = max(FWHM, noise_peak + 20)
                        ax_bot.axvline(fallback_x, color='red', linestyle='--', linewidth=2,
                                       label=f'Fallback threshold: {fallback_x:.4f}')

            # Mark any signal peaks beyond the noise peak
            for pk in peaks_idx[1:]:
                ax_bot.axvline(x_eval[pk], color='purple', linestyle=':', linewidth=1,
                               alpha=0.6, label=f'Signal peak: {x_eval[pk]:.4f}')
    except Exception as e:
        print(f"  (KDE overlay failed: {e})")

    ax_bot.set_xlabel('Normalized Capacitive Deviation  |  (value − KDE) / KDE  |', fontsize=11)
    ax_bot.set_ylabel('KDE Density', fontsize=11)
    ax_bot.set_title('KDE Density Curve — Valley Detection', fontsize=11)
    ax_bot.set_xlim(left=0, right=x_max * 1.02)
    ax_bot.set_ylim(bottom=0)
    ax_bot.legend(fontsize=9)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    print("✓ Histogram + KDE diagnostic plot displayed")

    # Ask if user wants to save the figure
    save_input = input("\nSave figure? (y/n) [default=n]: ").strip().lower()
    if save_input in ['y', 'yes']:
        # Default output filename in the current working directory
        default_output = os.path.join(os.getcwd(), os.path.splitext(filename)[0] + '_lick_analysis.svg')
        output_path = input(f"Enter output path [default={default_output}]: ").strip()
        
        if not output_path:
            output_path = default_output
        
        # Ensure .svg extension
        if not output_path.lower().endswith('.svg'):
            output_path += '.svg'
        
        try:
            fig.savefig(output_path, format='svg', bbox_inches='tight')
            print(f"✓ Figure saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error saving figure: {e}")
    
    print(f"\nAnalysis complete for: {filename}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
