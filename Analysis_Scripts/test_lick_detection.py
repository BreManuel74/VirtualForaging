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
import pandas as pd
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
    threshold_input = input("Enter detection threshold [default=dynamic (max/2)]: ").strip()
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
    
    # Ask if user wants to save the figure
    save_input = input("\nSave figure? (y/n) [default=n]: ").strip().lower()
    if save_input in ['y', 'yes']:
        # Default output filename based on input CSV
        default_output = os.path.splitext(csv_path)[0] + '_lick_analysis.svg'
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
