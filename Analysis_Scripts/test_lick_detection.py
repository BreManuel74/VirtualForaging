"""
Test Script for Lick Detection Algorithm

This script opens a folder-picker dialog, finds every CSV file whose name contains
"capacitive", and runs the full lick detection pipeline on each file sequentially,
displaying visualizations and summary statistics along the way.

Usage:
    python test_lick_detection.py
    (a folder selection window will pop up)

Expected CSV format:
    - Must have columns: arduino_timestamp, elapsed_time, and capacitive_value
    - Time column will be derived (elapsed_time or arduino_timestamp/1000)
"""

import os
import glob
import pickle
import hashlib
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import lick_detection_algorithm as lda

# Cache directory for KDE values (same cache used by longitudinal_analysis_new_hallway.py,
# so a file's KDE baseline only ever needs to be computed once across scripts). Fitting and
# evaluating a gaussian_kde over an entire raw capacitive signal is the slowest step in this
# pipeline, so skipping it on repeat runs of the same file is the main speedup here.
script_dir = os.path.dirname(os.path.abspath(__file__))
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


def _threshold_cache_file(capacitive_filepath, min_gap_floor, min_gap_ceiling, min_gap_fraction):
    """Build the cache path for a dynamic-threshold result.

    The dynamic KDE-valley threshold search (in detect_events_above_threshold) is the
    slowest step in the pipeline (fitting/evaluating a gaussian_kde over every deviation
    sample) and is deterministic for a given file and set of gap parameters, so it is
    cached the same way as the KDE baseline above, keyed on those parameters as well.
    """
    file_hash = get_file_hash(capacitive_filepath)
    key_input = f"{file_hash}_{min_gap_floor}_{min_gap_ceiling}_{min_gap_fraction}".encode('utf-8')
    key_hash = hashlib.md5(key_input).hexdigest()
    return os.path.join(KDE_CACHE_DIR, f"{key_hash}_threshold.pkl")


def get_cached_threshold(capacitive_filepath, min_gap_floor, min_gap_ceiling, min_gap_fraction):
    """Retrieve a cached dynamic threshold for a file + gap-parameter combination."""
    cache_file = _threshold_cache_file(capacitive_filepath, min_gap_floor, min_gap_ceiling, min_gap_fraction)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)['threshold']
        except Exception as e:
            print(f"Warning: Cache read failed for {capacitive_filepath}: {e}")
            return None
    return None


def cache_threshold_value(capacitive_filepath, min_gap_floor, min_gap_ceiling, min_gap_fraction, threshold):
    """Cache a dynamic threshold for a file + gap-parameter combination."""
    cache_file = _threshold_cache_file(capacitive_filepath, min_gap_floor, min_gap_ceiling, min_gap_fraction)

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'threshold': threshold}, f)
    except Exception as e:
        print(f"Warning: Cache write failed for {capacitive_filepath}: {e}")


def select_folder():
    """Open a tkinter dialog for the user to pick a folder."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="Select folder containing capacitive CSV files")
    root.destroy()
    return folder


def find_capacitive_csv_files(folder_path):
    """Find all CSV files in folder_path whose filename contains 'capacitive' (case-insensitive)."""
    all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))
    return sorted(f for f in all_csvs if 'capacitive' in os.path.basename(f).lower())


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


def run_lick_detection(df, csv_path, threshold=None, ili_cutoff=0.3,
                        min_gap_floor=6.1, min_gap_ceiling=15.0, min_gap_fraction=0.2,
                        debug=False):
    """Run the complete lick detection pipeline."""
    print("\n" + "="*60)
    print("RUNNING LICK DETECTION ALGORITHM")
    print("="*60)
    
    # Step 1: Compute KDE (cached per-file, since fitting/evaluating the KDE over the
    # full raw signal is the slowest step and the result never changes unless the file does)
    print("\n[1/5] Computing KDE baseline...")
    kde_value = get_cached_kde(csv_path)
    cache_hit = kde_value is not None
    if not cache_hit:
        kde_value = lda.compute_KDE(df, 'capacitive_value')
        if kde_value is None:
            print("✗ KDE computation failed")
            return None, None, None, None
        cache_kde_value(csv_path, kde_value)
    print(f"✓ KDE baseline: {kde_value:.4f}" + (" (cached)" if cache_hit else ""))
    
    # Step 2: Compute KDE normalizations
    print("\n[2/5] Computing KDE normalizations...")
    df_normalized = lda.compute_KDE_normalizations(df, 'capacitive_value', kde_value)
    print(f"✓ Added 'capacitive_value_deviation' column")
    
    # Step 3: Detect events (with dynamic threshold calculation if threshold is None)
    print(f"\n[3/5] Detecting events...")
    # The dynamic KDE-valley threshold search is the slowest step in the pipeline, so
    # reuse a cached result for this file + gap-parameter combo when threshold wasn't
    # explicitly provided (an explicit threshold always bypasses the dynamic search anyway).
    threshold_cache_hit = False
    effective_threshold = threshold
    if effective_threshold is None:
        cached_threshold = get_cached_threshold(csv_path, min_gap_floor, min_gap_ceiling, min_gap_fraction)
        if cached_threshold is not None:
            effective_threshold = cached_threshold
            threshold_cache_hit = True

    events_df, threshold = lda.detect_events_above_threshold(
        df_normalized, 'capacitive_value', threshold=effective_threshold,
        min_gap_floor=min_gap_floor, min_gap_ceiling=min_gap_ceiling,
        min_gap_fraction=min_gap_fraction, debug=debug
    )

    if threshold_cache_hit:
        print(f"✓ Loaded cached dynamic threshold: {threshold:.4f}")
    elif effective_threshold is None:
        # We just computed the dynamic threshold fresh; cache it for next time.
        cache_threshold_value(csv_path, min_gap_floor, min_gap_ceiling, min_gap_fraction, threshold)

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


def process_file(csv_path, threshold=None, ili_cutoff=0.3, auto_save=False, output_dir=None,
                  min_gap_floor=7.0, min_gap_ceiling=15.0, min_gap_fraction=0.2, debug=False):
    """Run the full lick detection pipeline and visualizations for a single CSV file."""
    print("\n" + "="*60)
    print(f"PROCESSING: {os.path.basename(csv_path)}")
    print("="*60)

    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    if df is None:
        return

    # Run lick detection
    df_normalized, events_df, kde_value, bout_dict, threshold = run_lick_detection(
        df, csv_path, threshold=threshold, ili_cutoff=ili_cutoff,
        min_gap_floor=min_gap_floor, min_gap_ceiling=min_gap_ceiling,
        min_gap_fraction=min_gap_fraction, debug=debug
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

    # Save or prompt to save the figure
    save = auto_save
    if not auto_save:
        save_input = input("\nSave figure? (y/n) [default=n]: ").strip().lower()
        save = save_input in ['y', 'yes']

    if save:
        default_dir = output_dir if output_dir else os.getcwd()
        default_output = os.path.join(default_dir, os.path.splitext(filename)[0] + '_lick_analysis.svg')
        output_path = default_output
        if not auto_save:
            output_path_input = input(f"Enter output path [default={default_output}]: ").strip()
            if output_path_input:
                output_path = output_path_input

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


def main():
    """Main execution function: select a folder, then process every capacitive CSV in it."""
    print("\n" + "="*60)
    print("LICK DETECTION ALGORITHM - BATCH TEST SCRIPT")
    print("="*60)

    print("\nSelect a folder containing capacitive CSV files...")
    folder_path = select_folder()
    if not folder_path:
        print("\n✗ No folder selected. Exiting.")
        return

    csv_files = find_capacitive_csv_files(folder_path)
    if not csv_files:
        print(f"\n✗ No CSV files containing 'capacitive' found in: {folder_path}")
        return

    print(f"\n✓ Found {len(csv_files)} capacitive CSV file(s) in: {folder_path}")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Get parameters once (applied to every file)
    print("\n" + "-"*60)
    threshold_input = input("Enter detection threshold [default=dynamic (KDE valley)]: ").strip()
    threshold = float(threshold_input) if threshold_input else None

    ili_cutoff_input = input("Enter ILI cutoff for bouts in seconds [default=0.3]: ").strip()
    ili_cutoff = float(ili_cutoff_input) if ili_cutoff_input else 0.3

    min_gap_floor = 6.1  # Minimum allowed adaptive gap (noise/signal boundary floor)

    min_gap_ceiling_input = input("Enter maximum adaptive gap ceiling [default=15.0]: ").strip()
    min_gap_ceiling = float(min_gap_ceiling_input) if min_gap_ceiling_input else 15.0

    auto_save_input = input("Auto-save figures for every file? (y/n) [default=n]: ").strip().lower()
    auto_save = auto_save_input in ['y', 'yes']

    for csv_path in csv_files:
        process_file(csv_path, threshold=threshold, ili_cutoff=ili_cutoff,
                      auto_save=auto_save, output_dir=folder_path,
                      min_gap_floor=min_gap_floor, min_gap_ceiling=min_gap_ceiling)

    print("\n" + "="*60)
    print(f"BATCH COMPLETE: processed {len(csv_files)} file(s)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
