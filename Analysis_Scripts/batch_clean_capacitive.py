"""
batch_clean_capacitive.py
Batch-cleans capacitive.csv files by removing the noisy header rows that
appear at the start of recordings.

Noise detection:
    The noisy header ends and clean data begins at the first row where the
    forward gap in elapsed_time exceeds `--gap` (default 0.4 s) AND the
    resulting elapsed_time is at least `--min-elapsed` (default 0.85 s).
    If no such gap is found the file is left untouched.

Behaviour:
    - Operates in-place.
    - Use --dry-run to preview what would be removed without writing anything.

Usage examples:
    python batch_clean_capacitive.py                        # folder dialog
    python batch_clean_capacitive.py --dir D:\\my_data
    python batch_clean_capacitive.py --dir D:\\my_data --dry-run
    python batch_clean_capacitive.py --dir D:\\my_data --gap 0.3 --min-elapsed 0.80
"""

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------

def find_clean_start(df, gap_threshold: float, min_clean_elapsed: float) -> int:
    """
    Return the row index where clean data begins.

    Primary method: arduino_timestamp drop.
        Noisy header rows have anomalously high arduino_timestamps (>5000).
        The last such row in the initial segment marks the end of noise; the
        next row is the clean start.  Only the first 300 rows are searched
        (the noisy header is always short).

    Fallback method: elapsed_time gap.
        Scans for the first forward gap in elapsed_time > gap_threshold that
        lands at a value >= min_clean_elapsed.

    Returns 0 (keep everything) if neither method detects noise.
    """
    # -- Primary: arduino_timestamp drop ------------------------------------
    if 'arduino_timestamp' in df.columns:
        arduino_ts = pd.to_numeric(df['arduino_timestamp'], errors='coerce').values
        HIGH_TS_THRESHOLD = 20000
        SEARCH_LIMIT = 300

        last_high_idx = -1
        for i in range(min(SEARCH_LIMIT, len(arduino_ts))):
            if not np.isnan(arduino_ts[i]) and arduino_ts[i] > HIGH_TS_THRESHOLD:
                last_high_idx = i

        if last_high_idx >= 0:
            clean_start = last_high_idx + 1
            if clean_start < len(df):
                return clean_start

    # -- Fallback: elapsed_time gap -----------------------------------------
    elapsed = pd.to_numeric(df['elapsed_time'], errors='coerce').values
    for i in range(1, len(elapsed)):
        prev = elapsed[i - 1]
        curr = elapsed[i]
        if np.isnan(prev) or np.isnan(curr):
            continue
        if (curr - prev) > gap_threshold and curr >= min_clean_elapsed:
            return i

    return 0  # no noise header detected


# ---------------------------------------------------------------------------
# Single-file cleaning
# ---------------------------------------------------------------------------

def clean_file(
    filepath: Path,
    gap_threshold: float,
    min_clean_elapsed: float,
    dry_run: bool,
) -> tuple[int, int] | None:
    """
    Clean one file.  Returns (rows_removed, total_rows), or None on error.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        print(f"  [ERROR] Could not read {filepath.name}: {exc}")
        return None

    if 'elapsed_time' not in df.columns:
        print(f"  [SKIP]  Missing 'elapsed_time' column: {filepath.name}")
        return None

    clean_start = find_clean_start(df, gap_threshold, min_clean_elapsed)
    rows_removed = clean_start
    total_rows = len(df)

    if rows_removed == 0:
        print(f"  [OK]    No noise detected — {total_rows} rows kept:  {filepath.name}")
        return (0, total_rows)

    kept = total_rows - rows_removed
    first_clean_elapsed = pd.to_numeric(df['elapsed_time'], errors='coerce').iloc[clean_start]

    if dry_run:
        print(
            f"  [DRY]   Would remove {rows_removed} rows "
            f"(keep {kept}/{total_rows}, clean start at elapsed={first_clean_elapsed:.3f} s):  "
            f"{filepath.name}"
        )
        return (rows_removed, total_rows)

    clean_df = df.iloc[clean_start:].reset_index(drop=True)

    try:
        clean_df.to_csv(filepath, index=False)
    except Exception as exc:
        print(f"  [ERROR] Could not write {filepath.name}: {exc}")
        return None

    print(
        f"  [DONE]  Removed {rows_removed} rows "
        f"(kept {kept}/{total_rows}, clean start at elapsed={first_clean_elapsed:.3f} s):  "
        f"{filepath.name}"
    )
    return (rows_removed, total_rows)


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def find_capacitive_files(directory: str) -> list[Path]:
    """Recursively find all files whose name ends with 'capacitive.csv'."""
    return sorted(Path(directory).rglob('*capacitive.csv'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Batch-remove noisy headers from capacitive.csv files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dir', type=str, default=None,
        help='Root directory to search (opens a folder dialog if omitted).',
    )
    parser.add_argument(
        '--gap', type=float, default=0.4,
        help='Forward elapsed_time gap (seconds) that signals end of noise.',
    )
    parser.add_argument(
        '--min-elapsed', type=float, default=0.85,
        help='Minimum elapsed_time value expected at the start of clean data.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview what would be removed without modifying any files.',
    )
    args = parser.parse_args()

    # Resolve search directory
    if args.dir:
        search_dir = args.dir
    else:
        root = tk.Tk()
        root.withdraw()
        search_dir = filedialog.askdirectory(
            title='Select root directory containing capacitive.csv files'
        )
        root.destroy()
        if not search_dir:
            print("No directory selected. Exiting.")
            sys.exit(0)

    if not os.path.isdir(search_dir):
        print(f"Directory not found: {search_dir}")
        sys.exit(1)

    print(f"\nSearching for *capacitive.csv under: {search_dir}")
    files = find_capacitive_files(search_dir)

    if not files:
        print("No *capacitive.csv files found.")
        sys.exit(0)

    print(f"Found {len(files)} file(s).")
    print(f"Settings: gap_threshold={args.gap} s, min_clean_elapsed={args.min_elapsed} s")
    if args.dry_run:
        print("DRY RUN — no files will be modified.")
    print()

    total_files   = len(files)
    cleaned_count = 0
    skipped_count = 0
    error_count   = 0
    total_removed = 0

    for filepath in files:
        result = clean_file(
            filepath,
            gap_threshold=args.gap,
            min_clean_elapsed=args.min_elapsed,
            dry_run=args.dry_run,
        )
        if result is None:
            error_count += 1
        else:
            rows_removed, _ = result
            total_removed += rows_removed
            if rows_removed > 0:
                cleaned_count += 1
            else:
                skipped_count += 1

    # Summary
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Files found          : {total_files}")
    print(f"  Files cleaned        : {cleaned_count}")
    print(f"  Files already clean  : {skipped_count}")
    print(f"  Errors               : {error_count}")
    print(f"  Total rows removed   : {total_removed}")
    print("=" * 55)


if __name__ == '__main__':
    main()
