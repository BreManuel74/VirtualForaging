"""
batch_clean_treadmill.py
Batch-cleans treadmill.csv files by removing the initialization rows that
appear at the start of recordings.

Noise detection:
    The very first data row is an initialization artifact if timestamp == 0,
    distance == 0.0, and speed == 0.0.  Only that single row is removed.
    Subsequent zero rows are real data (mouse not moving) and are kept.

Behaviour:
    - Operates in-place.
    - Use --dry-run to preview what would be removed without writing anything.

Usage examples:
    python batch_clean_treadmill.py                        # folder dialog
    python batch_clean_treadmill.py --dir D:\\my_data
    python batch_clean_treadmill.py --dir D:\\my_data --dry-run
"""

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import pandas as pd


# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------

def find_clean_start(df) -> int:
    """
    Return 1 if the first row is an initialization artifact, else 0.

    The first row is noise only when timestamp == 0, distance == 0.0,
    and speed == 0.0 simultaneously.  Subsequent zero rows are valid data.
    """
    if len(df) == 0:
        return 0

    row = df.iloc[0]
    ts   = pd.to_numeric(row.get('timestamp'), errors='coerce')
    dist = pd.to_numeric(row.get('distance'),  errors='coerce')
    spd  = pd.to_numeric(row.get('speed'),     errors='coerce')

    if ts == 0 and dist == 0.0 and spd == 0.0:
        return 1

    return 0


# ---------------------------------------------------------------------------
# Single-file cleaning
# ---------------------------------------------------------------------------

def clean_file(
    filepath: Path,
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

    if 'timestamp' not in df.columns:
        print(f"  [SKIP]  Missing 'timestamp' column: {filepath.name}")
        return None

    clean_start = find_clean_start(df)
    rows_removed = clean_start
    total_rows = len(df)

    if rows_removed == 0:
        print(f"  [OK]    No init row detected — {total_rows} rows kept:  {filepath.name}")
        return (0, total_rows)

    kept = total_rows - rows_removed
    first_clean_ts = pd.to_numeric(df['timestamp'], errors='coerce').iloc[clean_start]

    if dry_run:
        print(
            f"  [DRY]   Would remove {rows_removed} row(s) "
            f"(keep {kept}/{total_rows}, clean start at timestamp={int(first_clean_ts)}):  "
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
        f"  [DONE]  Removed {rows_removed} row(s) "
        f"(kept {kept}/{total_rows}, clean start at timestamp={int(first_clean_ts)}):  "
        f"{filepath.name}"
    )
    return (rows_removed, total_rows)


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def find_treadmill_files(directory: str) -> list[Path]:
    """Recursively find all files whose name ends with 'treadmill.csv'."""
    return sorted(Path(directory).rglob('*treadmill.csv'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Batch-remove initialization rows from treadmill.csv files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dir', type=str, default=None,
        help='Root directory to search (opens a folder dialog if omitted).',
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
            title='Select root directory containing treadmill.csv files'
        )
        root.destroy()
        if not search_dir:
            print("No directory selected. Exiting.")
            sys.exit(0)

    if not os.path.isdir(search_dir):
        print(f"Directory not found: {search_dir}")
        sys.exit(1)

    print(f"\nSearching for *treadmill.csv under: {search_dir}")
    files = find_treadmill_files(search_dir)

    if not files:
        print("No *treadmill.csv files found.")
        sys.exit(0)

    print(f"Found {len(files)} file(s).")
    if args.dry_run:
        print("DRY RUN — no files will be modified.")
    print()

    total_files   = len(files)
    cleaned_count = 0
    skipped_count = 0
    error_count   = 0
    total_removed = 0

    for filepath in files:
        result = clean_file(filepath, dry_run=args.dry_run)
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
