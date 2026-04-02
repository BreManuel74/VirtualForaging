"""Level Sorter
Author: Brenna Manuel

Loads one or more *_data.csv files via a tkinter dialog, then matches each
animal's Progress_Reports/*_log.csv by animal ID and attaches it to the
DataFrame for downstream analysis.

Usage:
    python level_sorter.py          # runs main()
    or import and call load_data()  # programmatic use
"""

import os
import re
import tkinter as tk
from tkinter import filedialog

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
MOUSEPORTAL_DIR  = os.path.dirname(SCRIPT_DIR)
PROGRESS_DIR     = os.path.join(MOUSEPORTAL_DIR, 'Progress_Reports')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _animal_id_from_path(filepath: str) -> str:
    """Extract animal ID (e.g. 'CAH1', 'VF11') from a *_data.csv path."""
    basename = os.path.basename(filepath)               # e.g. CAH1_data.csv
    return re.split(r'[_\.]', basename)[0]              # e.g. CAH1


def load_data_files(file_paths: list[str]) -> dict[str, pd.DataFrame]:
    """
    Read a list of *_data.csv paths into a dict keyed by animal ID.

    Each DataFrame has columns:
        timestamp, date, level, treadmill, trial_log, capacitive

    Returns
    -------
    dict[str, pd.DataFrame]
        { 'CAH1': df, 'CAH2': df, ... }
    """
    data = {}
    for path in file_paths:
        animal_id = _animal_id_from_path(path)
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        data[animal_id] = df
        print(f"  Loaded {animal_id}: {len(df)} sessions from {os.path.basename(path)}")
    return data


def load_log_files(animal_ids: list[str]) -> dict[str, pd.DataFrame | None]:
    """
    For each animal ID find its *_log.csv inside Progress_Reports and read it.

    Each log DataFrame has columns:
        Date, Time, Level, Batch ID

    Returns
    -------
    dict[str, pd.DataFrame | None]
        None when no matching log file is found.
    """
    logs = {}
    for animal_id in animal_ids:
        log_path = os.path.join(PROGRESS_DIR, f'{animal_id}_log.csv')
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            logs[animal_id] = df
            print(f"  Matched log  {animal_id}: {len(df)} entries")
        else:
            logs[animal_id] = None
            print(f"  [WARN] No log found for {animal_id} (expected {log_path})")
    return logs


def build_combined(data: dict[str, pd.DataFrame],
                   logs: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """
    Return a single 'long' DataFrame with every session row from every animal,
    augmented with a 'log' column that holds the per-session subset of the
    matching progress log (as a nested DataFrame stored in an object column).

    Columns added to each session row:
        animal_id   – e.g. 'CAH1'
        session_num – 0-based session index within that animal
        log_entries – subset of the log DataFrame for that session's date
                      (None if no log was found for this animal)
    """
    rows = []
    for animal_id, df in data.items():
        log_df = logs.get(animal_id)
        for session_idx, (_, row) in enumerate(df.iterrows()):
            session_date = row['date'].date() if pd.notna(row.get('date', None)) else None

            # Grab all log entries whose Date matches this session date
            if log_df is not None and session_date is not None:
                mask = log_df['Date'].dt.date == session_date
                log_entries = log_df[mask].reset_index(drop=True)
                log_entries = log_entries if len(log_entries) > 0 else None
            else:
                log_entries = None

            row_dict = row.to_dict()
            row_dict['animal_id']   = animal_id
            row_dict['session_num'] = session_idx
            row_dict['log_entries'] = log_entries
            rows.append(row_dict)

    combined = pd.DataFrame(rows).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame | None], pd.DataFrame]:
    """
    Open tkinter dialogs, load data + logs, print a summary, and return:
        data     – { animal_id: sessions_df }
        logs     – { animal_id: log_df | None }
        combined – long DataFrame with one row per session across all animals
    """
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title='Select *_data.csv files',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        initialdir=MOUSEPORTAL_DIR,
    )

    if not file_paths:
        print('No files selected. Exiting.')
        return {}, {}, pd.DataFrame()

    print(f'\nLoading {len(file_paths)} data file(s)...')
    data = load_data_files(list(file_paths))

    print(f'\nMatching Progress_Reports logs...')
    logs = load_log_files(list(data.keys()))

    combined = build_combined(data, logs)

    n_matched = sum(1 for v in logs.values() if v is not None)
    print(f'\nSummary: {len(data)} animals | {n_matched}/{len(logs)} logs matched '
          f'| {len(combined)} total sessions')

    return data, logs, combined


def main():
    data, logs, combined = load_data()
    if combined.empty:
        return

    # Print a compact view of the combined table (without the nested log_entries column)
    display_cols = [c for c in combined.columns if c != 'log_entries']
    print('\nCombined session table (first 20 rows):')
    print(combined[display_cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
