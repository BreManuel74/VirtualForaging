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
# CAH cohort level → reward threshold mapping
# level_1.json : 10 rewards required to advance
# level_2.json … level_50.json : 15 rewards required to advance
# ---------------------------------------------------------------------------
CAH_LEVEL_THRESHOLD_MATCHER: dict[str, int] = {
    'level_1.json':  10,
    'level_2.json':  15,
    'level_3.json':  15,
    'level_4.json':  15,
    'level_5.json':  15,
    'level_6.json':  15,
    'level_7.json':  15,
    'level_8.json':  15,
    'level_9.json':  15,
    'level_10.json': 15,
    'level_11.json': 15,
    'level_12.json': 15,
    'level_13.json': 15,
    'level_14.json': 15,
    'level_15.json': 15,
    'level_16.json': 15,
    'level_17.json': 15,
    'level_18.json': 15,
    'level_19.json': 15,
    'level_20.json': 15,
    'level_21.json': 15,
    'level_22.json': 15,
    'level_23.json': 15,
    'level_24.json': 15,
    'level_25.json': 15,
    'level_26.json': 15,
    'level_27.json': 15,
    'level_28.json': 15,
    'level_29.json': 15,
    'level_30.json': 15,
    'level_31.json': 15,
    'level_32.json': 15,
    'level_33.json': 15,
    'level_34.json': 15,
    'level_35.json': 15,
    'level_36.json': 15,
    'level_37.json': 15,
    'level_38.json': 15,
    'level_39.json': 15,
    'level_40.json': 15,
    'level_41.json': 15,
    'level_42.json': 15,
    'level_43.json': 15,
    'level_44.json': 15,
    'level_45.json': 15,
    'level_46.json': 15,
    'level_47.json': 15,
    'level_48.json': 15,
    'level_49.json': 15,
    'level_50.json': 15,
}


def get_reward_threshold(level_str: str, animal_id: str) -> int | None:
    """
    Return the reward threshold for a given level and animal.

    Currently only the CAH cohort has a defined threshold map.  Returns None
    for any other cohort, or when the level string is not recognised.

    Parameters
    ----------
    level_str  : e.g. 'level_6.json'
    animal_id  : e.g. 'CAH1' or 'VF11'
    """
    if isinstance(animal_id, str) and animal_id.upper().startswith('CAH'):
        return CAH_LEVEL_THRESHOLD_MATCHER.get(level_str)
    return None


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
            row_dict['reward_threshold'] = get_reward_threshold(
                row.get('level'), animal_id
            )
            rows.append(row_dict)

    combined = pd.DataFrame(rows).reset_index(drop=True)
    return combined


def build_flat_log(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the nested log_entries in `combined` into a flat DataFrame with one
    row per log entry.  Session context columns are repeated on every row so
    that both the session and its individual log events can be filtered and
    grouped in a single table.

    Session context columns carried over from `combined`:
        animal_id, session_num, date (session date), level (session start level),
        reward_threshold, timestamp

    Log columns from *_log.csv:
        Date, Time, Level, Batch ID

    Derived columns added here:
        is_session_end  – True when the Level entry contains '(Session End'
        end_rewards     – integer reward count parsed from Session End entries,
                          NaN for non-session-end rows
        log_level       – clean level name (e.g. 'level_6.json') stripped of
                          any '(Session End …)' suffix
    """
    SESSION_CONTEXT = ['animal_id', 'session_num', 'date', 'level',
                       'reward_threshold', 'timestamp']

    flat_rows = []
    for _, session_row in combined.iterrows():
        log_df = session_row.get('log_entries')
        if log_df is None or not isinstance(log_df, pd.DataFrame) or log_df.empty:
            continue

        # Build the context dict for this session
        ctx = {col: session_row[col] for col in SESSION_CONTEXT if col in session_row.index}

        for _, log_row in log_df.iterrows():
            level_str = str(log_row.get('Level', ''))
            is_end = '(Session End' in level_str

            # Parse reward count from e.g. "level_6.json (Session End - 8 rewards)"
            end_rewards = float('nan')
            if is_end:
                import re as _re
                m = _re.search(r'Session End\s*-\s*(\d+)\s*reward', level_str, _re.IGNORECASE)
                if m:
                    end_rewards = int(m.group(1))

            # Clean level name (strip suffix)
            log_level = level_str.split('(')[0].strip()

            row = {**ctx, **log_row.to_dict(),
                   'is_session_end': is_end,
                   'end_rewards':    end_rewards,
                   'log_level':      log_level}
            flat_rows.append(row)

    flat_log = pd.DataFrame(flat_rows).reset_index(drop=True)
    return flat_log


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_data() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame | None],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Open tkinter dialogs, load data + logs, print a summary, and return:
        data     – { animal_id: sessions_df }
        logs     – { animal_id: log_df | None }
        combined – one row per session; log_entries column holds matching log rows
        flat_log – one row per log entry, with session context columns joined in
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
    flat_log  = build_flat_log(combined)

    n_matched = sum(1 for v in logs.values() if v is not None)
    print(f'\nSummary: {len(data)} animals | {n_matched}/{len(logs)} logs matched '
          f'| {len(combined)} sessions | {len(flat_log)} log entries')

    return data, logs, combined, flat_log


def main():
    data, logs, combined, flat_log = load_data()
    if combined.empty:
        return

    # Print a compact view of the combined table (without the nested log_entries column)
    display_cols = [c for c in combined.columns if c != 'log_entries']
    print('\nCombined session table (first 20 rows):')
    print(combined[display_cols].head(20).to_string(index=False))

    # Print a preview of the flat log table
    print('\nFlat log table (first 20 rows):')
    flat_preview_cols = ['animal_id', 'session_num', 'date', 'level',
                         'reward_threshold', 'Date', 'Time',
                         'log_level', 'is_session_end', 'end_rewards']
    flat_preview_cols = [c for c in flat_preview_cols if c in flat_log.columns]
    print(flat_log[flat_preview_cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
