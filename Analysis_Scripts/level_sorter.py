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

            # Extract session end rewards from the Session End log entry
            session_end_rewards = None
            if log_entries is not None:
                for _, le_row in log_entries.iterrows():
                    level_str = str(le_row.get('Level', ''))
                    if '(Session End' in level_str:
                        m = re.search(r'Session End\s*-\s*(\d+)\s*reward',
                                      level_str, re.IGNORECASE)
                        if m:
                            session_end_rewards = int(m.group(1))
                        break

            row_dict = row.to_dict()
            row_dict['animal_id']           = animal_id
            row_dict['session_num']         = session_idx
            row_dict['log_entries']         = log_entries
            row_dict['session_end_rewards'] = session_end_rewards
            row_dict['reward_threshold']    = get_reward_threshold(
                row.get('level'), animal_id
            )
            rows.append(row_dict)

    combined = pd.DataFrame(rows).reset_index(drop=True)

    # start_rewards = prior session's end rewards (same animal); session 0 gets 0
    combined['start_rewards'] = (
        combined.groupby('animal_id', group_keys=False)['session_end_rewards']
        .apply(lambda s: s.shift(1))
    )
    first_session_mask = combined.groupby('animal_id').cumcount() == 0
    combined.loc[first_session_mask, 'start_rewards'] = 0

    # Compute per-session level transition timestamps (reads trial_log files)
    combined['level_transitions'] = combined.apply(extract_level_transitions, axis=1)

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
                       'reward_threshold', 'start_rewards', 'session_end_rewards',
                       'timestamp']

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


def extract_level_transitions(session_row: pd.Series) -> list[dict]:
    """
    For one session row from `combined`, find the reward_event timestamp at
    which each level threshold was hit (i.e., when the mouse advanced levels).

    Reward counting rules:
      - The first level of the session carries in `start_rewards` from the prior
        session; only (threshold - carry_in) additional rewards are needed.
      - Every subsequent level within the same session starts fresh from 0.

    Returns a list of dicts, one per level visited this session:
        level          – e.g. 'level_3.json'
        carry_in       – rewards inherited (> 0 only for the first level)
        threshold      – rewards required to advance (None if not in matcher)
        needed         – rewards still required = max(0, threshold - carry_in)
        transition_ts  – reward_event timestamp (s) when threshold was hit,
                         or None if session ended before reaching it
        completed      – True if threshold was hit, False if not, None if unknown
    """
    results = []
    trial_log_path      = session_row.get('trial_log')
    log_entries         = session_row.get('log_entries')
    animal_id           = str(session_row.get('animal_id', ''))
    session_end_rewards = session_row.get('session_end_rewards')

    raw_sr = session_row.get('start_rewards', 0)
    if raw_sr is None or (isinstance(raw_sr, float) and pd.isna(raw_sr)):
        start_rewards = 0
    else:
        start_rewards = int(raw_sr)

    if not isinstance(trial_log_path, str) or not trial_log_path:
        return results
    if log_entries is None or not isinstance(log_entries, pd.DataFrame) or log_entries.empty:
        return results

    try:
        tl = pd.read_csv(trial_log_path)
    except Exception as exc:
        print(f"  [WARN] Cannot read trial_log ({animal_id}): {exc}")
        return results

    if 'reward_event' not in tl.columns:
        return results

    reward_ts = tl['reward_event'].dropna().tolist()

    # Build ordered clean level sequence (exclude Session End marker rows)
    entries = log_entries.copy()
    if 'Time' in entries.columns:
        entries = entries.sort_values('Time')
    level_sequence = [
        str(row['Level']).split('(')[0].strip()
        for _, row in entries.iterrows()
        if '(Session End' not in str(row.get('Level', ''))
    ]

    n          = len(level_sequence)
    reward_idx = 0  # running position in reward_ts

    for i, level in enumerate(level_sequence):
        is_last   = (i == n - 1)
        threshold = get_reward_threshold(level, animal_id)
        carry_in  = start_rewards if i == 0 else 0

        if threshold is None:
            results.append(dict(level=level, carry_in=carry_in, threshold=None,
                                needed=None, transition_ts=None, completed=None))
            continue

        needed  = max(0, threshold - carry_in)
        end_idx = reward_idx + needed - 1  # index of the threshold-hitting reward

        if is_last:
            se = int(session_end_rewards) if session_end_rewards is not None else 0
            if needed > 0 and se >= needed and end_idx < len(reward_ts):
                transition_ts = reward_ts[end_idx]
                completed     = True
            elif needed == 0:
                transition_ts = None
                completed     = True
            else:
                transition_ts = None
                completed     = False
        else:
            if needed > 0 and end_idx < len(reward_ts):
                transition_ts = reward_ts[end_idx]
                completed     = True
            elif needed == 0:
                transition_ts = None
                completed     = True
            else:
                transition_ts = None
                completed     = False

        reward_idx += needed
        results.append(dict(level=level, carry_in=carry_in, threshold=threshold,
                            needed=needed, transition_ts=transition_ts,
                            completed=completed))

    return results


def write_transitions_report(transitions_df: pd.DataFrame, filepath: str) -> None:
    """
    Write a human-readable text file listing every level transition time for
    every mouse, grouped by animal → session → level.

    Format:
        === CAH1 ===
          Session 1  |  2026-01-26  |  start level: level_1.json  |  carry-in: 0
            level_1.json   threshold: 10  needed: 10   transition: 299.72 s   [completed]
            level_2.json   threshold: 15  needed: 15   transition: 1212.58 s  [completed]
            level_3.json   threshold: 15  needed: 15   transition: --          [not completed]
          ...
    """
    from datetime import datetime as _dt

    lines = [
        'Level Transition Times',
        f'Generated: {_dt.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
    ]

    for animal_id, grp in transitions_df.groupby('animal_id', sort=True):
        lines.append(f'=== {animal_id} ===')
        for _, sess_grp in grp.groupby('session_num', sort=True):
            first       = sess_grp.iloc[0]
            sess_num    = int(first['session_num']) + 1
            date_str    = str(first['date'])[:10]
            start_level = first.get('session_start_level', '?')
            carry_in    = int(first.get('carry_in', 0)) if pd.notna(first.get('carry_in', 0)) else 0
            lines.append(
                f'  Session {sess_num}  |  {date_str}  |  '
                f'start level: {start_level}  |  carry-in: {carry_in}'
            )
            for _, row in sess_grp.iterrows():
                level     = row.get('level', '?')
                threshold = row.get('threshold')
                needed    = row.get('needed')
                ts        = row.get('transition_ts')
                completed = row.get('completed')

                ts_str   = f'{ts:.2f} s' if pd.notna(ts) and ts is not None else '--'
                thr_str  = str(int(threshold)) if threshold is not None else '?'
                need_str = str(int(needed)) if needed is not None else '?'

                if completed is True:
                    status = '[completed]'
                elif completed is False:
                    status = '[not completed]'
                else:
                    status = '[unknown]'

                lines.append(
                    f'    {level:<20}  threshold: {thr_str:<4}  '
                    f'needed: {need_str:<4}  transition: {ts_str:<14}  {status}'
                )
        lines.append('')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nTransitions report saved → {filepath}')


def build_transitions_df(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the `level_transitions` column of `combined` into a one-row-per-
    level-visit DataFrame with session context repeated on every row.

    Columns:
        animal_id, session_num, date, session_start_level,
        start_rewards, session_end_rewards, timestamp,
        level, carry_in, threshold, needed, transition_ts, completed
    """
    SESSION_CONTEXT = ['animal_id', 'session_num', 'date', 'timestamp',
                       'start_rewards', 'session_end_rewards']
    rows = []
    for _, sess in combined.iterrows():
        transitions = sess.get('level_transitions')
        if not transitions:
            continue
        ctx = {col: sess[col] for col in SESSION_CONTEXT if col in sess.index}
        ctx['session_start_level'] = sess.get('level')
        for t in transitions:
            rows.append({**ctx, **t})

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_data() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame | None],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Open tkinter dialogs, load data + logs, print a summary, and return:
        data           – { animal_id: sessions_df }
        logs           – { animal_id: log_df | None }
        combined       – one row per session; log_entries / level_transitions columns
        flat_log       – one row per log entry, with session context joined in
        transitions_df – one row per level visit, with transition timestamps
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
        return {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f'\nLoading {len(file_paths)} data file(s)...')
    data = load_data_files(list(file_paths))

    print(f'\nMatching Progress_Reports logs...')
    logs = load_log_files(list(data.keys()))

    combined       = build_combined(data, logs)
    flat_log       = build_flat_log(combined)
    transitions_df = build_transitions_df(combined)

    n_matched = sum(1 for v in logs.values() if v is not None)
    print(f'\nSummary: {len(data)} animals | {n_matched}/{len(logs)} logs matched '
          f'| {len(combined)} sessions | {len(flat_log)} log entries '
          f'| {len(transitions_df)} level visits')

    return data, logs, combined, flat_log, transitions_df


def main():
    data, logs, combined, flat_log, transitions_df = load_data()
    if combined.empty:
        return

    # Print a compact view of the combined table (without nested object columns)
    display_cols = [c for c in combined.columns
                    if c not in ('log_entries', 'level_transitions')]
    print('\nCombined session table (first 20 rows):')
    print(combined[display_cols].head(20).to_string(index=False))

    # Print a preview of the flat log table
    print('\nFlat log table (first 20 rows):')
    flat_preview_cols = ['animal_id', 'session_num', 'date', 'level',
                         'reward_threshold', 'Date', 'Time',
                         'log_level', 'is_session_end', 'end_rewards']
    flat_preview_cols = [c for c in flat_preview_cols if c in flat_log.columns]
    print(flat_log[flat_preview_cols].head(20).to_string(index=False))

    # Print a preview of level transition timestamps
    print('\nLevel transitions table (first 20 rows):')
    trans_cols = ['animal_id', 'session_num', 'date', 'session_start_level',
                  'level', 'carry_in', 'threshold', 'needed',
                  'transition_ts', 'completed']
    trans_cols = [c for c in trans_cols if c in transitions_df.columns]
    print(transitions_df[trans_cols].head(20).to_string(index=False))

    # Save full transitions report as a text file
    if not transitions_df.empty:
        from datetime import datetime as _dt
        ts_stamp   = _dt.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(MOUSEPORTAL_DIR, f'level_transitions_{ts_stamp}.txt')
        write_transitions_report(transitions_df, report_path)


if __name__ == '__main__':
    main()
