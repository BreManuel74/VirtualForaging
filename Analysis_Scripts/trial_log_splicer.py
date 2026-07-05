"""
trial_log_splicer.py

Splices two trial log CSV files together using the ending elapsed_time from
the longer capacitive file as the time offset.

Workflow:
  1. Select the longer capacitive CSV  -> its final elapsed_time is the offset
  2. Select the longer trial log CSV   -> used as the base (not modified)
  3. Select the shorter trial log CSV  -> time columns shifted by the offset,
                                          then appended to the longer log

Time columns adjusted in the shorter trial log:
    texture_change_time, texture_revert, puff_event, reward_event

Output saved to the current working directory as 'spliced_trial_log.csv'.
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

TIME_COLS = ["texture_change_time", "texture_revert", "puff_event", "reward_event", "probe_time"]


def splice_trial_logs(cap_longer: str, log_longer: str, log_shorter: str, output_path: str) -> None:
    # Get offset from the longer capacitive file
    cap_df = pd.read_csv(cap_longer)
    offset = cap_df["elapsed_time"].iloc[-1]
    print(f"Time offset from capacitive file: {offset:.2f}s")

    df_longer = pd.read_csv(log_longer)
    df_shorter = pd.read_csv(log_shorter)

    # Drop fully blank rows from both files
    df_longer = df_longer.dropna(how="all").reset_index(drop=True)
    df_shorter = df_shorter.dropna(how="all").reset_index(drop=True)

    # Split each file into completed trials (have texture_history) and
    # pre-generated future trials (only hallway data, no texture_history yet)
    longer_done    = df_longer[df_longer["texture_history"].notna()].reset_index(drop=True)
    longer_pending = df_longer[df_longer["texture_history"].isna()].reset_index(drop=True)
    shorter_done    = df_shorter[df_shorter["texture_history"].notna()].reset_index(drop=True)
    shorter_pending = df_shorter[df_shorter["texture_history"].isna()].reset_index(drop=True)

    print(f"Longer  — {len(longer_done)} completed, {len(longer_pending)} pre-generated")
    print(f"Shorter — {len(shorter_done)} completed, {len(shorter_pending)} pre-generated")

    # Adjust time columns only in the shorter completed trials
    for col in TIME_COLS:
        if col in shorter_done.columns:
            shorter_done[col] = shorter_done[col].apply(
                lambda x: round(x + offset, 2) if pd.notna(x) else x
            )

    # Stack: completed trials from both files first, then all pre-generated rows
    spliced = pd.concat([longer_done, shorter_done, longer_pending, shorter_pending],
                        ignore_index=True)

    # puff_event and reward_event are independent event lists, not trial-aligned.
    # Compact them so all non-NaN values are packed sequentially from the top.
    for col in ["puff_event", "reward_event"]:
        if col in spliced.columns:
            events = spliced[col].dropna().tolist()
            new_col = events + [float("nan")] * (len(spliced) - len(events))
            spliced[col] = new_col

    spliced.to_csv(output_path, index=False)
    print(f"Spliced trial log saved to: {output_path}")
    print(f"Total trials: {len(spliced)}")


def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Trial Log Splicer - Step 1 of 3",
        "Select the LONGER capacitive CSV.\n\nIts final elapsed_time will be used as the time offset."
    )
    cap_longer = filedialog.askopenfilename(
        title="Longer capacitive CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not cap_longer:
        print("No file selected. Exiting.")
        return

    messagebox.showinfo(
        "Trial Log Splicer - Step 2 of 3",
        "Select the LONGER trial log CSV.\n\nThis is the base file — its rows are kept as-is."
    )
    log_longer = filedialog.askopenfilename(
        title="Longer trial log CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not log_longer:
        print("No file selected. Exiting.")
        return

    messagebox.showinfo(
        "Trial Log Splicer - Step 3 of 3",
        "Select the SHORTER trial log CSV.\n\nIts timestamps will be shifted by the offset and appended."
    )
    log_shorter = filedialog.askopenfilename(
        title="Shorter trial log CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not log_shorter:
        print("No file selected. Exiting.")
        return

    output = os.path.join(os.getcwd(), "spliced_trial_log.csv")

    splice_trial_logs(cap_longer, log_longer, log_shorter, output)
    messagebox.showinfo("Done", f"Spliced trial log saved to:\n{output}")


if __name__ == "__main__":
    main()
