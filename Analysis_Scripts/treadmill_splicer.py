"""
treadmill_splicer.py

Splices two treadmill CSV files together. The shorter file is appended to the
end of the longer file with global_time, timestamp, and distance adjusted to
continue seamlessly from where the longer file ends.

Output is saved to the current working directory as 'spliced_treadmill.csv'.
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def splice_treadmill_files(file_a: str, file_b: str, output_path: str) -> None:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    print(f"File 1 ({len(df_a)} rows): {os.path.basename(file_a)}")
    print(f"File 2 ({len(df_b)} rows): {os.path.basename(file_b)}")

    base_choice = messagebox.askquestion(
        "Choose Base File",
        f"Which file should be the BASE (the one appended TO)?\n\n"
        f"Yes = File 1: {os.path.basename(file_a)}\n"
        f"No  = File 2: {os.path.basename(file_b)}"
    )

    if base_choice == "yes":
        longer, shorter = df_a.copy(), df_b.copy()
        longer_name, shorter_name = file_a, file_b
    else:
        longer, shorter = df_b.copy(), df_a.copy()
        longer_name, shorter_name = file_b, file_a

    print(f"Base file     ({len(longer)} rows): {os.path.basename(longer_name)}")
    print(f"Appended file ({len(shorter)} rows): {os.path.basename(shorter_name)}")

    # Offsets: last values of the longer file
    global_time_offset = longer["global_time"].iloc[-1]
    timestamp_offset = longer["timestamp"].iloc[-1]
    distance_offset = longer["distance"].iloc[-1]

    # Drop the initialization row (timestamp=0) from the shorter file if present —
    # it has distance=0 which causes a false jump when normalizing
    if shorter["timestamp"].iloc[0] == 0:
        shorter = shorter.iloc[1:].reset_index(drop=True)

    shorter["global_time"] = shorter["global_time"] + global_time_offset
    shorter["timestamp"] = shorter["timestamp"] + timestamp_offset
    shorter["distance"] = shorter["distance"] + distance_offset

    # Concatenate and reset index
    spliced = pd.concat([longer, shorter], ignore_index=True)

    spliced["global_time"] = spliced["global_time"].round(2)

    spliced.to_csv(output_path, index=False, float_format="%.2f")
    print(f"Spliced file saved to: {output_path}")
    print(f"Total rows: {len(spliced)}")
    print(f"Global time range: {spliced['global_time'].iloc[0]:.2f}s "
          f"-> {spliced['global_time'].iloc[-1]:.2f}s")


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    messagebox.showinfo("Treadmill Splicer", "Select the first treadmill CSV file.")
    file1 = filedialog.askopenfilename(
        title="Select first treadmill CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file1:
        print("No file selected. Exiting.")
        return

    messagebox.showinfo("Treadmill Splicer", "Select the second treadmill CSV file.")
    file2 = filedialog.askopenfilename(
        title="Select second treadmill CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file2:
        print("No file selected. Exiting.")
        return

    output = os.path.join(os.getcwd(), "spliced_treadmill.csv")

    splice_treadmill_files(file1, file2, output)
    messagebox.showinfo("Done", f"Spliced file saved to:\n{output}")


if __name__ == "__main__":
    main()
