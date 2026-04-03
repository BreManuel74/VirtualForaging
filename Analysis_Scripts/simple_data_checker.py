import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.signal import savgol_filter, butter, filtfilt

# File paths (update these if your files are in a different location)
treadmill_path = r"D:\CAH_motivationSC_cohort\CAH1\all_csv_files\1771003869treadmill.csv"

# Read the CSV files into pandas DataFrames
treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')

# ── Filters ───────────────────────────────────────────────────────────────────

# Moving median filter (window = 5 samples)
MEDIAN_WINDOW = 5
speed_median_filtered = (
    treadmill_df['speed']
    .rolling(window=MEDIAN_WINDOW, center=True, min_periods=1)
    .median()
)

# Savitzky-Golay filter applied to median-filtered data
SG_WINDOW = 9
SG_POLYORDER = 2
speed_savgol_filtered = savgol_filter(speed_median_filtered, window_length=SG_WINDOW, polyorder=SG_POLYORDER)

# Butterworth low-pass filter applied to median-filtered data
BUTTER_CUTOFF_HZ = 5.0
BUTTER_ORDER = 3
fs = 1.0 / treadmill_df['global_time'].diff().median()  # estimate sampling frequency from data
b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ / (fs / 2.0), btype='low')
speed_butter_filtered = filtfilt(b, a, speed_median_filtered)

# ── Plot: Raw vs filtered treadmill speed ─────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
fig.suptitle('Treadmill Speed — Filter Comparison', fontsize=13)

# Raw speed
axes[0].plot(treadmill_df['global_time'], treadmill_df['speed'],
             color='purple', linewidth=0.8, label='Raw')
axes[0].set_ylabel('Speed (raw)')
axes[0].set_title('Raw')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_xlim(0, treadmill_df['global_time'].max())

# Moving median filtered speed
axes[1].plot(treadmill_df['global_time'], speed_median_filtered,
             color='darkorange', linewidth=0.8, label=f'Median (window={MEDIAN_WINDOW})')
axes[1].set_ylabel('Speed (median filtered)')
axes[1].set_xlabel('Elapsed Time (s)')
axes[1].set_title(f'Moving Median Filter (window = {MEDIAN_WINDOW} samples)')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Savitzky-Golay filtered speed (applied to median-filtered)
axes[2].plot(treadmill_df['global_time'], speed_savgol_filtered,
             color='steelblue', linewidth=0.8,
             label=f'Savitzky-Golay (window={SG_WINDOW}, poly={SG_POLYORDER})')
axes[2].set_ylabel('Speed (SG filtered)')
axes[2].set_xlabel('Elapsed Time (s)')
axes[2].set_title(f'Savitzky-Golay Filter (window = {SG_WINDOW}, polynomial order = {SG_POLYORDER}) — applied to median-filtered')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

# Butterworth low-pass filtered speed (applied to median-filtered)
axes[3].plot(treadmill_df['global_time'], speed_butter_filtered,
             color='seagreen', linewidth=0.8,
             label=f'Butterworth LP (cutoff={BUTTER_CUTOFF_HZ} Hz, order={BUTTER_ORDER})')
axes[3].set_ylabel('Speed (Butterworth filtered)')
axes[3].set_xlabel('Elapsed Time (s)')
axes[3].set_title(f'Butterworth Low-Pass Filter ({BUTTER_CUTOFF_HZ} Hz, order = {BUTTER_ORDER}) — applied to median-filtered')
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
plt.show()