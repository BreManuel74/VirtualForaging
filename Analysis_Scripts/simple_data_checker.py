from unicodedata import name

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'  # keep text as real font glyphs in SVG
import ast
from scipy.signal import butter, filtfilt

# ── File paths — update as needed ─────────────────────────────────────────────
treadmill_path = r"D:\CAH_motivationSC_cohort\CAH7\Session_4\beh\1769801410treadmill.csv"

# Three capacitive files to compare (update paths)
capacitive_paths = [
    r"D:\CAH_motivationSC_cohort\CAH4\Session_5\beh\1770061057capacitive.csv",
    r"D:\CAH_motivationSC_cohort\CAH11\Session_5\beh\1770065258capacitive.csv",  # replace with file 2
    r"D:\CAH_motivationSC_cohort\CAH2\Session_5\beh\1770057698capacitive.csv",  # replace with file 3
]
capacitive_labels = ['File 1', 'File 2', 'File 3']

# Read the CSV files into pandas DataFrames
treadmill_df = pd.read_csv(treadmill_path, comment='/', engine='python')

# ── Filters ───────────────────────────────────────────────────────────────────

# Butterworth low-pass filter applied to raw speed
BUTTER_CUTOFF_HZ = 0.25
BUTTER_ORDER = 3
fs = 1.0 / treadmill_df['global_time'].diff().median()  # estimate sampling frequency from data
b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ / (fs / 2.0), btype='low')
speed_butter_filtered = filtfilt(b, a, treadmill_df['speed'])

# ── Plot: Raw vs Butterworth filtered treadmill speed ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Treadmill Speed — Raw vs Butterworth Filtered', fontsize=13)

# Raw speed
axes[0].plot(treadmill_df['global_time'], treadmill_df['speed'],
             color='purple', linewidth=0.8, label='Raw')
axes[0].set_ylabel('Speed (raw)')
axes[0].set_title('Raw')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_xlim(0, treadmill_df['global_time'].max())

# Butterworth low-pass filtered speed
axes[1].plot(treadmill_df['global_time'], speed_butter_filtered,
             color='seagreen', linewidth=0.8,
             label=f'Butterworth LP (cutoff={BUTTER_CUTOFF_HZ} Hz, order={BUTTER_ORDER})')
axes[1].set_ylabel('Speed (Butterworth filtered)')
axes[1].set_xlabel('Elapsed Time (s)')
axes[1].set_title(f'Butterworth Low-Pass Filter ({BUTTER_CUTOFF_HZ} Hz, order = {BUTTER_ORDER})')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig('treadmill_speed_raw_vs_butter.svg', format='svg')
plt.show()

# ── Locomotion bout detection ─────────────────────────────────────────────────

BOUT_THRESHOLD_CM_S  = 2.0   # cm/s — minimum speed to count as moving
MIN_BOUT_DURATION_S  = 2.0   # s    — minimum continuous time above threshold
MAX_INTER_BOUT_GAP_S = 2.0   # s    — gaps <= this between bouts are merged

# Work in cm/s (raw speed is in mm/s)
speed_filt_cm = speed_butter_filtered / 10.0
time_arr = treadmill_df['global_time'].values


def detect_locomotion_bouts(time, speed_cm_s,
                             threshold=BOUT_THRESHOLD_CM_S,
                             min_duration=MIN_BOUT_DURATION_S,
                             max_inter_bout_gap=MAX_INTER_BOUT_GAP_S):
    """
    1. Build contiguous above/below-threshold runs.
    2. Bridge short below-threshold gaps (<= max_inter_bout_gap).
    3. Keep merged above-threshold spans >= min_duration.
    """
    above = speed_cm_s >= threshold
    n = len(above)

    runs = []
    start = 0
    for i in range(1, n):
        if above[i] != above[start]:
            runs.append((bool(above[start]), start, i - 1))
            start = i
    runs.append((bool(above[start]), start, n - 1))

    merged = []
    i = 0
    while i < len(runs):
        if runs[i][0]:
            j = i
            while j + 2 < len(runs):
                gap_dur = time[runs[j + 1][2]] - time[runs[j + 1][1]]
                if (not runs[j + 1][0]) and gap_dur <= max_inter_bout_gap:
                    j += 2
                else:
                    break
            merged.append((True, runs[i][1], runs[j][2]))
            i = j + 1
        else:
            merged.append(runs[i])
            i += 1

    bouts = []
    for is_above, s, e in merged:
        if is_above and (time[e] - time[s]) >= min_duration:
            bouts.append((time[s], time[e]))
    return bouts


bouts = detect_locomotion_bouts(time_arr, speed_filt_cm)
print(f"Detected {len(bouts)} locomotion bouts")

# ── Plot: filtered speed with bout highlights ─────────────────────────────────
fig_bouts, ax_bout = plt.subplots(figsize=(12, 4))
ax_bout.plot(time_arr, speed_filt_cm, color='seagreen', linewidth=0.8,
             label='Butterworth filtered')
ax_bout.axhline(BOUT_THRESHOLD_CM_S, color='gray', linestyle='--', linewidth=0.8,
                label=f'Threshold ({BOUT_THRESHOLD_CM_S} cm/s)')
for i, (t_start, t_end) in enumerate(bouts):
    ax_bout.axvspan(t_start, t_end, alpha=0.25, color='gold',
                    label='Locomotion bout' if i == 0 else None)
ax_bout.set_xlabel('Elapsed Time (s)')
ax_bout.set_ylabel('Speed (cm/s)')
ax_bout.set_title(
    f'Locomotion Bout Detection  |  threshold = {BOUT_THRESHOLD_CM_S} cm/s, '
    f'min duration = {MIN_BOUT_DURATION_S}s, IBI \u2264 {MAX_INTER_BOUT_GAP_S}s  |  '
    f'{len(bouts)} bouts detected'
)
ax_bout.set_xlim(0, time_arr[-1])
ax_bout.set_ylim(bottom=0)
ax_bout.spines['top'].set_visible(False)
ax_bout.spines['right'].set_visible(False)
ax_bout.legend()
fig_bouts.tight_layout()
fig_bouts.savefig('locomotion_bouts.svg', format='svg')
plt.show()

# ── Capacitive signal: z-score normalization comparison ──────────────────────
# Load each file, compute z-score, and overlay on shared axes.
# Z-score: subtract session mean, divide by session std.  Accounts for both
# baseline offset and gain differences between sessions/animals.

cap_colors = ['royalblue', 'tomato', 'goldenrod']

fig_cap, axes_cap = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
fig_cap.suptitle('Capacitive Signal — Z-Score Normalized', fontsize=13)

fig_cap_overlay, ax_overlay = plt.subplots(figsize=(14, 4))
ax_overlay.set_title('Capacitive Signal — Z-Score Normalized (Overlay)', fontsize=13)
ax_overlay.set_xlabel('Elapsed Time (s)')
ax_overlay.set_ylabel('Z-score')
ax_overlay.spines['top'].set_visible(False)
ax_overlay.spines['right'].set_visible(False)

for i, (path, label, color) in enumerate(
        zip(capacitive_paths, capacitive_labels, cap_colors)):
    df = pd.read_csv(path, comment='/', engine='python')
    time_col = 'elapsed_time' if 'elapsed_time' in df.columns else df.columns[0]
    val_col  = 'capacitive_value' if 'capacitive_value' in df.columns else df.columns[1]

    t   = pd.to_numeric(df[time_col],  errors='coerce').values
    raw = pd.to_numeric(df[val_col],   errors='coerce').values

    # Drop NaN rows
    valid   = ~(np.isnan(t) | np.isnan(raw))
    t, raw  = t[valid], raw[valid]

    # Z-score: (x - mean) / std
    mu, sigma = raw.mean(), raw.std(ddof=1)
    z = (raw - mu) / sigma if sigma > 0 else np.zeros_like(raw)

    print(f"{label}: mean={mu:.4f}, std={sigma:.4f}, n={len(raw)} samples")

    # Individual subplot
    ax = axes_cap[i]
    ax.plot(t, z, color=color, linewidth=0.7, label=label)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.6)
    ax.set_ylabel('Z-score')
    ax.set_title(f'{label}  (raw mean={mu:.2f}, std={sigma:.2f})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, loc='upper right')
    if i == 2:
        ax.set_xlabel('Elapsed Time (s)')

    # Overlay
    ax_overlay.plot(t, z, color=color, linewidth=0.7, alpha=0.8, label=label)

ax_overlay.axhline(0, color='gray', linestyle='--', linewidth=0.6)
ax_overlay.legend()
fig_cap.tight_layout()
fig_cap_overlay.tight_layout()
fig_cap.savefig('capacitive_zscore_individual.svg', format='svg')
fig_cap_overlay.savefig('capacitive_zscore_overlay.svg', format='svg')
plt.show()
