"""
Capacitive Signal Filter Test Script

Problem: The raw capacitive baseline sometimes drifts into the range of real lick
signals, causing the global KDE method to either miss true licks or generate false
positives. This script lets you compare filtering strategies that estimate and remove
the slow-varying baseline drift before lick detection.

Strategy:
    1. Estimate the slow-varying baseline using a Savitzky-Golay filter (large window).
    2. Subtract the SG baseline from the raw signal to get a detrended residual.
    3. Run KDE-based lick detection on the residual.
    4. Compare original (unfiltered) vs filtered detections side-by-side.

Usage:
    python test_capacitive_filter.py [path/to/file.csv]

    Or run without arguments and you'll be prompted for the file path.

Tuning parameters (prompted interactively):
    - SG window length  : number of samples spanning the baseline drift
                          (must be odd; e.g. 501 ≈ ~5 s at 100 Hz)
    - SG poly order     : polynomial order (default 3; higher = more shape preserved)
    - Detection threshold: leave blank for automatic KDE valley detection
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import lick_detection_algorithm as lda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame | None:
    """Load CSV and attach Time_sec column."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return None

    if 'capacitive_value' not in df.columns:
        print(f"✗ 'capacitive_value' column not found. Available: {list(df.columns)}")
        return None

    if 'Time_sec' not in df.columns:
        if 'elapsed_time' in df.columns:
            df['Time_sec'] = df['elapsed_time']
        elif 'arduino_timestamp' in df.columns:
            df['Time_sec'] = df['arduino_timestamp'] / 1000.0
        else:
            print("✗ No time column found (need 'elapsed_time' or 'arduino_timestamp')")
            return None

    print(f"✓ Loaded {len(df)} rows  |  {df['Time_sec'].min():.2f}s – {df['Time_sec'].max():.2f}s")
    return df


def _estimate_sampling_rate(df: pd.DataFrame) -> float:
    """Return median sampling rate in Hz."""
    diffs = df['Time_sec'].diff().dropna()
    median_dt = diffs.median()
    return 1.0 / median_dt if median_dt > 0 else 100.0


def _make_window_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


# Minimum deviation gap for this script's dynamic threshold.
# Using 1.0 (instead of the global default 10.0) to allow the valley search
# to find boundaries closer to the noise peak — useful when the baseline drift
# compresses the noise/signal separation after SG detrending.
_FILTER_MIN_DEVIATION_GAP = 0.5


def _run_pipeline(df: pd.DataFrame, threshold=None):
    """Run the standard KDE lick detection pipeline. Returns (df_norm, events_df, kde_val, threshold)."""
    kde_val = lda.compute_KDE(df, 'capacitive_value')
    df_norm = lda.compute_KDE_normalizations(df, 'capacitive_value', kde_val)
    events_df, thr = lda.detect_events_above_threshold(
        df_norm, 'capacitive_value',
        threshold=threshold,
        min_deviation_gap=_FILTER_MIN_DEVIATION_GAP,
    )
    return df_norm, events_df, kde_val, thr


# ---------------------------------------------------------------------------
# Core filter application
# ---------------------------------------------------------------------------

def apply_sg_baseline(df: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to estimate the slow baseline, then subtract it.

    The detrended signal is:
        detrended = raw - SG_baseline + median(SG_baseline)

    Adding back the median preserves the signal on an absolute scale so that
    the KDE normalization in lick_detection_algorithm still makes sense.

    Returns a copy of df with two extra columns:
        - 'sg_baseline'   : the SG smooth estimate of the drift
        - 'capacitive_value_filtered' : detrended + median (baseline-corrected)
    """
    raw = pd.to_numeric(df['capacitive_value'], errors='coerce').ffill().bfill().values
    window_length = _make_window_odd(window_length)
    # Clamp window to data length
    if window_length >= len(raw):
        window_length = _make_window_odd(len(raw) - 2)
    polyorder = min(polyorder, window_length - 1)

    baseline = savgol_filter(raw, window_length=window_length, polyorder=polyorder)
    median_baseline = float(np.median(baseline))
    detrended = raw - baseline + median_baseline

    out = df.copy()
    out['sg_baseline'] = baseline
    out['capacitive_value_filtered'] = detrended
    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(df_raw: pd.DataFrame,
                    df_filt: pd.DataFrame,
                    events_orig: pd.DataFrame,
                    events_filt: pd.DataFrame,
                    kde_orig: float,
                    kde_filt: float,
                    thr_orig: float,
                    thr_filt: float,
                    window_length: int,
                    polyorder: int,
                    filename: str) -> plt.Figure:
    """6-panel comparison figure."""

    time = df_raw['Time_sec'].values
    raw  = pd.to_numeric(df_raw['capacitive_value'], errors='coerce').values
    baseline = df_filt['sg_baseline'].values
    filtered = df_filt['capacitive_value_filtered'].values

    dev_orig = df_raw.get('capacitive_value_deviation', pd.Series(dtype=float)).values \
               if 'capacitive_value_deviation' in df_raw.columns else np.full(len(time), np.nan)
    dev_filt = df_filt.get('capacitive_value_deviation', pd.Series(dtype=float)).values \
               if 'capacitive_value_deviation' in df_filt.columns else np.full(len(time), np.nan)

    lick_t_orig = events_orig.loc[events_orig['capacitive_value_event'] == 1, 'Time_sec'].values
    lick_t_filt = events_filt.loc[events_filt['capacitive_value_event'] == 1, 'Time_sec'].values

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        f'Capacitive Filter Comparison — {filename}\n'
        f'SG window={window_length} samples, polyorder={polyorder}',
        fontsize=12, y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: raw signal ───────────────────────────────────────────────────
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.plot(time, raw, color='steelblue', linewidth=0.6, alpha=0.8, label='Raw')
    ax_raw.axhline(kde_orig, color='darkorange', linestyle='--', linewidth=1.2,
                   label=f'KDE baseline: {kde_orig:.1f}')
    ax_raw.set_title('Original raw signal + KDE baseline', fontsize=10)
    ax_raw.set_ylabel('Capacitive value', fontsize=9)
    ax_raw.legend(fontsize=8)
    ax_raw.spines['top'].set_visible(False); ax_raw.spines['right'].set_visible(False)

    ax_sg = fig.add_subplot(gs[0, 1])
    ax_sg.plot(time, raw, color='steelblue', linewidth=0.6, alpha=0.5, label='Raw')
    ax_sg.plot(time, baseline, color='crimson', linewidth=1.5, label='SG baseline')
    ax_sg.axhline(kde_filt, color='darkorange', linestyle='--', linewidth=1.2,
                  label=f'KDE filtered: {kde_filt:.1f}')
    ax_sg.set_title('Raw signal + SG estimated baseline', fontsize=10)
    ax_sg.set_ylabel('Capacitive value', fontsize=9)
    ax_sg.legend(fontsize=8)
    ax_sg.spines['top'].set_visible(False); ax_sg.spines['right'].set_visible(False)

    # ── Row 1: deviation signals ─────────────────────────────────────────────
    ax_dev_orig = fig.add_subplot(gs[1, 0])
    ax_dev_orig.plot(time, dev_orig, color='steelblue', linewidth=0.6, alpha=0.8)
    ax_dev_orig.axhline(thr_orig, color='red', linestyle='--', linewidth=1.5,
                        label=f'Threshold: {thr_orig:.4f}')
    for t in lick_t_orig:
        ax_dev_orig.axvline(t, color='red', alpha=0.3, linewidth=0.8)
    ax_dev_orig.set_title(f'Original deviation  ({len(lick_t_orig)} licks)', fontsize=10)
    ax_dev_orig.set_ylabel('|( val − KDE ) / KDE|', fontsize=9)
    ax_dev_orig.legend(fontsize=8)
    ax_dev_orig.spines['top'].set_visible(False); ax_dev_orig.spines['right'].set_visible(False)

    ax_dev_filt = fig.add_subplot(gs[1, 1])
    ax_dev_filt.plot(time, dev_filt, color='mediumseagreen', linewidth=0.6, alpha=0.8)
    ax_dev_filt.axhline(thr_filt, color='red', linestyle='--', linewidth=1.5,
                        label=f'Threshold: {thr_filt:.4f}')
    for t in lick_t_filt:
        ax_dev_filt.axvline(t, color='red', alpha=0.3, linewidth=0.8)
    ax_dev_filt.set_title(f'SG-detrended deviation  ({len(lick_t_filt)} licks)', fontsize=10)
    ax_dev_filt.set_ylabel('|( val − KDE ) / KDE|', fontsize=9)
    ax_dev_filt.legend(fontsize=8)
    ax_dev_filt.spines['top'].set_visible(False); ax_dev_filt.spines['right'].set_visible(False)

    # ── Row 2: deviation histograms ──────────────────────────────────────────
    ax_hist_orig = fig.add_subplot(gs[2, 0])
    _plot_deviation_hist(ax_hist_orig, dev_orig, thr_orig,
                         color='steelblue', title='Original deviation distribution')

    ax_hist_filt = fig.add_subplot(gs[2, 1])
    _plot_deviation_hist(ax_hist_filt, dev_filt, thr_filt,
                         color='mediumseagreen', title='SG-filtered deviation distribution')

    # Shared x-label for bottom row
    for ax in [ax_raw, ax_sg, ax_dev_orig, ax_dev_filt]:
        ax.set_xlabel('Time (s)', fontsize=9)

    return fig


def _plot_deviation_hist(ax, deviations, threshold, color, title):
    """Helper: density histogram + KDE curve + threshold line."""
    clean = deviations[np.isfinite(deviations) & (deviations >= 0)]
    if len(clean) == 0:
        return
    x_max = np.percentile(clean, 99.5) * 1.5
    ax.hist(clean, bins=80, density=True, color=color, alpha=0.5,
            edgecolor='white', linewidth=0.3)
    try:
        kde_curve = stats.gaussian_kde(clean, bw_method='scott')
        x_eval = np.linspace(0, x_max, 600)
        ax.plot(x_eval, kde_curve(x_eval), color='darkorange', linewidth=1.5)
    except Exception:
        pass
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold: {threshold:.4f}')
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Deviation', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("CAPACITIVE FILTER TEST SCRIPT")
    print("="*60)

    # ── File path ────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\nEnter path to CSV file: ").strip().strip('"').strip("'")

    if not os.path.exists(csv_path):
        print(f"✗ File not found: {csv_path}")
        return

    df = _load_csv(csv_path)
    if df is None:
        return
    filename = os.path.basename(csv_path)

    # ── Sampling rate & suggested window ────────────────────────────────────
    fs = _estimate_sampling_rate(df)
    print(f"✓ Estimated sampling rate: {fs:.1f} Hz")
    suggested_window = _make_window_odd(int(round(fs * 7)))   # 7 s worth of samples
    print(f"  Suggested SG window (7 s): {suggested_window} samples")

    # ── Parameters ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    win_input = input(f"SG window length (samples, must be odd) [default={suggested_window}]: ").strip()
    window_length = int(win_input) if win_input else suggested_window
    window_length = _make_window_odd(window_length)

    poly_input = input("SG polynomial order [default=3]: ").strip()
    polyorder = int(poly_input) if poly_input else 3

    thr_input = input("Detection threshold [default=dynamic KDE valley, min_gap=1]: ").strip()
    manual_threshold = float(thr_input) if thr_input else None

    print(f"\n→ SG window: {window_length} samples  ({window_length / fs:.2f} s)")
    print(f"→ SG poly order: {polyorder}")
    print(f"→ Threshold: {'dynamic (KDE valley, min_gap=1)' if manual_threshold is None else manual_threshold}")

    # ── Original pipeline ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ORIGINAL (unfiltered) pipeline")
    print("="*60)
    df_norm_orig, events_orig, kde_orig, thr_orig = _run_pipeline(df.copy(), threshold=manual_threshold)
    df['capacitive_value_deviation'] = df_norm_orig['capacitive_value_deviation'].values
    n_orig = int(events_orig['capacitive_value_event'].sum())
    print(f"  KDE baseline : {kde_orig:.4f}")
    print(f"  Threshold    : {thr_orig:.4f}")
    print(f"  Licks detected: {n_orig}")

    # ── SG-filtered pipeline ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("SG-FILTERED pipeline")
    print("="*60)
    df_filt = apply_sg_baseline(df.copy(), window_length=window_length, polyorder=polyorder)
    # Swap the capacitive column to the detrended version
    df_filt_for_lda = df_filt.rename(columns={'capacitive_value_filtered': '_cap_filt_tmp'}).copy()
    df_filt_for_lda['capacitive_value'] = df_filt_for_lda['_cap_filt_tmp']
    df_norm_filt, events_filt, kde_filt, thr_filt = _run_pipeline(df_filt_for_lda, threshold=manual_threshold)
    df_filt['capacitive_value_deviation'] = df_norm_filt['capacitive_value_deviation'].values
    n_filt = int(events_filt['capacitive_value_event'].sum())
    print(f"  KDE baseline (filtered): {kde_filt:.4f}")
    print(f"  Threshold (filtered)   : {thr_filt:.4f}")
    print(f"  Licks detected         : {n_filt}")

    delta = n_filt - n_orig
    sign = "+" if delta >= 0 else ""
    print(f"\n  Change in lick count: {sign}{delta}")

    # ── Comparison figure ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOT")
    print("="*60)

    fig = plot_comparison(
        df_raw=df,
        df_filt=df_filt,
        events_orig=events_orig,
        events_filt=events_filt,
        kde_orig=kde_orig,
        kde_filt=kde_filt,
        thr_orig=thr_orig,
        thr_filt=thr_filt,
        window_length=window_length,
        polyorder=polyorder,
        filename=filename,
    )
    plt.show()
    print("✓ Comparison plot displayed")

    # ── Save ──────────────────────────────────────────────────────────────
    save_input = input("\nSave figure? (y/n) [default=n]: ").strip().lower()
    if save_input in ['y', 'yes']:
        stem = os.path.splitext(filename)[0]
        default_out = os.path.join(os.getcwd(), f"{stem}_filter_comparison_w{window_length}_p{polyorder}.svg")
        out_path = input(f"Output path [default={default_out}]: ").strip() or default_out
        if not out_path.lower().endswith('.svg'):
            out_path += '.svg'
        try:
            fig.savefig(out_path, format='svg', bbox_inches='tight')
            print(f"✓ Saved to: {out_path}")
        except Exception as e:
            print(f"✗ Save failed: {e}")

    print(f"\nDone. Original: {n_orig} licks  |  SG-filtered: {n_filt} licks")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
