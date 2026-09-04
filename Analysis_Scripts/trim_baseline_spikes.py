"""
Capacitive Baseline Spike Trimmer

Detects and removes sections of a raw capacitive recording where the signal
stays consecutively above a threshold value for a minimum duration. These
sections represent baseline drift that has crept into lick-signal territory
and would cause false-positive lick detections.

Detection logic:
    A "spike region" is any contiguous run where
        capacitive_value > spike_threshold
    for at least min_duration_s seconds.

The cleaned CSV is written with spike rows removed. The row timestamps and
elapsed_time values are preserved as-is (no re-numbering) so that downstream
alignment with trial logs and treadmill data is not broken.

Usage:
    python trim_baseline_spikes.py [path/to/file.csv]
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib

# Interactive widgets (buttons, key/mouse events, SpanSelector) require a GUI
# backend. Non-interactive/inline backends (e.g. the Jupyter/VS Code notebook
# "inline" backend) only render static images and silently ignore all mouse
# and key events, which looks exactly like "the buttons don't work".
if not matplotlib.get_backend().lower().startswith(('qt', 'tk', 'wx', 'gtk', 'macosx')):
    for _candidate in ('QtAgg', 'TkAgg'):
        try:
            matplotlib.use(_candidate)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import SpanSelector, Button


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame | None:
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

    df['capacitive_value'] = pd.to_numeric(df['capacitive_value'], errors='coerce')
    print(f"✓ Loaded {len(df)} rows  |  "
          f"{df['Time_sec'].min():.2f}s – {df['Time_sec'].max():.2f}s")
    return df


def _estimate_sampling_rate(df: pd.DataFrame) -> float:
    diffs = df['Time_sec'].diff().dropna()
    median_dt = float(diffs.median())
    return 1.0 / median_dt if median_dt > 0 else 100.0


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def detect_spike_regions(df: pd.DataFrame,
                          spike_threshold: float,
                          min_duration_s: float) -> list[tuple[float, float]]:
    """
    Return a list of (t_start, t_end) pairs for every contiguous run where
    capacitive_value > spike_threshold lasting >= min_duration_s seconds.
    """
    above = (df['capacitive_value'] > spike_threshold).values
    time  = df['Time_sec'].values

    regions = []
    in_run  = False
    run_start_t = None
    run_start_i = None

    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run = True
            run_start_t = time[i]
            run_start_i = i
        elif not flag and in_run:
            run_end_t = time[i - 1]
            if run_end_t - run_start_t >= min_duration_s:
                regions.append((run_start_t, run_end_t))
            in_run = False

    # Handle run that reaches the end of the file
    if in_run:
        run_end_t = time[-1]
        if run_end_t - run_start_t >= min_duration_s:
            regions.append((run_start_t, run_end_t))

    return regions


def remove_spike_regions(df: pd.DataFrame,
                          regions: list[tuple[float, float]]) -> pd.DataFrame:
    """Return df with all rows that fall inside any spike region removed."""
    if not regions:
        return df.copy()

    time = df['Time_sec'].values
    keep = np.ones(len(df), dtype=bool)

    for t_start, t_end in regions:
        keep &= ~((time >= t_start) & (time <= t_end))

    return df[keep].copy()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_spike_regions(df: pd.DataFrame,
                       regions: list[tuple[float, float]],
                       spike_threshold: float,
                       min_duration_s: float,
                       filename: str,
                       interactive: bool = True) -> tuple[plt.Figure, list]:
    """
    Two-panel figure:
      Top:    Raw capacitive signal with spike regions shaded red.
      Bottom: Cleaned signal (spike rows removed).

    If interactive=True, click-drag on the top panel to mark additional
    regions (shaded blue) that the automated detector missed. Use the
    "Select: OFF/ON" button to arm/disarm dragging (so the toolbar's zoom/pan
    keeps working) and the "Undo last" button to remove the most recent
    selection. Returns (fig, manual_regions) where manual_regions is a list
    populated live as selections are made — read it after plt.show() returns
    (i.e. after the window is closed) for the final set.
    """
    time_raw = df['Time_sec'].values
    cap_raw  = df['capacitive_value'].values

    df_clean = remove_spike_regions(df, regions)
    time_clean = df_clean['Time_sec'].values
    cap_clean  = df_clean['capacitive_value'].values

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f'Baseline Spike Trimmer — {filename}\n'
        f'threshold={spike_threshold}, min duration={min_duration_s}s  |  '
        f'{len(regions)} spike region(s) found',
        fontsize=12
    )

    # ── Top: raw + highlighted spikes ───────────────────────────────────────
    def _draw_raw_panel():
        ax_top.cla()
        ax_top.plot(time_raw, cap_raw, color='steelblue', linewidth=0.6, alpha=0.8,
                    label='Raw signal')
        ax_top.axhline(spike_threshold, color='darkorange', linestyle='--', linewidth=1.2,
                       label=f'Spike threshold: {spike_threshold}')
        for t_start, t_end in regions:
            ax_top.axvspan(t_start, t_end, color='red', alpha=0.25)
        for t_start, t_end in manual_regions:
            ax_top.axvspan(t_start, t_end, color='dodgerblue', alpha=0.3)
        ax_top.set_ylabel('Capacitive value', fontsize=10)
        title = ("Raw signal — use the Select button, then drag to mark a region"
                  if interactive else 'Raw signal — spike regions highlighted')
        ax_top.set_title(title, fontsize=10)
        ax_top.legend(handles=legend_handles, fontsize=9)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

    spike_patch = mpatches.Patch(color='red', alpha=0.4, label=f'Spike regions ({len(regions)})')
    legend_handles = [
        plt.Line2D([0], [0], color='steelblue', linewidth=1.5, label='Raw signal'),
        plt.Line2D([0], [0], color='darkorange', linestyle='--', label=f'Spike threshold: {spike_threshold}'),
        spike_patch,
    ]

    manual_regions: list = []

    if interactive:
        legend_handles.append(mpatches.Patch(color='dodgerblue', alpha=0.4,
                                              label='Manual regions (use Select/Undo buttons)'))

        def _on_select(xmin, xmax):
            if xmax <= xmin:
                return
            manual_regions.append((float(xmin), float(xmax)))
            _draw_raw_panel()
            fig.canvas.draw_idle()
            print(f"  + Manual region added: {xmin:.2f}s – {xmax:.2f}s  "
                  f"(total manual: {len(manual_regions)})")

        try:
            span = SpanSelector(ax_top, _on_select, 'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor='dodgerblue'),
                                 interactive=False, button=1)
        except TypeError:
            span = SpanSelector(ax_top, _on_select, 'horizontal', useblit=True,
                                 rectprops=dict(alpha=0.3, facecolor='dodgerblue'),
                                 interactive=False, button=1)

        # Start with selection OFF so the toolbar's zoom/pan (left-click drag)
        # works normally. Clicking "Select: OFF" arms the selector (drag to
        # mark a region); clicking it again disarms it so zoom/pan resumes —
        # both features use left-click drag, so they can't be active at once.
        # A clickable button is used instead of a keyboard shortcut since key
        # events don't reliably reach the canvas across backends/platforms.
        span.active = False

        # Reserve room above the top axes for the two control buttons
        # (applied via tight_layout's rect below, not subplots_adjust, so it
        # isn't undone by the tight_layout() call at the end of this function).
        ax_toggle = fig.add_axes([0.72, 0.94, 0.12, 0.04])
        ax_undo = fig.add_axes([0.85, 0.94, 0.12, 0.04])
        btn_toggle = Button(ax_toggle, 'Select: OFF')
        btn_undo = Button(ax_undo, 'Undo last')

        def _on_toggle_clicked(event):
            span.active = not span.active
            if span.active:
                # The toolbar's zoom/pan tools hold the canvas widgetlock once
                # engaged (even after you're done dragging), which blocks the
                # SpanSelector from ever receiving drag events again. Release
                # it here so selection actually works after zooming/panning.
                toolbar = getattr(fig.canvas, 'toolbar', None)
                mode = getattr(toolbar, 'mode', None) if toolbar is not None else None
                mode_name = getattr(mode, 'name', str(mode) if mode else '')
                if mode_name == 'ZOOM':
                    toolbar.zoom()
                elif mode_name == 'PAN':
                    toolbar.pan()
            btn_toggle.label.set_text(f"Select: {'ON' if span.active else 'OFF'}")
            mode = 'ON — drag to mark a region' if span.active else 'OFF — toolbar zoom/pan enabled'
            print(f"  Region-select mode: {mode}")
            fig.canvas.draw_idle()

        def _on_undo_clicked(event):
            if manual_regions:
                manual_regions.pop()
                _draw_raw_panel()
                fig.canvas.draw_idle()
                print(f"  - Removed last manual region  (remaining: {len(manual_regions)})")

        btn_toggle.on_clicked(_on_toggle_clicked)
        btn_undo.on_clicked(_on_undo_clicked)
        fig._spike_widgets = (span, btn_toggle, btn_undo)  # keep references alive

    _draw_raw_panel()

    # ── Bottom: cleaned signal ───────────────────────────────────────────────
    ax_bot.plot(time_clean, cap_clean, color='mediumseagreen', linewidth=0.6, alpha=0.9,
                label=f'Cleaned signal ({len(df_clean)} rows)')
    ax_bot.axhline(spike_threshold, color='darkorange', linestyle='--', linewidth=1.2,
                   label=f'Spike threshold: {spike_threshold}')
    ax_bot.set_xlabel('Time (s)', fontsize=10)
    ax_bot.set_ylabel('Capacitive value', fontsize=10)
    ax_bot.set_title('Cleaned signal (auto-detected regions removed)', fontsize=10)
    ax_bot.legend(fontsize=9)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.92) if interactive else (0, 0, 1, 1))
    return fig, manual_regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("CAPACITIVE BASELINE SPIKE TRIMMER")
    print("="*60)
    print(f"Matplotlib backend: {matplotlib.get_backend()}  "
          f"(must be an interactive GUI backend, e.g. TkAgg/QtAgg, for buttons/selection to work)")

    # ── File path ────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\nEnter path to capacitive CSV file: ").strip().strip('"').strip("'")

    if not os.path.exists(csv_path):
        print(f"✗ File not found: {csv_path}")
        return

    df = _load_csv(csv_path)
    if df is None:
        return
    filename = os.path.basename(csv_path)

    fs = _estimate_sampling_rate(df)
    cap_vals = df['capacitive_value'].dropna()
    print(f"✓ Sampling rate  : {fs:.1f} Hz")
    print(f"  Signal range   : {cap_vals.min():.1f} – {cap_vals.max():.1f}")
    print(f"  Signal median  : {cap_vals.median():.1f}")

    # ── Parameters ───────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    thr_input = input("Spike threshold (raw capacitive value) [default=50]: ").strip()
    spike_threshold = float(thr_input) if thr_input else 50.0

    dur_input = input("Minimum spike duration in seconds [default=2]: ").strip()
    min_duration_s = float(dur_input) if dur_input else 2.0

    print(f"\n→ Spike threshold : {spike_threshold}")
    print(f"→ Min duration    : {min_duration_s}s")

    # ── Detect ───────────────────────────────────────────────────────────────
    regions = detect_spike_regions(df, spike_threshold, min_duration_s)
    total_spike_s = sum(t1 - t0 for t0, t1 in regions)
    session_duration = df['Time_sec'].max() - df['Time_sec'].min()

    print(f"\n{'─'*40}")
    print(f"  Spike regions found : {len(regions)}")
    if regions:
        print(f"  Total spiked time   : {total_spike_s:.2f}s  "
              f"({100 * total_spike_s / session_duration:.1f}% of session)")
        print(f"\n  Regions:")
        for i, (t0, t1) in enumerate(regions, 1):
            print(f"    [{i}]  {t0:.2f}s – {t1:.2f}s  (duration: {t1-t0:.2f}s)")
    else:
        print("  No spike regions detected with these parameters.")
    print(f"{'─'*40}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, manual_regions = plot_spike_regions(df, regions, spike_threshold, min_duration_s, filename)
    print("\nTip: region-select mode starts OFF so the toolbar's zoom/pan works normally.")
    print("     Click the 'Select: OFF' button to arm selection, then click-drag to mark a region.")
    print("     Click it again to disarm and go back to zooming/panning.")
    print("     Click 'Undo last' to remove the most recent selection. Close the window when done.")
    plt.show()
    print("✓ Plot displayed")

    if manual_regions:
        print(f"\n  Manual regions added : {len(manual_regions)}")
        for i, (t0, t1) in enumerate(manual_regions, 1):
            print(f"    [{i}]  {t0:.2f}s – {t1:.2f}s  (duration: {t1-t0:.2f}s)")

    all_regions = regions + manual_regions

    if not all_regions:
        print("\nNothing to trim — exiting.")
        return

    # ── Confirm trim & save ───────────────────────────────────────────────────
    trim_input = input("\nRemove spike regions and save cleaned CSV? (y/n) [default=n]: ").strip().lower()
    if trim_input not in ['y', 'yes']:
        print("No file saved.")
        return

    df_clean = remove_spike_regions(df, all_regions)
    rows_removed = len(df) - len(df_clean)
    print(f"  Rows removed : {rows_removed}  ({rows_removed/len(df)*100:.1f}%)")
    print(f"  Rows kept    : {len(df_clean)}")

    # Drop the helper Time_sec column if it wasn't in the original file
    original_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if 'Time_sec' not in original_cols:
        df_clean = df_clean.drop(columns=['Time_sec'], errors='ignore')

    stem = os.path.splitext(filename)[0]
    default_out = os.path.join(os.getcwd(), f"{stem}_trimmed.csv")
    out_path = input(f"Output path [default={default_out}]: ").strip() or default_out
    if not out_path.lower().endswith('.csv'):
        out_path += '.csv'

    try:
        df_clean.to_csv(out_path, index=False)
        print(f"✓ Cleaned CSV saved to: {out_path}")
    except Exception as e:
        print(f"✗ Save failed: {e}")

    # ── Optionally save the figure ────────────────────────────────────────────
    save_fig = input("\nSave figure as SVG? (y/n) [default=n]: ").strip().lower()
    if save_fig in ['y', 'yes']:
        svg_path = os.path.splitext(out_path)[0] + '_spike_plot.svg'
        try:
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"✓ Figure saved to: {svg_path}")
        except Exception as e:
            print(f"✗ Figure save failed: {e}")

    print(f"\nDone. {len(all_regions)} region(s) trimmed  |  {rows_removed} rows removed.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
