"""
Modular Lick Detection Algorithm

This module provides a reusable set of functions for detecting and analyzing lick events
from capacitive sensor data. It can be imported and used across multiple analysis scripts.

Expected data format:
    - CSV with columns: arduino_timestamp, elapsed_time, capacitive_value
    - Must have Time_sec column (elapsed_time or arduino_timestamp/1000)
    - capacitive_value column (or specify different column name)

Core functionality:
    - KDE (Kernel Density Estimation) normalization of capacitive data
    - Event detection using threshold-based peak detection
    - Inter-lick interval (ILI) computation
    - Lick bout detection and analysis
    - Visualization of signals, deviations, and detected events

Usage:
    import pandas as pd
    import lick_detection_algorithm as lda
    
    # Load capacitive CSV
    df = pd.read_csv("capacitive.csv")
    df['Time_sec'] = df['elapsed_time']
    
    # Compute KDE normalization
    kde_val = lda.compute_KDE(df, 'capacitive_value')
    df = lda.compute_KDE_normalizations(df, 'capacitive_value', kde_val)
    
    # Detect lick events (dynamic threshold - recommended)
    events_df, threshold = lda.detect_events_above_threshold(df, 'capacitive_value')
    
    # Compute inter-lick intervals
    ili_array = lda.compute_inter_lick_intervals(events_df)
    
    # Detect lick bouts
    bout_dict = lda.compute_lick_bouts(events_df, ili_cutoff=0.3)
    
    # Visualize results
    lda.plot_summary(df, events_df, kde_value=kde_val, threshold=threshold, bout_dict=bout_dict)

Author: Brenna Manuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks


# ============================================================================
# KDE NORMALIZATION FUNCTIONS
# ============================================================================

def compute_KDE(df: pd.DataFrame, column: str = 'capacitive_value') -> float:
    """Compute the KDE (Kernel Density Estimation) peak for the capacitive column.
    
    The KDE peak represents the most probable value (mode) in the distribution of 
    capacitive readings. This is used as the baseline for normalization.
    
    Parameters:
        df: DataFrame containing the capacitive column
        column: Name of the capacitive column (default: 'capacitive_value')
        
    Returns:
        The KDE peak value. If KDE computation fails, returns the mean.
        Returns None if insufficient data.
        
    Example:
        >>> kde_baseline = compute_KDE(df, 'capacitive_value')
        >>> print(f"Baseline: {kde_baseline:.2f}")
        Baseline: 245.32
    """
    series = pd.to_numeric(df[column], errors="coerce")
    series = series.dropna()
    
    if len(series) > 1:  # Need at least 2 points for KDE
        try:
            # Create KDE
            kde = stats.gaussian_kde(series)
            # Create evaluation points around the data range
            min_val, max_val = series.min(), series.max()
            x_eval = np.linspace(min_val, max_val, 1000)
            # Find the peak of the KDE
            density = kde(x_eval)
            peak_idx = np.argmax(density)
            return x_eval[peak_idx]
        except Exception:
            # Fall back to mean if KDE fails
            return series.mean()
    elif len(series) == 1:
        return series.iloc[0]
    else:
        return None


def compute_KDE_normalizations(df: pd.DataFrame, column: str, kde_value: float) -> pd.DataFrame:
    """Compute KDE normalization: abs((value - KDE) / KDE).
    
    Creates a new column with suffix '_deviation' containing the absolute normalized 
    deviation from the KDE baseline: abs((capacitance_value - KDE) / KDE).
    This normalization makes lick detection threshold-based.
    
    Parameters:
        df: DataFrame containing the capacitive column
        column: Name of the capacitive column
        kde_value: The KDE peak value (from compute_KDE)
        
    Returns:
        Copy of the dataframe with new '{column}_deviation' column added
        
    Example:
        >>> kde_val = compute_KDE(df, 'capacitive_value')
        >>> df_normalized = compute_KDE_normalizations(df, 'capacitive_value', kde_val)
        >>> print(df_normalized.columns)
        Index(['capacitive_value', 'Time_sec', 'capacitive_value_deviation'])
    """
    df_with_normalizations = df.copy()
    
    if kde_value is not None and kde_value != 0 and column in df.columns:
        # Compute KDE normalization: abs((value - KDE) / KDE)
        cap_series = pd.to_numeric(df[column], errors="coerce")
        deviation_col = f"{column}_deviation"
        df_with_normalizations[deviation_col] = abs((cap_series - kde_value) / kde_value)
    else:
        # If KDE is None or zero, set normalization to NaN
        deviation_col = f"{column}_deviation"
        df_with_normalizations[deviation_col] = pd.NA
    
    return df_with_normalizations


# ============================================================================
# EVENT DETECTION FUNCTIONS
# ============================================================================

def _kde_valley_search(x: np.ndarray, density: np.ndarray, min_deviation_gap: float = 10.0):
    """Shared valley-search logic for _compute_kde_valley_threshold.

    Returns (valley_x, fwhm_x) where valley_x is the earliest valid valley past the
    noise FWHM, and fwhm_x is the FWHM right edge (fallback). Either may be None.

    Two guards prevent spurious valleys inside the noise distribution:

    1. Minimum deviation gap: the valley search starts at whichever boundary is
       furthest right among (a) the FWHM right edge and (b) the point where the
       x-axis value is at least `min_deviation_gap` units beyond the noise peak.
       This is enforced in actual deviation-value units, not grid-index units, so
       it is scale-invariant regardless of the eval-grid resolution or data range.

    2. Post-valley signal peak: a genuine noise/signal boundary must have a
       signal peak rising after the valley. If only a decaying tail follows, the
       "valley" is a noise-tail artefact and is rejected.

    The earliest (not deepest) qualifying valley is chosen. If lick amplitudes are
    bimodal (e.g. a population of strong licks ~20 deviation units above a weaker
    population), the density can show noise -> weak-lick -> strong-lick bumps, and
    the valley between the two lick bumps is often deeper than the noise/signal
    valley. Picking the deepest valley would then push the threshold above the
    weak-lick population, misclassifying it as noise. Picking the earliest
    qualifying valley anchors the threshold at the true noise/signal boundary
    regardless of how many signal sub-populations exist above it.
    """
    peaks, _ = find_peaks(density)
    if len(peaks) == 0:
        return None, None

    noise_peak_idx = peaks[0]
    noise_peak_x = float(x[noise_peak_idx])
    half_max = density[noise_peak_idx] / 2.0
    # The fallback threshold must be AT LEAST this far into the deviation axis.
    min_gap_x = noise_peak_x + min_deviation_gap

    right_half = density[noise_peak_idx:]
    below_half = np.where(right_half < half_max)[0]
    if len(below_half) == 0:
        return None, None

    fwhm_right_idx = noise_peak_idx + below_half[0]
    fwhm_x = float(x[fwhm_right_idx])
    # Fallback is always at least at the minimum gap, even when no valley found.
    fallback_x = max(fwhm_x, min_gap_x)

    # Guard 1: enforce minimum deviation-value gap from the noise peak.
    # Find the first index where x >= noise_peak_x + min_deviation_gap.
    gap_indices = np.where(x >= min_gap_x)[0]
    if len(gap_indices) == 0:
        # The required gap extends beyond the eval range entirely.
        return None, fallback_x
    min_gap_idx = gap_indices[0]

    search_start = max(fwhm_right_idx, min_gap_idx)
    if search_start >= len(density):
        return None, fallback_x

    post_search_density = density[search_start:]
    valleys_relative, _ = find_peaks(-post_search_density)
    if len(valleys_relative) == 0:
        return None, fallback_x

    # Walk valleys in ascending (earliest-first) order rather than picking the
    # deepest one — see docstring for why this guards against heterogeneous
    # lick-amplitude sub-populations. Guard 2 (post-valley signal peak) is
    # applied to each candidate until one qualifies.
    for valley_rel in np.sort(valleys_relative):
        valley_idx = search_start + valley_rel
        post_valley_peaks, _ = find_peaks(density[valley_idx:])
        if len(post_valley_peaks) > 0:
            return float(x[valley_idx]), fwhm_x

    return None, fallback_x


def _compute_kde_valley_threshold(
    deviations: np.ndarray,
    min_deviation_gap: float = None,
    min_gap_fraction: float = 0.2,
    min_gap_floor: float = 3.0,
    min_gap_ceiling: float = 15.0
) -> float:
    """Find the noise/signal boundary using FWHM-gated KDE valley detection.

    Uses a two-pass strategy to handle both normal sessions (many licks) and
    edge cases (zero or one lick events):

    Pass 1 — standard range (percentile 99.5): preserves the grid resolution
        needed to detect the noise/lick valley in typical sessions. A single
        lick whose deviation exceeds the 99.5th percentile is not visible here,
        but that is fine because any lick cluster present in a normal session
        will be within this range.

    Pass 2 — extended range (full data max, 3000 points): only attempted when
        Pass 1 finds no valley AND the data contains values beyond the Pass 1
        range. This catches the single-lick edge case where the lone lick point
        was clipped from the first-pass eval grid.

    Falls back to the FWHM right edge (outer noise boundary) when no valley is
    found in either pass. For zero-lick sessions, falls back to max/2.

    The minimum deviation gap (see _kde_valley_search) no longer defaults to a
    fixed constant. A fixed gap of ~10 works for files where lick deviations
    reach large magnitudes, but is too large for files where raw capacitance
    never rises far above baseline (e.g. max capacitance < 200) — there, the
    gap can skip past the entire lick population and only the FWHM/fallback
    threshold gets used. Instead, each pass computes its own gap as
    `min_gap_fraction` of that pass's eval-range scale (the 99.5th percentile
    for Pass 1, the data max for Pass 2), clamped to [min_gap_floor,
    min_gap_ceiling] so a single outlier deviation value can't stretch or
    collapse the gap to an unsafe extreme. Pass an explicit `min_deviation_gap`
    to opt out of this scaling and use a fixed absolute gap instead.

    Parameters:
        deviations: 1-D array of non-negative deviation values
        min_deviation_gap: Fixed absolute gap override. If None (default), the
            gap is auto-scaled per pass via `min_gap_fraction` and clamped to
            [min_gap_floor, min_gap_ceiling].
        min_gap_fraction: Fraction of each pass's eval-range scale used as the
            gap when min_deviation_gap is None (default 0.2).
        min_gap_floor: Minimum allowed adaptive gap (default 3.0).
        min_gap_ceiling: Maximum allowed adaptive gap (default 15.0).

    Returns:
        Threshold value (float)
    """
    clean = deviations[np.isfinite(deviations) & (deviations >= 0)]
    if len(clean) < 10:
        return float(np.nanmax(deviations)) / 2.0

    try:
        kde = stats.gaussian_kde(clean, bw_method='scott')

        # --- Pass 1: standard range (works for sessions with many licks) ---
        p995 = np.percentile(clean, 99.5)
        x1 = np.linspace(0, p995, 1000)
        gap1 = min_deviation_gap if min_deviation_gap is not None else np.clip(
            min_gap_fraction * p995, min_gap_floor, min_gap_ceiling
        )
        valley1, fwhm1 = _kde_valley_search(x1, kde(x1), min_deviation_gap=gap1)
        if valley1 is not None:
            return valley1

        # --- Pass 2: extended range (catches rare / single-lick edge cases) ---
        # Only attempt if there is meaningful data beyond the Pass 1 range.
        data_max = float(clean.max())
        if data_max > x1[-1] * 1.05:
            x2 = np.linspace(0, data_max, 3000)
            gap2 = min_deviation_gap if min_deviation_gap is not None else np.clip(
                min_gap_fraction * data_max, min_gap_floor, min_gap_ceiling
            )
            valley2, fwhm2 = _kde_valley_search(x2, kde(x2), min_deviation_gap=gap2)
            if valley2 is not None:
                return valley2
            # Use FWHM from the extended pass if available
            if fwhm2 is not None:
                return fwhm2

        # Fall back to FWHM from Pass 1, then max/2
        return fwhm1 if fwhm1 is not None else float(np.nanmax(deviations)) / 2.0

    except Exception:
        return float(np.nanmax(deviations)) / 2.0


def detect_events_above_threshold(
    df: pd.DataFrame,
    column: str = 'capacitive_value',
    threshold: float = None,
    min_deviation_gap: float = None,
    min_gap_fraction: float = 0.2,
    min_gap_floor: float = 4.5,
    min_gap_ceiling: float = 15.0
) -> tuple:
    """Detect time points where KDE normalized deviation exceeds the threshold.
    
    Creates boolean column indicating when the deviation peaks above the threshold.
    Uses scipy.signal.find_peaks for robust peak detection in discrete sampled data.
    
    If threshold is None, automatically calculates a dynamic threshold using KDE valley
    detection: a KDE is fit to the deviation distribution, the dominant noise peak near 0
    is located, and the first valley after that peak is used as the threshold. This valley
    represents the natural boundary between sensor noise and true lick events. Falls back
    to max_deviation / 2 if the distribution is unimodal.
    
    Parameters:
        df: DataFrame with Time_sec and deviation column (from compute_KDE_normalizations)
        column: Name of the capacitive column (default: 'capacitive_value')
        threshold: Threshold value for peak detection. If None (default), calculates dynamically
                   using KDE valley detection (falls back to max_deviation / 2 if unimodal)
        min_deviation_gap: Fixed absolute minimum-gap override passed to the valley search.
                   If None (default), the gap is auto-scaled per file via `min_gap_fraction`,
                   clamped to [min_gap_floor, min_gap_ceiling].
        min_gap_fraction: Fraction of the file's own deviation-range scale used as the
                   minimum gap when min_deviation_gap is None (default 0.2).
        min_gap_floor: Minimum allowed adaptive gap (default 3.0).
        min_gap_ceiling: Maximum allowed adaptive gap (default 15.0).
        
    Returns:
        Tuple of (DataFrame, threshold_used) where:
            DataFrame contains columns:
                - Time_sec: Time in seconds
                - {column}_event: Boolean indicating detected peaks above threshold
                - {column}_deviation: Original deviation value
                - {column}_derivative: First-order derivative of deviation
            threshold_used: The threshold value that was used (calculated or provided)
            
    Example:
        >>> # Dynamic threshold (recommended)
        >>> events_df, threshold = detect_events_above_threshold(df_normalized, 'capacitive_value')
        >>> print(f"Used threshold: {threshold:.4f}")
        >>> print(f"Lick events: {events_df['capacitive_value_event'].sum()}")
        
        >>> # Static threshold
        >>> events_df, threshold = detect_events_above_threshold(df_normalized, 'capacitive_value', threshold=0.01)
        >>> print(f"Lick events: {events_df['capacitive_value_event'].sum()}")
    """
    result = pd.DataFrame()
    result['Time_sec'] = df['Time_sec']
    
    dev_col = f"{column}_deviation"
    event_col = f"{column}_event"
    deriv_col = f"{column}_derivative"
    
    if dev_col not in df.columns:
        result[event_col] = False
        result[dev_col] = np.nan
        result[deriv_col] = np.nan
        return result, threshold
    
    # Get deviations and calculate first-order derivative
    deviations = pd.to_numeric(df[dev_col], errors="coerce")
    result[dev_col] = deviations
    
    # Calculate first-order derivative using forward difference
    clean_deviations = deviations.fillna(0)
    derivative = np.zeros_like(clean_deviations)
    derivative[:-1] = np.diff(clean_deviations)  # Forward difference
    derivative[-1] = derivative[-2] if len(derivative) > 1 else 0  # Handle last point
    result[deriv_col] = derivative
    
    # Calculate dynamic threshold if not provided
    if threshold is None:
        threshold = _compute_kde_valley_threshold(
            clean_deviations.values,
            min_deviation_gap=min_deviation_gap,
            min_gap_fraction=min_gap_fraction,
            min_gap_floor=min_gap_floor,
            min_gap_ceiling=min_gap_ceiling
        )
    
    # Find peaks in the deviation signal using scipy.signal.find_peaks
    peaks, _ = find_peaks(clean_deviations, height=threshold, distance=1)
    
    # Create boolean mask for detected peaks
    peak_mask = np.zeros(len(clean_deviations), dtype=bool)
    peak_mask[peaks] = True
    
    result[event_col] = peak_mask
    
    return result, threshold


# ============================================================================
# INTER-LICK INTERVAL FUNCTIONS
# ============================================================================

def compute_inter_lick_intervals(
    events_df: pd.DataFrame,
    column: str = 'capacitive_value'
) -> np.ndarray:
    """Compute inter-lick intervals (ILI).
    
    Finds all time points where events occur (event == True), then computes 
    the time difference between consecutive events.
    
    Parameters:
        events_df: DataFrame from detect_events_above_threshold with Time_sec and event column
        column: Name of the capacitive column (default: 'capacitive_value')
        
    Returns:
        Array of inter-lick intervals (in seconds).
        If fewer than 2 events, returns an empty array.
        
    Example:
        >>> ili_array = compute_inter_lick_intervals(events_df, 'capacitive_value')
        >>> print(f"Mean ILI: {ili_array.mean():.3f}s")
        Mean ILI: 0.237s
        >>> print(f"First 5 ILIs: {ili_array[:5]}")
        First 5 ILIs: [0.245 0.189 0.312 0.276 0.198]
    """
    event_col = f"{column}_event"
    
    if event_col not in events_df.columns:
        return np.array([])
    
    # Get timestamps where events occurred
    event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
    
    if len(event_times) < 2:
        # Need at least 2 events to compute an interval
        return np.array([])
    
    # Compute differences between consecutive event times
    intervals = np.diff(event_times)
    return intervals


# ============================================================================
# LICK BOUT DETECTION FUNCTIONS
# ============================================================================

def compute_lick_bouts(
    events_df: pd.DataFrame,
    column: str = 'capacitive_value',
    ili_cutoff: float = 0.3
) -> dict:
    """Compute lick bouts using an inter-lick interval cutoff.
    
    A lick bout is a sequence of consecutive lick events where the time between
    events (ILI) is less than the cutoff. When ILI >= cutoff, a new bout begins.
    This is the standard method for identifying discrete licking episodes.
    
    Parameters:
        events_df: DataFrame from detect_events_above_threshold with Time_sec and event column
        column: Name of the capacitive column (default: 'capacitive_value')
        ili_cutoff: Maximum ILI (in seconds) to consider events part of the same bout (default 0.3s)
        
    Returns:
        Dictionary containing:
            - 'bout_count': Total number of bouts
            - 'bout_sizes': Array of lick counts per bout
            - 'bout_durations': Array of bout durations (in seconds)
            - 'bout_start_times': Array of bout start times (in seconds)
            - 'bout_end_times': Array of bout end times (in seconds)
        
    Example:
        >>> bouts = compute_lick_bouts(events_df, 'capacitive_value', ili_cutoff=0.3)
        >>> print(f"Total bouts: {bouts['bout_count']}")
        Total bouts: 23
        >>> print(f"Average bout size: {bouts['bout_sizes'].mean():.1f} licks")
        Average bout size: 6.2 licks
        >>> print(f"Average bout duration: {bouts['bout_durations'].mean():.2f}s")
        Average bout duration: 1.45s
    """
    event_col = f"{column}_event"
    
    if event_col not in events_df.columns:
        return {
            'bout_count': 0,
            'bout_sizes': np.array([]),
            'bout_durations': np.array([]),
            'bout_start_times': np.array([]),
            'bout_end_times': np.array([])
        }
    
    # Get timestamps where events occurred
    event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
    
    if len(event_times) == 0:
        return {
            'bout_count': 0,
            'bout_sizes': np.array([]),
            'bout_durations': np.array([]),
            'bout_start_times': np.array([]),
            'bout_end_times': np.array([])
        }
    
    # Single event = single bout of size 1
    if len(event_times) == 1:
        return {
            'bout_count': 1,
            'bout_sizes': np.array([1]),
            'bout_durations': np.array([0.0]),
            'bout_start_times': event_times,
            'bout_end_times': event_times
        }
    
    # Compute ILIs
    intervals = np.diff(event_times)
    
    # Identify bout boundaries: wherever ILI >= cutoff, a new bout starts
    bout_breaks = intervals >= ili_cutoff
    
    # Build bouts by iterating through events
    bout_sizes = []
    bout_start_times = []
    bout_end_times = []
    
    current_bout_start = event_times[0]
    current_bout_size = 1
    
    for i in range(len(intervals)):
        if bout_breaks[i]:
            # End current bout
            bout_sizes.append(current_bout_size)
            bout_start_times.append(current_bout_start)
            bout_end_times.append(event_times[i])
            
            # Start new bout
            current_bout_start = event_times[i + 1]
            current_bout_size = 1
        else:
            # Continue current bout
            current_bout_size += 1
    
    # Don't forget the last bout
    bout_sizes.append(current_bout_size)
    bout_start_times.append(current_bout_start)
    bout_end_times.append(event_times[-1])
    
    # Convert to numpy arrays
    bout_sizes = np.array(bout_sizes)
    bout_start_times = np.array(bout_start_times)
    bout_end_times = np.array(bout_end_times)
    bout_durations = bout_end_times - bout_start_times
    
    return {
        'bout_count': len(bout_sizes),
        'bout_sizes': bout_sizes,
        'bout_durations': bout_durations,
        'bout_start_times': bout_start_times,
        'bout_end_times': bout_end_times
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_capacitive_signal(
    df: pd.DataFrame,
    column: str = 'capacitive_value',
    kde_value: float = None,
    title: str = None,
    figsize: tuple = (12, 6),
    show: bool = True
) -> plt.Figure:
    """Plot the raw capacitive signal over time.
    
    Parameters:
        df: DataFrame with Time_sec and capacitive column
        column: Name of the capacitive column (default: 'capacitive_value')
        kde_value: If provided, draws a horizontal line at the KDE baseline
        title: Optional plot title
        figsize: Figure size (width, height) in inches
        show: If True, displays the plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> kde_val = compute_KDE(df, 'capacitive_value')
        >>> fig = plot_capacitive_signal(df, 'capacitive_value', kde_value=kde_val)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot capacitive signal
    ax.plot(df['Time_sec'], df[column], linewidth=1, color='#1f77b4', alpha=0.8)
    
    # Add KDE baseline if provided
    if kde_value is not None:
        ax.axhline(y=kde_value, color='red', linestyle='--', linewidth=2, 
                   label=f'KDE Baseline: {kde_value:.2f}', alpha=0.7)
        ax.legend(loc='best')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Capacitive Value (a.u.)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Capacitive Signal over Time', fontsize=13)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    ax.set_xlim(left=0)
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_deviation_signal(
    df: pd.DataFrame,
    column: str = 'capacitive_value',
    title: str = None,
    figsize: tuple = (12, 6),
    show: bool = True
) -> plt.Figure:
    """Plot the KDE normalized deviation signal over time.
    
    Parameters:
        df: DataFrame with Time_sec and deviation column (from compute_KDE_normalizations)
        column: Name of the capacitive column (default: 'capacitive_value')
        title: Optional plot title
        figsize: Figure size (width, height) in inches
        show: If True, displays the plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_deviation_signal(df_normalized, 'capacitive_value')
    """
    dev_col = f"{column}_deviation"
    
    if dev_col not in df.columns:
        raise ValueError(f"Deviation column '{dev_col}' not found. Run compute_KDE_normalizations first.")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot deviation signal
    ax.plot(df['Time_sec'], df[dev_col], linewidth=1, color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('KDE Normalized Deviation', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('KDE Normalized Deviation over Time', fontsize=13)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_detected_events(
    events_df: pd.DataFrame,
    column: str = 'capacitive_value',
    threshold: float = 0.01,
    title: str = None,
    figsize: tuple = (12, 6),
    show: bool = True
) -> plt.Figure:
    """Plot the deviation signal with detected lick events highlighted.
    
    Parameters:
        events_df: DataFrame from detect_events_above_threshold
        column: Name of the capacitive column (default: 'capacitive_value')
        threshold: Threshold value used for detection (for visualization)
        title: Optional plot title
        figsize: Figure size (width, height) in inches
        show: If True, displays the plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_detected_events(events_df, 'capacitive_value', threshold=0.01)
    """
    dev_col = f"{column}_deviation"
    event_col = f"{column}_event"
    
    if dev_col not in events_df.columns or event_col not in events_df.columns:
        raise ValueError("Required columns not found. Run detect_events_above_threshold first.")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot deviation signal
    ax.plot(events_df['Time_sec'], events_df[dev_col], linewidth=1, 
            color='#2ca02c', alpha=0.6, label='Deviation')
    
    # Add threshold line
    ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold}', alpha=0.7)
    
    # Highlight detected events
    event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
    event_deviations = events_df.loc[events_df[event_col] == True, dev_col].values
    
    if len(event_times) > 0:
        ax.scatter(event_times, event_deviations, color='red', s=50, 
                   marker='o', zorder=5, label=f'Detected Events (n={len(event_times)})')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('KDE Normalized Deviation', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Lick Event Detection', fontsize=13)
    
    ax.legend(loc='best', fontsize=10)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_lick_raster(
    events_df: pd.DataFrame,
    column: str = 'capacitive_value',
    bout_dict: dict = None,
    title: str = None,
    figsize: tuple = (12, 4),
    show: bool = True
) -> plt.Figure:
    """Plot lick events as a raster (vertical tick marks) over time.
    
    Optionally shows bout boundaries if bout_dict is provided.
    
    Parameters:
        events_df: DataFrame from detect_events_above_threshold
        column: Name of the capacitive column (default: 'capacitive_value')
        bout_dict: Optional dictionary from compute_lick_bouts (shows bout boundaries)
        title: Optional plot title
        figsize: Figure size (width, height) in inches
        show: If True, displays the plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_lick_raster(events_df, 'capacitive_value', bout_dict=bouts)
    """
    event_col = f"{column}_event"
    
    if event_col not in events_df.columns:
        raise ValueError(f"Event column '{event_col}' not found. Run detect_events_above_threshold first.")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get event times
    event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
    
    # Plot licks as vertical lines
    if len(event_times) > 0:
        ax.vlines(event_times, 0, 1, colors='black', linewidth=1.5, alpha=0.8)
    
    # Add bout boundaries if provided
    if bout_dict is not None and bout_dict['bout_count'] > 0:
        for start, end in zip(bout_dict['bout_start_times'], bout_dict['bout_end_times']):
            ax.axvspan(start, end, alpha=0.2, color='blue')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Lick Events', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    if title:
        ax.set_title(title, fontsize=13)
    else:
        n_events = len(event_times)
        n_bouts = bout_dict['bout_count'] if bout_dict else 0
        if bout_dict:
            ax.set_title(f'Lick Raster: {n_events} licks in {n_bouts} bouts', fontsize=13)
        else:
            ax.set_title(f'Lick Raster: {n_events} licks', fontsize=13)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    ax.set_xlim(left=0)
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_summary(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    column: str = 'capacitive_value',
    kde_value: float = None,
    threshold: float = 0.01,
    bout_dict: dict = None,
    title: str = None,
    figsize: tuple = (12, 10),
    show: bool = True
) -> plt.Figure:
    """Create a comprehensive summary plot with all visualizations.
    
    Shows 4 subplots:
    1. Raw capacitive signal with KDE baseline
    2. KDE normalized deviation
    3. Deviation with detected events
    4. Lick raster with bout boundaries
    
    Parameters:
        df: DataFrame with Time_sec, capacitive, and deviation columns
        events_df: DataFrame from detect_events_above_threshold
        column: Name of the capacitive column (default: 'capacitive_value')
        kde_value: KDE baseline value (for subplot 1)
        threshold: Threshold value used for detection
        bout_dict: Optional dictionary from compute_lick_bouts
        title: Optional overall title
        figsize: Figure size (width, height) in inches
        show: If True, displays the plot
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_summary(df, events_df, kde_value=kde_val, threshold=0.01, bout_dict=bouts)
    """
    dev_col = f"{column}_deviation"
    event_col = f"{column}_event"
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Subplot 1: Raw capacitive signal
    ax1 = axes[0]
    ax1.plot(df['Time_sec'], df[column], linewidth=1, color='#1f77b4', alpha=0.8)
    if kde_value is not None:
        ax1.axhline(y=kde_value, color='red', linestyle='--', linewidth=2, 
                    label=f'KDE: {kde_value:.2f}', alpha=0.7)
        ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylabel('Capacitive Value', fontsize=11)
    ax1.set_title('Raw Capacitive Signal', fontsize=12)
    
    # Subplot 2: KDE normalized deviation
    ax2 = axes[1]
    if dev_col in df.columns:
        ax2.plot(df['Time_sec'], df[dev_col], linewidth=1, color='#2ca02c', alpha=0.8)
        ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Deviation', fontsize=11)
    ax2.set_title('KDE Normalized Deviation', fontsize=12)
    
    # Subplot 3: Deviation with detected events
    ax3 = axes[2]
    if dev_col in events_df.columns and event_col in events_df.columns:
        ax3.plot(events_df['Time_sec'], events_df[dev_col], linewidth=1, 
                color='#2ca02c', alpha=0.6)
        ax3.axhline(y=threshold, color='blue', linestyle='--', linewidth=1.5, 
                   label=f'Threshold: {threshold}', alpha=0.7)
        
        event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
        event_deviations = events_df.loc[events_df[event_col] == True, dev_col].values
        
        if len(event_times) > 0:
            ax3.scatter(event_times, event_deviations, color='red', s=30, 
                       marker='o', zorder=5, label=f'Events (n={len(event_times)})')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_ylim(bottom=0)
    ax3.set_ylabel('Deviation', fontsize=11)
    ax3.set_title('Detected Lick Events', fontsize=12)
    
    # Subplot 4: Lick raster
    ax4 = axes[3]
    if event_col in events_df.columns:
        event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
        if len(event_times) > 0:
            ax4.vlines(event_times, 0, 1, colors='black', linewidth=1.5, alpha=0.8)
        
        # Add bout boundaries if provided
        if bout_dict is not None and bout_dict['bout_count'] > 0:
            for start, end in zip(bout_dict['bout_start_times'], bout_dict['bout_end_times']):
                ax4.axvspan(start, end, alpha=0.2, color='blue')
    
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Licks', fontsize=11)
    
    n_events = len(event_times) if event_col in events_df.columns else 0
    n_bouts = bout_dict['bout_count'] if bout_dict else 0
    if bout_dict:
        ax4.set_title(f'Lick Raster: {n_events} licks in {n_bouts} bouts', fontsize=12)
    else:
        ax4.set_title(f'Lick Raster: {n_events} licks', fontsize=12)
    
    # Apply styling to all axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both', length=5)
        ax.set_xlim(left=0)
    
    ax4.spines['left'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig
