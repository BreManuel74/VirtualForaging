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

def detect_events_above_threshold(
    df: pd.DataFrame,
    column: str = 'capacitive_value',
    threshold: float = None
) -> tuple:
    """Detect time points where KDE normalized deviation exceeds the threshold.
    
    Creates boolean column indicating when the deviation peaks above the threshold.
    Uses scipy.signal.find_peaks for robust peak detection in discrete sampled data.
    
    If threshold is None, automatically calculates a dynamic threshold as max_deviation / 2.
    
    Parameters:
        df: DataFrame with Time_sec and deviation column (from compute_KDE_normalizations)
        column: Name of the capacitive column (default: 'capacitive_value')
        threshold: Threshold value for peak detection. If None (default), calculates dynamically
                   as max_deviation / 2
        
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
        max_deviation = deviations.max()
        threshold = max_deviation / 2.0
        # print(f"[Dynamic Threshold] Max deviation: {max_deviation:.4f}")
        # print(f"[Dynamic Threshold] Calculated threshold: {threshold:.4f} (max/2)")
    
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
