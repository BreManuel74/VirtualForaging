# Timeline Analysis Script - Refactoring Guide

## Overview

The original `timeline.py` script has been refactored into `timeline_refactored.py` with well-organized, clearly-named functions. This makes the code easier to navigate, maintain, and extend.

## Function Organization

The refactored script is organized into the following sections:

### 1. UTILITY FUNCTIONS

| Function | Purpose |
|----------|---------|
| `safe_literal_eval(val)` | Safely parse string representations of lists |
| `pad_list(lst, length)` | Pad lists to specified length |
| `save_figure(fig, name, output_folder)` | Save matplotlib figures as SVG |
| `setup_plot_style(ax)` | Apply consistent plot styling (remove borders) |

### 2. FILE LOADING FUNCTIONS

| Function | Purpose |
|----------|---------|
| `select_data_folder()` | Open dialog to select data folder |
| `validate_and_find_files(folder_path)` | Check for required files and return file paths |
| `load_data_files(file_paths, has_pupil_data)` | Load all CSV files into DataFrames |
| `create_output_folder(folder_path)` | Create directory for saving figures |

### 3. DATA PREPROCESSING FUNCTIONS

| Function | Purpose |
|----------|---------|
| `process_texture_history(trial_log_df)` | Parse and organize texture change data |
| `create_empty_texture_arrays()` | Generate empty arrays when no texture data exists |
| `interpolate_treadmill_to_capacitive(treadmill_df, capacitive_df)` | Align treadmill data to capacitive timeline |
| `process_pupil_data(pupil_df, frame_log_df, capacitive_df)` | Calculate and interpolate pupil diameter |

### 4. EVENT MATCHING FUNCTIONS

| Function | Purpose |
|----------|---------|
| `match_reward_zones_to_events(trial_log_df, reward_texture_change_time)` | Match reward zone entries to delivery events |
| `match_puff_zones_to_events(trial_log_df, punish_texture_change_time_first)` | Match puff zone entries to delivery events |

### 5. WINDOW EXTRACTION FUNCTIONS

| Function | Purpose |
|----------|---------|
| `create_aligned_windows(time_array, data_array, event_times, window_size)` | Extract time-aligned data windows around events |

### 6. PLOTTING FUNCTIONS: TIMELINE

| Function | Purpose |
|----------|---------|
| `plot_main_timeline(...)` | Create the main multi-panel timeline plot |
| `plot_capacitive_timeline(ax, ...)` | Plot capacitive sensor data with events |
| `plot_treadmill_timeline(ax, ...)` | Plot treadmill speed with events |
| `plot_pupil_timeline(ax, ...)` | Plot pupil diameter with events |
| `add_event_markers(ax, ...)` | Add vertical lines for reward/puff/probe events |
| `add_texture_intervals(ax, ...)` | Add shaded regions for reward/puff zones |

### 7. PLOTTING FUNCTIONS: RASTER PLOTS

| Function | Purpose |
|----------|---------|
| `plot_raster_heatmap(windows_padded, ...)` | Create raster/heatmap plots for aligned data |

### 8. ANALYSIS FUNCTIONS: REWARD ZONES

| Function | Purpose |
|----------|---------|
| `analyze_reward_zones(...)` | Perform complete reward zone analysis |
| `analyze_reward_deliveries(...)` | Analyze data aligned to reward delivery times |

### 9. ANALYSIS FUNCTIONS: PUFF ZONES

| Function | Purpose |
|----------|---------|
| `analyze_puff_zones(...)` | Perform complete puff zone analysis |
| `analyze_puff_deliveries(...)` | Analyze data aligned to puff delivery times |

### 10. MAIN EXECUTION

| Function | Purpose |
|----------|---------|
| `main()` | Orchestrate the complete analysis workflow |

## Usage

Run the refactored script the same way as the original:

```python
python timeline_refactored.py
```

## Key Improvements

1. **Modular Design**: Each function has a single, well-defined purpose
2. **Clear Naming**: Function names clearly describe what they do
3. **Docstrings**: All functions include documentation
4. **Organized Sections**: Related functions are grouped together
5. **Easy Navigation**: Jump directly to the function you need
6. **Maintainability**: Easier to modify, debug, and extend
7. **Reusability**: Functions can be imported and used in other scripts

## Function Call Flow

```
main()
  ├─ select_data_folder()
  ├─ validate_and_find_files()
  ├─ load_data_files()
  ├─ create_output_folder()
  ├─ process_texture_history()
  ├─ interpolate_treadmill_to_capacitive()
  ├─ process_pupil_data()
  ├─ plot_main_timeline()
  │   ├─ plot_capacitive_timeline()
  │   ├─ plot_treadmill_timeline()
  │   ├─ plot_pupil_timeline()
  │   ├─ add_event_markers()
  │   └─ add_texture_intervals()
  ├─ match_reward_zones_to_events()
  ├─ analyze_reward_zones()
  │   ├─ create_aligned_windows()
  │   ├─ plot_raster_heatmap()
  │   └─ analyze_reward_deliveries()
  ├─ match_puff_zones_to_events()
  └─ analyze_puff_zones()
      ├─ create_aligned_windows()
      ├─ plot_raster_heatmap()
      └─ analyze_puff_deliveries()
```

## Quick Reference: Finding What You Need

**Want to modify data loading?** → See "FILE LOADING FUNCTIONS"

**Want to change interpolation?** → See "DATA PREPROCESSING FUNCTIONS"

**Want to adjust plot appearance?** → See "PLOTTING FUNCTIONS: TIMELINE"

**Want to modify raster plots?** → See "PLOTTING FUNCTIONS: RASTER PLOTS"

**Want to change analysis windows?** → See "WINDOW EXTRACTION FUNCTIONS"

**Want to modify event matching logic?** → See "EVENT MATCHING FUNCTIONS"

**Want to add new analyses?** → Add function to relevant section, call from `main()`

## Notes

- The refactored version preserves all functionality from the original
- All plot styling and colormaps are identical to the original:
  - Capacitive rasters use 'binary' colormap (black and white)
  - Treadmill speed rasters use 'coolwarm' colormap with fixed vmin/vmax
  - Reward zone speed: vmin=-300, vmax=300
  - Puff delivery speed: vmin=-400, vmax=400
- Output is identical to the original script
- The original `timeline.py` is preserved for reference
- All analysis logic remains unchanged, only organization improved
