import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from datetime import datetime
import colorsys
import matplotlib.pyplot as plt
from scipy import stats

def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.rand() * 0.3  # Random between 0.7-1.0
        value = 0.7 + np.random.rand() * 0.3       # Random between 0.7-1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def get_behavioral_start_day():
    """Ask user for the day number when behavioral data starts."""
    try:
        # Create a simple input dialog using the native system dialog
        answer = tk.simpledialog.askinteger(
            "Behavioral Data Start Day",
            "Enter the cohort day number when behavioral data starts:\n(e.g., if training day 0 = citric acid day 19, enter 19)",
            initialvalue=19)
        #print(f"Behavioral data starts at cohort day: {answer}")
        return answer
    except Exception as e:
        print(f"Error getting behavioral start day: {e}")
        return 19  # Default value if dialog fails

def load_cohort_data(file_path):
    """Load and process cohort-style CSV data."""
    df = pd.read_csv(file_path)
    
    # Validate that this is actually a cohort file
    required_columns = ['ID', 'Sex']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"This does not appear to be a cohort file. Missing required columns: {', '.join(missing_columns)}")
    
    print(f"\nLoading cohort data from: {os.path.basename(file_path)}")
    
    # Ensure ID and Sex are strings and strip any whitespace
    df['ID'] = df['ID'].astype(str).str.strip()
    df['Sex'] = df['Sex'].astype(str).str.strip()
    
    # Melt the dataframe to convert from wide to long format
    # This makes days into a single column
    melted_df = pd.melt(
        df, 
        id_vars=['ID', 'Sex'], 
        var_name='Day',
        value_name='Value'
    )
    
    # Convert 'Day' column from 'Day0', 'Day1', etc. to numeric
    melted_df['Day'] = melted_df['Day'].str.extract(r'(\d+)').astype(int)
    
    # Convert Value column to numeric, replacing any non-numeric values with NaN
    melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce')
    
    # Get the behavioral start day from user
    behavioral_start_day = get_behavioral_start_day()
    if behavioral_start_day is None:
        raise ValueError("Behavioral start day is required to align the data")
    
    # Adjust day numbers to align with behavioral data
    melted_df['Day'] = melted_df['Day'] - behavioral_start_day
    
    # Sort by ID and Day
    melted_df = melted_df.sort_values(['ID', 'Day'])
    
    # Print day range for verification
    # print(f"\nCohort data day range after alignment: {melted_df['Day'].min()} to {melted_df['Day'].max()}")
    # print("Note: Day 0 corresponds to the first day in the behavioral data")
    
    return melted_df

def analyze_mouse_data(behavior_files, cohort_files):
    """Analyze both behavioral data and cohort data."""
    
    colors = generate_colors(len(behavior_files))  # Generate colors based on number of mice
    
    all_results = []
    cohort_data = []
    
    # Load cohort data first - we expect only one cohort file
    if cohort_files:
        cohort_df = load_cohort_data(cohort_files[0])
        cohort_data.append(cohort_df)
        
        # Get unique combinations of ID and Sex from the cohort dataframe
        unique_mice = pd.read_csv(cohort_files[0])[['ID', 'Sex']].drop_duplicates().copy()
        unique_mice['ID'] = unique_mice['ID'].astype(str).str.strip()
        unique_mice['Sex'] = unique_mice['Sex'].astype(str).str.strip()
        sex_map = unique_mice.set_index('ID')['Sex'].to_dict()
        
        # print("\nSex mapping:")
        # for mouse_id, sex in sex_map.items():
        #     print(f"Mouse {mouse_id}: Sex = {sex}")
        
        # Create markers based on sex ('s' for Male/M, 'o' for Female/F)
        markers = {}
        # print("\nCreating markers for each mouse:")
        for mouse_id, sex in sex_map.items():
            #print(f"Processing {mouse_id}: Original sex value = '{sex}'")
            sex_upper = sex.upper()
            #print(f"  Uppercase value = '{sex_upper}'")
            is_male = sex_upper.startswith('M')
            #print(f"  Starts with M? {is_male}")
            markers[mouse_id] = 's' if is_male else 'o'
            #print(f"  Setting {mouse_id} as {'male' if is_male else 'female'} with marker {'square' if is_male else 'circle'}")
    
    # Combine all cohort data
    if cohort_data:
        combined_cohort_data = pd.concat(cohort_data, ignore_index=True)
    else:
        combined_cohort_data = None
    
    # Process behavioral data
    for idx, data_file in enumerate(behavior_files):
        # Read the combined data file
        df = pd.read_csv(data_file, index_col='timestamp')

        #print(f"Reading data from: {data_file}")

        # Initialize lists to store results
        dates = []
        reward_counts = []
        avg_speeds = []
        lick_counts = []
        
        # Process each date's data
        for timestamp, row in df.iterrows():
            try:
                # Read trial log file for rewards
                trial_log = pd.read_csv(row['trial_log'])
                
                # Read treadmill file for speed data
                treadmill_data = pd.read_csv(row['treadmill'])
                # Calculate average speed (excluding NaN values)
                avg_speed = treadmill_data['speed'].mean()
                
                # Read capacitive data and perform z-score normalization
                capacitive_data = pd.read_csv(row['capacitive'])
                capacitive_values = capacitive_data['capacitive_value']
                
                # Calculate z-score: z = (x - μ) / σ
                z_scores = (capacitive_values - capacitive_values.mean()) / capacitive_values.std()
                
                # Binarize the data based on z-score threshold of 3
                binary_data = (z_scores > 3).astype(int)
                
                # Count lick bouts (transitions from 0 to 1)
                # Convert to numpy array for easier manipulation
                binary_array = binary_data.values
                # Find where values change (True where value changes, False where it stays the same)
                transitions = np.diff(binary_array, prepend=0)
                # Count only 0->1 transitions (positive changes)
                lick_count = np.sum(transitions == 1)
                
                # Convert Unix timestamp to datetime
                date = datetime.fromtimestamp(int(timestamp))
                
                # Count rewards (length of non-null reward events)
                reward_count = len(trial_log['reward_event'].dropna())
                
                # Store the data
                dates.append(date)
                reward_counts.append(reward_count)
                avg_speeds.append(avg_speed)
                lick_counts.append(lick_count)
                
            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                continue
        
        # Get mouse name from filename
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Create daily metrics DataFrame
        daily_df = pd.DataFrame({
            'day': range(len(dates)),
            'reward_count': reward_counts,
            'avg_speed': avg_speeds,
            'lick_count': lick_counts
        })
        
        # Get cohort data for this mouse if available
        cohort_metrics = None
        if combined_cohort_data is not None:
            #print(f"\nChecking cohort data alignment for mouse {mouse_name}:")
            cohort_metrics = combined_cohort_data[combined_cohort_data['ID'] == mouse_name]
        
        # Store results for this mouse
        all_results.append({
            'mouse': mouse_name,
            'dates': dates,
            'daily_metrics': daily_df,
            'cohort_metrics': cohort_metrics,
            'color': colors[idx],
            'marker': markers[mouse_name]
        })
        
        print(f"Processed {len(dates)} days of data for mouse {mouse_name}")
    
    return all_results

def identify_outliers(data):
    """Identify outliers using the z-score method (threshold = ±3 standard deviations)."""
    z_scores = (data - data.mean()) / data.std()
    outlier_mask = (z_scores < -3) | (z_scores > 3)
    return outlier_mask

def plot_correlation(valid_data, mouse, corr_coef, p_value):
    """Create a scatter plot of weight loss vs rewards with correlation info."""
    # Set global font parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Create figure and get the figure reference
    fig = plt.figure(figsize=(10, 6))
    
    # Create axis
    ax = fig.add_subplot(111)
    ax.scatter(valid_data['reward_count'], valid_data['Value'], alpha=0.6)
    
    # Add trend line
    z = np.polyfit(valid_data['reward_count'], valid_data['Value'], 1)
    p = np.poly1d(z)
    ax.plot(valid_data['reward_count'], p(valid_data['reward_count']), "r--", alpha=0.8)
    
    ax.set_xlabel('Number of Rewards')
    ax.set_ylabel('Weight Loss (%)')
    ax.set_title(f'Weight Loss vs Rewards for Mouse {mouse}\nr = {corr_coef:.3f}, p = {p_value:.3f}')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()
    
    # Ask user if they want to save the plot
    save_plot = messagebox.askyesno("Save Plot", "Would you like to save this plot?")
    if save_plot:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile=f'correlation_weight_and_rewards_{mouse}.svg',
            title="Save Plot As",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_lick_correlation_subplots(all_results, correlations_list):
    """Create a single figure with lick count vs reward correlation subplots for all mice."""
    # Set global font parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Create a list to store outlier information
    outlier_info = []
    
    # Count mice with valid data for subplot layout
    n_mice = len(all_results)
    
    if n_mice == 0:
        return
    
    # Calculate optimal subplot layout
    if n_mice <= 4:
        n_cols = min(2, n_mice)
    else:
        n_cols = 3  # Use 3 columns for 5 or more mice
    
    n_rows = (n_mice + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(8*n_cols, 6*n_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # First determine global axis limits
    all_rewards = []
    all_licks = []
    for result in all_results:
        all_rewards.extend(result['daily_metrics']['reward_count'])
        all_licks.extend(result['daily_metrics']['lick_count'])
    
    x_min, x_max = min(all_rewards), max(all_rewards)
    y_min, y_max = min(all_licks), max(all_licks)
    # Add 5% padding to the limits
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_limits = [x_min - x_pad, x_max + x_pad]
    y_limits = [y_min - y_pad, y_max + y_pad]
    
    for idx, (result, corr_data) in enumerate(zip(all_results, correlations_list)):
        mouse = result['mouse']
        daily_metrics = result['daily_metrics']
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Identify outliers
        reward_outliers = identify_outliers(daily_metrics['reward_count'])
        lick_outliers = identify_outliers(daily_metrics['lick_count'])
        combined_outliers = reward_outliers | lick_outliers
        
        # Store outlier information
        outlier_days = daily_metrics.loc[combined_outliers, 'day'].tolist()
        outlier_rewards = daily_metrics.loc[combined_outliers, 'reward_count'].tolist()
        outlier_licks = daily_metrics.loc[combined_outliers, 'lick_count'].tolist()
        
        if any(combined_outliers):
            outlier_info.append({
                'mouse': mouse,
                'data_type': 'licks',
                'outlier_days': outlier_days,
                'outlier_rewards': outlier_rewards,
                'outlier_licks': outlier_licks
            })
        
        # Plot scatter points
        scatter = ax.scatter(daily_metrics.loc[~combined_outliers, 'reward_count'], 
                            daily_metrics.loc[~combined_outliers, 'lick_count'], 
                            alpha=0.6, label='Normal data')
        
        # Plot outliers in red
        if any(combined_outliers):
            ax.scatter(daily_metrics.loc[combined_outliers, 'reward_count'],
                      daily_metrics.loc[combined_outliers, 'lick_count'],
                      color='red', alpha=0.6, label='Outliers')
            ax.legend()
        
        # Set consistent axis limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        
        # Add day labels to each point
        for i, txt in enumerate(daily_metrics['day']):
            ax.annotate(f"d{txt + 1}", 
                       (daily_metrics['reward_count'].iloc[i], daily_metrics['lick_count'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        # Calculate correlation and regression line using non-outlier data only
        non_outlier_rewards = daily_metrics.loc[~combined_outliers, 'reward_count']
        non_outlier_licks = daily_metrics.loc[~combined_outliers, 'lick_count']
        corr_coef, p_val = stats.pearsonr(non_outlier_rewards, non_outlier_licks)
        
        # Add trend line using non-outlier data
        z = np.polyfit(non_outlier_rewards, non_outlier_licks, 1)
        p = np.poly1d(z)
        ax.plot(daily_metrics['reward_count'], p(daily_metrics['reward_count']), "r--", alpha=0.8)
        
        # Add labels and title with adjusted font sizes
        ax.set_xlabel('Number of Rewards', fontsize=10)
        ax.set_ylabel('Number of Licks', fontsize=10)
        ax.set_title(f'Mouse {mouse}\nr = {corr_coef:.3f}\np = {p_val:.3f}\n(outliers excluded)',
                    fontsize=11, pad=10)
        
        # Adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
    
    # Ask user if they want to save the plot
    save_plot = messagebox.askyesno("Save Plot", "Would you like to save the lick correlation plot?")
    if save_plot:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile='lick_correlation_plots_all_mice.svg',
            title="Save Lick Correlation Plot As",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_speed_correlation_subplots(all_results, correlations_list):
    """Create a single figure with speed vs reward correlation subplots for all mice."""
    # Set global font parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Create a list to store outlier information
    outlier_info = []
    
    # Count mice with valid data for subplot layout
    n_mice = len(all_results)
    
    if n_mice == 0:
        return
    
    # Calculate optimal subplot layout
    if n_mice <= 4:
        n_cols = min(2, n_mice)
    else:
        n_cols = 3  # Use 3 columns for 5 or more mice
    
    n_rows = (n_mice + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(8*n_cols, 6*n_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # First determine global axis limits
    all_rewards = []
    all_speeds = []
    for result in all_results:
        all_rewards.extend(result['daily_metrics']['reward_count'])
        all_speeds.extend(result['daily_metrics']['avg_speed'])
    
    x_min, x_max = min(all_rewards), max(all_rewards)
    y_min, y_max = min(all_speeds), max(all_speeds)
    # Add 5% padding to the limits
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_limits = [x_min - x_pad, x_max + x_pad]
    y_limits = [y_min - y_pad, y_max + y_pad]
    
    for idx, (result, corr_data) in enumerate(zip(all_results, correlations_list)):
        mouse = result['mouse']
        daily_metrics = result['daily_metrics']
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        
        # Identify outliers
        reward_outliers = identify_outliers(daily_metrics['reward_count'])
        speed_outliers = identify_outliers(daily_metrics['avg_speed'])
        combined_outliers = reward_outliers | speed_outliers
        
        # Store outlier information
        outlier_days = daily_metrics.loc[combined_outliers, 'day'].tolist()
        outlier_rewards = daily_metrics.loc[combined_outliers, 'reward_count'].tolist()
        outlier_speeds = daily_metrics.loc[combined_outliers, 'avg_speed'].tolist()
        
        if any(combined_outliers):
            outlier_info.append({
                'mouse': mouse,
                'data_type': 'speed',
                'outlier_days': outlier_days,
                'outlier_rewards': outlier_rewards,
                'outlier_speeds': outlier_speeds
            })
        
        # Plot scatter points
        scatter = ax.scatter(daily_metrics.loc[~combined_outliers, 'reward_count'], 
                            daily_metrics.loc[~combined_outliers, 'avg_speed'], 
                            alpha=0.6, label='Normal data')
        
        # Plot outliers in red
        if any(combined_outliers):
            ax.scatter(daily_metrics.loc[combined_outliers, 'reward_count'],
                      daily_metrics.loc[combined_outliers, 'avg_speed'],
                      color='red', alpha=0.6, label='Outliers')
            ax.legend()
        
        # Set consistent axis limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        
        # Add day labels to each point
        for i, txt in enumerate(daily_metrics['day']):
            ax.annotate(f"d{txt + 1}", 
                       (daily_metrics['reward_count'].iloc[i], daily_metrics['avg_speed'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        # Calculate correlation and regression line using non-outlier data only
        non_outlier_rewards = daily_metrics.loc[~combined_outliers, 'reward_count']
        non_outlier_speeds = daily_metrics.loc[~combined_outliers, 'avg_speed']
        corr_coef, p_val = stats.pearsonr(non_outlier_rewards, non_outlier_speeds)
        
        # Add trend line using non-outlier data
        z = np.polyfit(non_outlier_rewards, non_outlier_speeds, 1)
        p = np.poly1d(z)
        ax.plot(daily_metrics['reward_count'], p(daily_metrics['reward_count']), "r--", alpha=0.8)
        
        # Add labels and title with adjusted font sizes
        ax.set_xlabel('Number of Rewards', fontsize=10)
        ax.set_ylabel('Average Speed', fontsize=10)
        ax.set_title(f'Mouse {mouse}\nr = {corr_coef:.3f}\np = {p_val:.3f}\n(outliers excluded)',
                    fontsize=11, pad=10)
        
        # Adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
    
    # Ask user if they want to save the plot
    save_plot = messagebox.askyesno("Save Plot", "Would you like to save the speed correlation plot?")
    if save_plot:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile='speed_correlation_plots_all_mice.svg',
            title="Save Speed Correlation Plot As",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_correlation_subplots(all_results, correlations_list):
    """Create a single figure with correlation subplots for all mice."""
    # Set global font parameters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Create a list to store outlier information
    outlier_info = []
    
    # Count mice with valid data for subplot layout
    valid_mice = [result for result in all_results 
                 if result['cohort_metrics'] is not None]
    n_mice = len(valid_mice)
    
    if n_mice == 0:
        return
    
    # Calculate optimal subplot layout
    if n_mice <= 4:
        n_cols = min(2, n_mice)
    else:
        n_cols = 3  # Use 3 columns for 5 or more mice
    
    n_rows = (n_mice + n_cols - 1) // n_cols
    
    # Create figure with subplots
    # Increase the height per row and width per column for better spacing
    fig = plt.figure(figsize=(8*n_cols, 6*n_rows))
    # Increase spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # First determine global axis limits
    all_rewards = []
    all_weights = []
    for result in valid_mice:
        merged_data = pd.merge(
            result['daily_metrics'],
            result['cohort_metrics'],
            left_on='day',
            right_on='Day',
            how='inner'
        )
        valid_data = merged_data.dropna()
        if len(valid_data) >= 2:
            all_rewards.extend(valid_data['reward_count'])
            all_weights.extend(valid_data['Value'])
    
    if all_rewards and all_weights:  # Only if we have valid data
        x_min, x_max = min(all_rewards), max(all_rewards)
        y_min, y_max = min(all_weights), max(all_weights)
        # Add 5% padding to the limits
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        x_limits = [x_min - x_pad, x_max + x_pad]
        y_limits = [y_min - y_pad, y_max + y_pad]
    
    for idx, (result, corr_data) in enumerate(zip(valid_mice, correlations_list)):
        mouse = result['mouse']
        # Get correlation data
        merged_data = pd.merge(
            result['daily_metrics'],
            result['cohort_metrics'],
            left_on='day',
            right_on='Day',
            how='inner'
        )
        valid_data = merged_data.dropna()
        
        if len(valid_data) >= 2:
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            # Identify outliers
            reward_outliers = identify_outliers(valid_data['reward_count'])
            weight_outliers = identify_outliers(valid_data['Value'])
            combined_outliers = reward_outliers | weight_outliers
            
            # Store outlier information
            outlier_days = valid_data.loc[combined_outliers, 'day'].tolist()
            outlier_rewards = valid_data.loc[combined_outliers, 'reward_count'].tolist()
            outlier_weights = valid_data.loc[combined_outliers, 'Value'].tolist()
            
            if any(combined_outliers):
                outlier_info.append({
                    'mouse': mouse,
                    'data_type': 'weight',
                    'outlier_days': outlier_days,
                    'outlier_rewards': outlier_rewards,
                    'outlier_weights': outlier_weights
                })
            
            # Plot scatter points
            scatter = ax.scatter(valid_data.loc[~combined_outliers, 'reward_count'], 
                                valid_data.loc[~combined_outliers, 'Value'], 
                                alpha=0.6, label='Normal data')
            
            # Plot outliers in red
            if any(combined_outliers):
                ax.scatter(valid_data.loc[combined_outliers, 'reward_count'],
                          valid_data.loc[combined_outliers, 'Value'],
                          color='red', alpha=0.6, label='Outliers')
                ax.legend()
            
            # Set consistent axis limits if we have valid data
            if all_rewards and all_weights:
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
            
            # Add day labels to each point
            for i, txt in enumerate(valid_data['day']):
                ax.annotate(f"d{txt + 1}", 
                           (valid_data['reward_count'].iloc[i], valid_data['Value'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
            
            # Calculate correlation and regression line using non-outlier data only
            non_outlier_rewards = valid_data.loc[~combined_outliers, 'reward_count']
            non_outlier_weights = valid_data.loc[~combined_outliers, 'Value']
            corr_coef, p_val = stats.pearsonr(non_outlier_rewards, non_outlier_weights)
            
            # Add trend line using non-outlier data
            z = np.polyfit(non_outlier_rewards, non_outlier_weights, 1)
            p = np.poly1d(z)
            ax.plot(valid_data['reward_count'], p(valid_data['reward_count']), "r--", alpha=0.8)
            
            # Add labels and title with adjusted font sizes
            ax.set_xlabel('Number of Rewards', fontsize=10)
            ax.set_ylabel('Body Weight Change (%)', fontsize=10)
            ax.set_title(f'Mouse {mouse}\nr = {corr_coef:.3f}\np = {p_val:.3f}\n(outliers excluded)',
                        fontsize=11, pad=10)  # Add padding to title
            
            # Adjust tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the complete figure
    plt.show()
    
    # Ask user if they want to save the plot
    save_plot = messagebox.askyesno("Save Plot", "Would you like to save the combined plot?")
    if save_plot:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile='correlation_plots_all_mice.svg',
            title="Save Combined Plot As",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def calculate_correlations(all_results):
    """Calculate correlations between behavioral metrics and cohort data."""
    correlations = []
    
    for result in all_results:
        mouse = result['mouse']
        daily_metrics = result['daily_metrics']
        cohort_metrics = result['cohort_metrics']
        
        if cohort_metrics is not None:
            print(f"\n=== DATA ALIGNMENT CHECK FOR MOUSE {mouse} ===")
            
            print("\nBehavioral data days:", daily_metrics['day'].tolist())
            print("Weight data days:", cohort_metrics['Day'].tolist())
            
            # Show the first few rows of each dataset for verification
            print("\nFirst 5 rows of behavioral data:")
            print(daily_metrics[['day', 'reward_count', 'avg_speed', 'lick_count']].head().to_string())
            
            print("\nFirst 5 rows of weight data:")
            print(cohort_metrics[['Day', 'Value']].head().to_string())
            
            # Match days between behavioral and cohort data
            merged_data = pd.merge(
                daily_metrics,
                cohort_metrics,
                left_on='day',
                right_on='Day',
                how='inner'
            )
            
            print(f"\nNumber of days with both behavioral and weight data: {len(merged_data)}")
            print("Days that have both behavioral and weight data:", merged_data['day'].tolist())
            
            if len(merged_data) > 0:
                print("\nMERGED DATA (showing all matched days):")
                print(merged_data[['day', 'Day', 'reward_count', 'Value', 'avg_speed', 'lick_count']].to_string())
            else:
                print("\nWARNING: No matching days found between behavioral and weight data!")
            
            print("=" * 60)
            
            if not merged_data.empty:
                # print("\nDetailed matched data:")
                # print(merged_data[['day', 'reward_count', 'Value']].to_string())
                # Ensure numeric data types for correlation
                reward_counts = merged_data['reward_count'].astype(float)
                weight_loss = merged_data['Value'].astype(float)
                
                # Check for NaN values
                nan_rewards = reward_counts.isna().sum()
                nan_weights = weight_loss.isna().sum()
                if nan_rewards > 0 or nan_weights > 0:
                    print(f"WARNING: Found {nan_rewards} NaN values in rewards and {nan_weights} NaN values in weight loss")
                
                # print("\nMatched data points:")
                # print(merged_data[['day', 'reward_count', 'Value']].to_string())
                
                # Remove rows with NaN values
                valid_data = merged_data.dropna()
                #print(f"\nNumber of valid data points after removing NaN: {len(valid_data)}")
                
                if len(valid_data) >= 2:  # Need at least 2 points for correlation
                    # Calculate correlations between daily rewards and percent weight loss
                    corr_rewards = stats.pearsonr(valid_data['reward_count'], valid_data['Value'])
                    # Calculate correlations between speed and rewards
                    corr_speed = stats.pearsonr(valid_data['reward_count'], valid_data['avg_speed'])
                    # Calculate correlations between licks and rewards
                    corr_licks = stats.pearsonr(valid_data['reward_count'], valid_data['lick_count'])
                    # print("\nCorrelation based on valid data points:")
                    # print(valid_data[['day', 'reward_count', 'Value', 'avg_speed', 'lick_count']].to_string())
                else:
                    print("\nWARNING: Not enough valid data points for correlation")
                
                correlations.append({
                    'mouse': mouse,
                    'reward_vs_weight_loss_correlation': corr_rewards[0],
                    'reward_vs_weight_loss_p_value': corr_rewards[1],
                    'reward_vs_speed_correlation': corr_speed[0],
                    'reward_vs_speed_p_value': corr_speed[1],
                    'reward_vs_lick_correlation': corr_licks[0],
                    'reward_vs_lick_p_value': corr_licks[1]
                })
    
    return pd.DataFrame(correlations)

def main():
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # First, select behavior data files
    behavior_paths = filedialog.askopenfilenames(
        title='Select mouse behavior data files (treadmill/capacitive/trial logs)',
        filetypes=[('CSV files', '*.csv')],
        initialdir=os.getcwd()
    )
    
    if behavior_paths:
        print("\nSelected behavior files for mice:", 
              ", ".join([os.path.basename(path).split("_")[0] for path in behavior_paths]))

    # Then, select the cohort data file
    cohort_paths = filedialog.askopenfilename(  # Note: Changed to askopenfilename (singular)
        title='Select the cohort data file (must contain ID and Sex columns)',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        initialdir=os.getcwd()
    )
    
    # Convert single path to list for compatibility with rest of code
    cohort_paths = [cohort_paths] if cohort_paths else []
    
    if behavior_paths:
        # Analyze data and get results (markers will be determined by sex in the analyze_mouse_data function)
        all_results = analyze_mouse_data(behavior_paths, cohort_paths)
        
        # Calculate correlations
        correlations_df = calculate_correlations(all_results)
        
        # Print basic information about the data loaded
        for result in all_results:
            mouse = result['mouse']
            dates = result['dates']
            # print(f"\nMouse: {mouse}")
            # print(f"Number of days: {len(dates)}")
            # print(f"Date range: {min(dates)} to {max(dates)}")
            
            # # Print data summary
            # print("\nBehavioral data summary:")
            # print(f"Average daily rewards: {result['daily_metrics']['reward_count'].mean():.2f}")
            # print(f"Max daily rewards: {result['daily_metrics']['reward_count'].max()}")
            # print(f"Min daily rewards: {result['daily_metrics']['reward_count'].min()}")
            # print(f"\nTreadmill speed summary:")
            # print(f"Average speed: {result['daily_metrics']['avg_speed'].mean():.2f}")
            # print(f"Max speed: {result['daily_metrics']['avg_speed'].max():.2f}")
            # print(f"Min speed: {result['daily_metrics']['avg_speed'].min():.2f}")
            # print(f"\nLick count summary:")
            # print(f"Average daily licks: {result['daily_metrics']['lick_count'].mean():.2f}")
            # print(f"Max daily licks: {result['daily_metrics']['lick_count'].max()}")
            # print(f"Min daily licks: {result['daily_metrics']['lick_count'].min()}")
            
            if result['cohort_metrics'] is not None:
                # Get only valid weight measurements (non-NaN)
                valid_weights = result['cohort_metrics']['Value'].dropna()
                # print("\nCohort data (percent weight loss) summary:")
                # print("Number of weight measurements:", len(valid_weights))
                # print(f"Average weight loss %: {valid_weights.mean():.2f}")
                # print(f"Max weight loss %: {valid_weights.max():.2f}")
                # print(f"Min weight loss %: {valid_weights.min():.2f}")
        
        # Create summary dataframes
        summary_data = []
        merged_data_all = []
        
        for result in all_results:
            mouse = result['mouse']
            
            # Basic summary stats
            summary = {
                'Mouse_ID': mouse,
                'Number_of_Days': len(result['dates']),
                'Start_Date': min(result['dates']),
                'End_Date': max(result['dates']),
                'Avg_Daily_Rewards': result['daily_metrics']['reward_count'].mean(),
                'Max_Daily_Rewards': result['daily_metrics']['reward_count'].max(),
                'Min_Daily_Rewards': result['daily_metrics']['reward_count'].min(),
            }
            
            if result['cohort_metrics'] is not None:
                # Count only valid (non-NaN) weight measurements
                valid_weights = result['cohort_metrics']['Value'].dropna()
                summary.update({
                    'Number_of_Weight_Measurements': len(valid_weights),
                    'Avg_Weight_Loss': valid_weights.mean(),
                    'Max_Weight_Loss': valid_weights.max(),
                    'Min_Weight_Loss': valid_weights.min(),
                })
                
                # Get merged data for this mouse
                merged = pd.merge(
                    result['daily_metrics'],
                    result['cohort_metrics'],
                    left_on='day',
                    right_on='Day',
                    how='outer'
                )
                merged['Mouse_ID'] = mouse
                merged_data_all.append(merged)
            
            summary_data.append(summary)
        
        # Create summary DataFrame with sections clearly marked
        combined_data = []
        
        # Add header section
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined_data.append(pd.DataFrame([{
            'Section': 'Analysis Information',
            'Parameter': 'Analysis Date',
            'Value': timestamp
        }]))
        
        # Add summary statistics section
        for summary in summary_data:
            mouse_summary = []
            mouse_summary.append({
                'Section': 'Summary Statistics',
                'Parameter': 'Mouse ID',
                'Value': summary['Mouse_ID']
            })
            for key, value in summary.items():
                if key != 'Mouse_ID':
                    mouse_summary.append({
                        'Section': 'Summary Statistics',
                        'Parameter': key.replace('_', ' '),
                        'Value': value
                    })
            combined_data.append(pd.DataFrame(mouse_summary))
        
        # Add correlation results section
        if len(correlations_df) > 0:
            for _, row in correlations_df.iterrows():
                corr_data = []
                corr_data.append({
                    'Section': 'Correlation Results',
                    'Parameter': 'Mouse ID',
                    'Value': row['mouse']
                })
                corr_data.append({
                    'Section': 'Correlation Results',
                    'Parameter': 'Correlation Coefficient',
                    'Value': row['reward_vs_weight_loss_correlation']
                })
                corr_data.append({
                    'Section': 'Correlation Results',
                    'Parameter': 'P-value',
                    'Value': row['reward_vs_weight_loss_p_value']
                })
                combined_data.append(pd.DataFrame(corr_data))
        
        # Add daily data section
        daily_data = []
        for merged in merged_data_all:
            for _, row in merged.iterrows():
                daily_data.append({
                    'Section': 'Daily Data',
                    'Parameter': f"{row['Mouse_ID']} - Day {row['day']}",
                    'Rewards': row['reward_count'],
                    'Weight_Loss': row['Value']
                })
        combined_data.append(pd.DataFrame(daily_data))
        
        # Combine all sections
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Save to a single CSV file
        os.makedirs("analysis_results", exist_ok=True)
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_results/mouse_analysis_{timestamp_file}.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nSaved complete analysis to: {output_file}")
        
        # Print correlation results
        if len(correlations_df) > 0:
            # print("\nCorrelation Results:")
            # print(correlations_df.to_string(index=False))
            
            # Create weight loss correlation subplots
            create_correlation_subplots(all_results, correlations_df.to_dict('records'))
            
            # Create speed correlation subplots
            create_speed_correlation_subplots(all_results, correlations_df.to_dict('records'))
            
            # Create lick count correlation subplots
            create_lick_correlation_subplots(all_results, correlations_df.to_dict('records'))
            
            # # Print speed correlation results
            # print("\nSpeed vs Reward Correlation Results:")
            # print("Mouse\tCorrelation\tP-value")
            # for _, row in correlations_df.iterrows():
            #     print(f"{row['mouse']}\t{row['reward_vs_speed_correlation']:.3f}\t{row['reward_vs_speed_p_value']:.3f}")
            
            # # Print lick correlation results
            # print("\nLick Count vs Reward Correlation Results:")
            # print("Mouse\tCorrelation\tP-value")
            # for _, row in correlations_df.iterrows():
            #     print(f"{row['mouse']}\t{row['reward_vs_lick_correlation']:.3f}\t{row['reward_vs_lick_p_value']:.3f}")
    else:
        print("No behavior files selected. Exiting...")

if __name__ == "__main__":
    main()