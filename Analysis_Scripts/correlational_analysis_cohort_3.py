import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
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

def load_cohort_data(file_path):
    """Load and process cohort-style CSV data."""
    df = pd.read_csv(file_path)
    
    # Validate that this is actually a cohort file
    required_columns = ['ID', 'Sex']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"This does not appear to be a cohort file. Missing required columns: {', '.join(missing_columns)}")
    
    print(f"\nLoading cohort data from: {os.path.basename(file_path)}")
    print(f"Available mouse IDs in cohort file: {', '.join(df['ID'].tolist())}")
    
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
    
    # Adjust day numbers to align with behavioral data
    # Day 19 in cohort file = Day 0 in behavioral data
    melted_df['Day'] = melted_df['Day'] - 19
    
    # Sort by ID and Day
    melted_df = melted_df.sort_values(['ID', 'Day'])
    
    # Print day range for verification
    print(f"\nCohort data day range after alignment: {melted_df['Day'].min()} to {melted_df['Day'].max()}")
    print("Note: Day 0 corresponds to the first day in the behavioral data")
    
    return melted_df

def analyze_mouse_data(behavior_files, cohort_files):
    """Analyze both behavioral data and cohort data."""
    print("\nAnalyzing data with the following alignment:")
    print("- Behavioral data starts at day 0")
    print("- Cohort data has been shifted (original day 19 = new day 0)")
    
    colors = generate_colors(len(behavior_files))  # Generate colors based on number of mice
    
    all_results = []
    cohort_data = []
    
    # Load cohort data first - we expect only one cohort file
    if cohort_files:
        cohort_df = load_cohort_data(cohort_files[0])
        cohort_data.append(cohort_df)
        
        # Create a dictionary to map mouse IDs to their sex
        sex_map = cohort_df.groupby('ID')['Sex'].first().to_dict()
        # Create markers based on sex ('s' for Male, 'o' for Female)
        markers = {mouse_id: 's' if sex == 'M' else 'o' for mouse_id, sex in sex_map.items()}
    
    # Combine all cohort data
    if cohort_data:
        combined_cohort_data = pd.concat(cohort_data, ignore_index=True)
    else:
        combined_cohort_data = None
    
    # Process behavioral data
    for idx, data_file in enumerate(behavior_files):
        # Read the combined data file
        df = pd.read_csv(data_file, index_col='timestamp')
        
        print(f"Reading data from: {data_file}")
        
        # Initialize lists to store results
        dates = []
        reward_counts = []
        
        # Process each date's data
        for timestamp, row in df.iterrows():
            try:
                # Read only the trial log file
                trial_log = pd.read_csv(row['trial_log'])
                
                # Convert Unix timestamp to datetime
                date = datetime.fromtimestamp(int(timestamp))
                
                # Count rewards (length of non-null reward events)
                reward_count = len(trial_log['reward_event'].dropna())
                
                # Store the data
                dates.append(date)
                reward_counts.append(reward_count)
                
            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                continue
        
        # Get mouse name from filename
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Create daily metrics DataFrame
        daily_df = pd.DataFrame({
            'day': range(len(dates)),
            'reward_count': reward_counts
        })
        
        # Get cohort data for this mouse if available
        cohort_metrics = None
        if combined_cohort_data is not None:
            print(f"\nChecking cohort data alignment for mouse {mouse_name}:")
            cohort_metrics = combined_cohort_data[combined_cohort_data['ID'] == mouse_name]
            if len(cohort_metrics) > 0:
                print(f"Found matching cohort data: {len(cohort_metrics)} measurements")
                print(f"Cohort data day range: {cohort_metrics['Day'].min()} to {cohort_metrics['Day'].max()}")
            else:
                print("WARNING: No matching cohort data found for this mouse!")
        
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

def calculate_correlations(all_results):
    """Calculate correlations between behavioral metrics and cohort data."""
    correlations = []
    
    for result in all_results:
        mouse = result['mouse']
        daily_metrics = result['daily_metrics']
        cohort_metrics = result['cohort_metrics']
        
        if cohort_metrics is not None:
            print(f"\nAnalyzing correlations for mouse {mouse}:")
            
            # Match days between behavioral and cohort data
            merged_data = pd.merge(
                daily_metrics,
                cohort_metrics,
                left_on='day',
                right_on='Day',
                how='inner'
            )
            
            print(f"Number of days with both behavioral and weight data: {len(merged_data)}")
            
            if not merged_data.empty:
                # Ensure numeric data types for correlation
                reward_counts = merged_data['reward_count'].astype(float)
                weight_loss = merged_data['Value'].astype(float)
                
                # Check for NaN values
                nan_rewards = reward_counts.isna().sum()
                nan_weights = weight_loss.isna().sum()
                if nan_rewards > 0 or nan_weights > 0:
                    print(f"WARNING: Found {nan_rewards} NaN values in rewards and {nan_weights} NaN values in weight loss")
                
                print("\nMatched data points:")
                print(merged_data[['day', 'reward_count', 'Value']].to_string())
                
                # Remove rows with NaN values
                valid_data = merged_data.dropna()
                #print(f"\nNumber of valid data points after removing NaN: {len(valid_data)}")
                
                if len(valid_data) >= 2:  # Need at least 2 points for correlation
                    # Calculate correlations between daily rewards and percent weight loss
                    corr_rewards = stats.pearsonr(valid_data['reward_count'], valid_data['Value'])
                    print("\nCorrelation based on valid data points:")
                    print(valid_data[['day', 'reward_count', 'Value']].to_string())
                else:
                    print("\nWARNING: Not enough valid data points for correlation")
                
                correlations.append({
                    'mouse': mouse,
                    'reward_vs_weight_loss_correlation': corr_rewards[0],
                    'reward_vs_weight_loss_p_value': corr_rewards[1]
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
            print(f"\nMouse: {mouse}")
            print(f"Number of days: {len(dates)}")
            print(f"Date range: {min(dates)} to {max(dates)}")
            
            # Print data summary
            print("\nBehavioral data summary:")
            print(f"Average daily rewards: {result['daily_metrics']['reward_count'].mean():.2f}")
            print(f"Max daily rewards: {result['daily_metrics']['reward_count'].max()}")
            print(f"Min daily rewards: {result['daily_metrics']['reward_count'].min()}")
            
            if result['cohort_metrics'] is not None:
                print("\nCohort data (percent weight loss) summary:")
                print("Number of weight measurements:", len(result['cohort_metrics']))
                print(f"Average weight loss %: {result['cohort_metrics']['Value'].mean():.2f}")
                print(f"Max weight loss %: {result['cohort_metrics']['Value'].max():.2f}")
                print(f"Min weight loss %: {result['cohort_metrics']['Value'].min():.2f}")
        
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
            print("\nCorrelation Results:")
            print(correlations_df.to_string(index=False))
    else:
        print("No behavior files selected. Exiting...")

if __name__ == "__main__":
    main()
