import pandas as pd
import os

def add_level_column(mouse_id):
    # Set paths
    base_path = r'c:\Users\Brenna\OneDrive - The Pennsylvania State University\Desktop\KaufmanProject\MousePortal'
    virtual_foraging_path = r'F:\virtual_foraging_cohort_3'
    
    # Read the log and data CSV files
    log_file = os.path.join(virtual_foraging_path, mouse_id, f'{mouse_id}_log.csv')
    data_file = os.path.join(base_path, f'{mouse_id}_data.csv')
    
    # Read the CSV files
    log_df = pd.read_csv(log_file)
    data_df = pd.read_csv(data_file)
    
    # Create a dictionary mapping dates to the FIRST level file for each date
    # Group by date and take the first occurrence (earliest time for that date)
    log_df_sorted = log_df.sort_values('Time')  # Sort by time to ensure chronological order
    first_levels_per_date = log_df_sorted.groupby('Date').first()
    date_to_level = dict(zip(first_levels_per_date.index, first_levels_per_date['Level']))
    
    # Convert timestamp to date in data_df
    data_df['date'] = pd.to_datetime(data_df['date'], unit='s').dt.strftime('%Y-%m-%d')
    
    # Create the level column by mapping dates
    level_column = data_df['date'].map(date_to_level)
    
    # Insert the level column after the date column
    data_df.insert(2, 'level', level_column)
    
    # Save the updated dataframe
    data_df.to_csv(data_file, index=False)

if __name__ == '__main__':
    # You can modify this to handle multiple mouse IDs if needed
    add_level_column('VF8')