import pandas as pd
import os

def add_level_column(mouse_id):
    # Set paths
    base_path = r'c:\Users\Brenna\OneDrive - The Pennsylvania State University\Desktop\KaufmanProject\MousePortal'
    virtual_foraging_path = r'e:\virtual_foraging_cohort_2'
    
    # Read the log and data CSV files
    log_file = os.path.join(virtual_foraging_path, mouse_id, f'{mouse_id}_log.csv')
    data_file = os.path.join(base_path, f'{mouse_id}_data.csv')
    
    # Read the CSV files
    log_df = pd.read_csv(log_file)
    data_df = pd.read_csv(data_file)
    
    # Create a dictionary mapping dates to level files
    date_to_level = dict(zip(log_df['Date'], log_df['Level File']))
    
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
    add_level_column('BM22')