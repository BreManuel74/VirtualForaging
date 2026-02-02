import pandas as pd
import os
import glob

def add_level_column(mouse_id, base_path, log_base_path):
    # Set file paths
    log_file = os.path.join(log_base_path, f'{mouse_id}_log.csv')
    data_file = os.path.join(base_path, f'{mouse_id}_data.csv')
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"  Warning: Log file not found: {log_file}")
        return False
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"  Warning: Data file not found: {data_file}")
        return False
    
    # Read the CSV files
    log_df = pd.read_csv(log_file)
    data_df = pd.read_csv(data_file)
    
    # Create a dictionary mapping dates to the FIRST level file for each date
    # Group by date and take the first occurrence (earliest time for that date)
    log_df_sorted = log_df.sort_values('Time')  # Sort by time to ensure chronological order
    first_levels_per_date = log_df_sorted.groupby('Date').first()
    date_to_level = dict(zip(first_levels_per_date.index, first_levels_per_date['Level']))
    
    # Convert timestamp to date in data_df
    data_df['date'] = pd.to_datetime(data_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    
    # Create the level column by mapping dates
    level_column = data_df['date'].map(date_to_level)
    
    # Insert the level column after the date column
    data_df.insert(2, 'level', level_column)
    
    # Save the updated dataframe
    data_df.to_csv(data_file, index=False)
    print(f"  Added level column to {mouse_id}_data.csv")
    return True

def batch_process_all_animals():
    """Process all animal data files in the base path."""
    print("Add Level Column - Batch Processing Mode")
    print("=" * 60)
    
    # Set paths
    base_path = r'c:\Users\Brenna\OneDrive - The Pennsylvania State University\Desktop\KaufmanProject\MousePortal'
    log_base_path = r'C:\Users\Brenna\OneDrive - The Pennsylvania State University\Desktop\KaufmanProject\MousePortal\Progress_Reports'
    
    print(f"Base path: {base_path}")
    print(f"Log base path: {log_base_path}")
    
    # Find all *_data.csv files in the base path
    data_files = glob.glob(os.path.join(base_path, '*_data.csv'))
    
    if not data_files:
        print("\nNo *_data.csv files found in the base path.")
        return
    
    print(f"\nFound {len(data_files)} data file(s)")
    print("-" * 60)
    
    # Process each file
    successful = 0
    failed = 0
    
    for data_file in data_files:
        filename = os.path.basename(data_file)
        # Extract mouse ID from filename (e.g., "CAH1_data.csv" -> "CAH1")
        mouse_id = filename.replace('_data.csv', '')
        
        print(f"\nProcessing {mouse_id}...")
        
        try:
            if add_level_column(mouse_id, base_path, log_base_path):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 60)

if __name__ == '__main__':
    batch_process_all_animals()