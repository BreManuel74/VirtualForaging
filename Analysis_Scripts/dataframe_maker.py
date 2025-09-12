import pandas as pd
import glob
import os
import argparse

def create_dataframe(mouse_id=None, save_csv=True, output_file=None):
    # Define the root directory where the BM* folders are located
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Kaufman_Project')
    
    # Get CSV files based on mouse_id
    if mouse_id:
        all_files = glob.glob(os.path.join(root_dir, f'BM{mouse_id}', '*.csv'))
        if not all_files:
            raise ValueError(f"No CSV files found for BM{mouse_id}")
    else:
        all_files = glob.glob(os.path.join(root_dir, 'BM*', '*.csv'))

    # Create a dictionary to store data for each date
    data_by_date = {}

    # First, group files by their approximate timestamp
    file_groups = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        timestamp = filename.split('capacitive.csv')[0].split('treadmill.csv')[0].split('trial_log.csv')[0]
        
        if timestamp:
            base_time = int(timestamp)  # Convert to integer for comparison
            # Look for existing groups within 60 seconds
            found_group = None
            for existing_time in file_groups:
                if abs(existing_time - base_time) <= 60:  # Increased window to 60 seconds
                    found_group = existing_time
                    break
            
            group_time = found_group if found_group else base_time
            if group_time not in file_groups:
                file_groups[group_time] = []
            file_groups[group_time].append(file_path)
    
    # Now process each group of files
    for base_time, files in file_groups.items():
        # Check if we have all three required file types before processing
        has_capacitive = any('capacitive.csv' in os.path.basename(f) for f in files)
        has_treadmill = any('treadmill.csv' in os.path.basename(f) for f in files)
        has_trial_log = any('trial_log.csv' in os.path.basename(f) for f in files)
        
        if has_capacitive and has_treadmill and has_trial_log:
            timestamp = str(base_time)
            if timestamp not in data_by_date:
                data_by_date[timestamp] = {'date': timestamp}
            
            # Process each file in the group
            for file_path in files:
                filename = os.path.basename(file_path)
                if 'capacitive.csv' in filename:
                    data_by_date[timestamp]['capacitive'] = file_path
                elif 'treadmill.csv' in filename:
                    data_by_date[timestamp]['treadmill'] = file_path
                elif 'trial_log.csv' in filename:
                    data_by_date[timestamp]['trial_log'] = file_path    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data_by_date, orient='index')

    # Sort the DataFrame by date
    df = df.sort_index()

    # Print all unique dates found
    print("\nDates found:")
    for date in sorted(data_by_date.keys()):
        print(f"  {date}")
    print(f"\nTotal dates: {len(df)}")
    
    # Save DataFrame to CSV if requested
    if save_csv:
        if output_file is None and mouse_id:
            output_file = f"BM{mouse_id}_data.csv"
        elif output_file is None:
            output_file = "all_mice_data.csv"
            
        # Save the DataFrame index as a column named 'timestamp'
        df_to_save = df.copy()
        df_to_save.index.name = 'timestamp'
        df_to_save.to_csv(output_file)
        print(f"\nSaved DataFrame to {output_file}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a DataFrame from CSV files for a specific mouse')
    parser.add_argument('--mouse', type=str, help='Mouse ID (e.g., "12" for BM12)')
    parser.add_argument('--output', type=str, help='Output CSV file name (optional)')
    parser.add_argument('--no-save', action='store_true', help='Do not save the DataFrame to CSV')
    args = parser.parse_args()
    
    df = create_dataframe(args.mouse, save_csv=not args.no_save, output_file=args.output)

