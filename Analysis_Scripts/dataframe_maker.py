import pandas as pd
import glob
import os
import argparse
from tkinter import filedialog
import tkinter as tk

def create_dataframe(mouse_id=None, save_csv=True, output_file=None, selected_folder=None):
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Define the default root directory where the BM* folders are located
    default_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Citric_Acid_Project')
    
    # Use selected folder if provided, otherwise use file dialog
    if selected_folder:
        root_dir = selected_folder
    else:
        # Browse for the folder containing your data files
        root_dir = filedialog.askdirectory(
            title="Select folder containing behavioral data files (VF* or BM* folders)",
            initialdir=default_root_dir
        )
        
        if not root_dir:
            print("No folder selected. Exiting...")
            return None
    
    print(f"Selected folder: {root_dir}")
    
    # Get CSV files based on mouse_id
    if mouse_id:
        # Check if the selected folder itself contains the files, or if we need to look in subdirectories
        direct_files = glob.glob(os.path.join(root_dir, '*.csv'))
        subfolder_files = glob.glob(os.path.join(root_dir, f'CAH{mouse_id}', '*.csv'))
        
        if direct_files:
            all_files = direct_files
        elif subfolder_files:
            all_files = subfolder_files
        else:
            raise ValueError(f"No CSV files found for CAH{mouse_id} in selected folder")
    else:
        # Look for files in CAH* subfolders or directly in the selected folder
        subfolder_files = glob.glob(os.path.join(root_dir, 'CAH*', '*.csv'))
        direct_files = glob.glob(os.path.join(root_dir, '*.csv'))
        
        if subfolder_files:
            all_files = subfolder_files
        elif direct_files:
            all_files = direct_files
        else:
            all_files = []
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
        
        if has_capacitive or has_treadmill or has_trial_log:
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
            output_file = f"VF{mouse_id}_data.csv"
        elif output_file is None:
            output_file = "all_mice_data.csv"
            
        # Save the DataFrame index as a column named 'timestamp'
        df_to_save = df.copy()
        df_to_save.index.name = 'timestamp'
        df_to_save.to_csv(output_file)
        print(f"\nSaved DataFrame to {output_file}")
    
    return df

def find_all_csv_folders(root_dir):
    """Find all 'all_csv_files' folders within the root directory."""
    all_csv_folders = []
    
    print(f"Searching for 'all_csv_files' folders in: {root_dir}")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == 'all_csv_files':
                folder_path = os.path.join(dirpath, dirname)
                # Get the parent folder name (animal ID)
                parent_folder = os.path.basename(dirpath)
                all_csv_folders.append({
                    'path': folder_path,
                    'animal_id': parent_folder
                })
                print(f"  Found: {parent_folder}/all_csv_files")
    
    return all_csv_folders

def process_all_animals():
    """Process all animals by finding all 'all_csv_files' folders."""
    print("DataFrame Maker - Batch Processing Mode")
    print("=" * 60)
    
    # Select root directory using tkinter
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    root_dir = filedialog.askdirectory(
        title="Select Root Directory Containing Animal Folders"
    )
    root.destroy()
    
    if not root_dir:
        print("Operation cancelled.")
        return
    
    print(f"\nSelected directory: {root_dir}")
    
    # Find all 'all_csv_files' folders
    all_csv_folders = find_all_csv_folders(root_dir)
    
    if not all_csv_folders:
        print("\nNo 'all_csv_files' folders found in the selected directory.")
        return
    
    print(f"\nFound {len(all_csv_folders)} 'all_csv_files' folder(s)")
    print("-" * 60)
    
    # Process each folder
    successful = 0
    failed = 0
    
    for folder_info in all_csv_folders:
        animal_id = folder_info['animal_id']
        folder_path = folder_info['path']
        
        print(f"\nProcessing {animal_id}...")
        print(f"Folder: {folder_path}")
        
        try:
            # Get all CSV files in the folder and subdirectories recursively
            all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
            
            print(f"  Found {len(all_files)} CSV files")
            
            if not all_files:
                print(f"  No CSV files found")
                failed += 1
                continue
            
            # Create a dictionary to store data for each date
            data_by_date = {}
            
            # First, group files by their approximate timestamp
            file_groups = {}
            for file_path in all_files:
                filename = os.path.basename(file_path)
                timestamp = filename.split('capacitive.csv')[0].split('treadmill.csv')[0].split('trial_log.csv')[0]
                
                if timestamp:
                    try:
                        base_time = int(timestamp)
                        # Look for existing groups within 60 seconds
                        found_group = None
                        for existing_time in file_groups:
                            if abs(existing_time - base_time) <= 60:
                                found_group = existing_time
                                break
                        
                        group_time = found_group if found_group else base_time
                        if group_time not in file_groups:
                            file_groups[group_time] = []
                        file_groups[group_time].append(file_path)
                    except ValueError:
                        print(f"  Warning: Could not parse timestamp from {filename}")
                        continue
            
            # Now process each group of files
            for base_time, files in file_groups.items():
                # Check if we have all three required file types
                has_capacitive = any('capacitive.csv' in os.path.basename(f) for f in files)
                has_treadmill = any('treadmill.csv' in os.path.basename(f) for f in files)
                has_trial_log = any('trial_log.csv' in os.path.basename(f) for f in files)
                
                if has_capacitive or has_treadmill or has_trial_log:
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
                            data_by_date[timestamp]['trial_log'] = file_path
            
            if not data_by_date:
                print(f"  No complete datasets found")
                failed += 1
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data_by_date, orient='index')
            df = df.sort_index()
            
            print(f"  Found {len(df)} complete datasets")
            
            # Save DataFrame to the script's parent directory (MousePortal root)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.dirname(script_dir)  # Go up to MousePortal root
            output_file = os.path.join(output_dir, f"{animal_id}_data.csv")
            df_to_save = df.copy()
            df_to_save.index.name = 'timestamp'
            df_to_save.to_csv(output_file)
            print(f"  Saved DataFrame to {output_file}")
            
            successful += 1
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a DataFrame from CSV files for a specific mouse')
    parser.add_argument('--mouse', type=str, help='Mouse ID (e.g., "12" for BM12)')
    parser.add_argument('--output', type=str, help='Output CSV file name (optional)')
    parser.add_argument('--no-save', action='store_true', help='Do not save the DataFrame to CSV')
    parser.add_argument('--folder', type=str, help='Folder path (skip dialog if provided)')
    parser.add_argument('--single', action='store_true', help='Process single animal (original mode)')
    args = parser.parse_args()
    
    if args.single:
        df = create_dataframe(args.mouse, save_csv=not args.no_save, output_file=args.output, selected_folder=args.folder)
    else:
        # Default to batch processing mode
        process_all_animals()

