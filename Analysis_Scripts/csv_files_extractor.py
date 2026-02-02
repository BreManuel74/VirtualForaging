import os
import tkinter as tk
from tkinter import filedialog
import shutil

def select_directory():
    """Open a tkinter dialog to select a root directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    directory = filedialog.askdirectory(
        title="Select Root Directory to Extract CSV Files"
    )
    
    root.destroy()
    return directory

def extract_csv_files(root_dir, output_dir=None):
    """Extract all CSV files from the selected directory and its subdirectories."""
    if not root_dir:
        print("No directory selected.")
        return
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(root_dir, "all_csv_files")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_count = 0
    
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                source_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(output_dir, filename)
                
                # Handle duplicate filenames
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                        counter += 1
                
                shutil.copy2(source_path, dest_path)
                csv_count += 1
                print(f"Copied: {filename} -> {dest_path}")
    
    print(f"\nTotal CSV files extracted: {csv_count}")
    print(f"Output directory: {output_dir}")

def main():
    """Main function to run the CSV extractor."""
    print("CSV Files Extractor")
    print("-" * 50)
    
    # Select root directory using tkinter
    root_dir = select_directory()
    
    if root_dir:
        print(f"Selected directory: {root_dir}\n")
        extract_csv_files(root_dir)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
