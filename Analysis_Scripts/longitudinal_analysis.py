import pandas as pd
import glob
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import colorsys
import matplotlib.dates as mdates

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

def analyze_mouse_data(data_files, markers):
    fig = plt.figure(figsize=(12, 6))
    colors = generate_colors(len(data_files))  # Generate colors based on number of mice
    
    all_results = []
    
    for idx, data_file in enumerate(data_files):
        # Read the combined data file
        df = pd.read_csv(data_file, index_col='timestamp')
        
        print(f"Reading data from: {data_file}")
        
        # Initialize lists to store results
        dates = []
        speeds = []
        
        # Process each date's data
        for timestamp, row in df.iterrows():
            try:
                # Read the treadmill data from the file path
                treadmill_data = pd.read_csv(row['treadmill'])
                
                # Calculate average speed for this date
                avg_speed = treadmill_data['speed'].mean()
                
                # Convert Unix timestamp to datetime and store results
                date = datetime.fromtimestamp(int(timestamp))
                dates.append(date)
                speeds.append(avg_speed)
                
                #print(f"Processed date {date.strftime('%Y-%m-%d')}: Average speed = {avg_speed:.2f}")
                
            except Exception as e:
                print(f"Error processing date {timestamp}: {str(e)}")
                print("Raw treadmill data:")
                print(row['treadmill'][:500])  # Print first 500 chars
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'average_speed': speeds
        })
        
        # Sort and remove duplicates
        results_df = results_df.drop_duplicates(subset=['date'])
        results_df = results_df.sort_values('date')
        
        # Get mouse name
        mouse_name = os.path.basename(data_file).split("_")[0]
        
        # Store results for this mouse
        all_results.append({
            'mouse': mouse_name,
            'dates': dates,
            'speeds': speeds,
            'df': results_df
        })
        
        # Plot this mouse's data with sequential day numbers and specified marker
        day_numbers = np.arange(1, len(results_df) + 1)
        mouse_name = os.path.basename(data_file).split("_")[0]
        plt.plot(day_numbers, results_df['average_speed'], 
            f'{markers[mouse_name]}-', color=colors[idx], markersize=8, label=mouse_name)
    
    return fig, all_results

if __name__ == "__main__":
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select multiple data files
    file_paths = filedialog.askopenfilenames(
        title='Select mouse data files',
        filetypes=[('CSV files', '*.csv')],
        initialdir=os.getcwd()  # Start in current directory
    )
    
    if file_paths:
        # Create a dictionary to store marker choices
        markers = {}
        
        # Ask for marker type for each mouse
        for file_path in file_paths:
            mouse_name = os.path.basename(file_path).split("_")[0]
            while True:
                choice = input(f"Enter marker type for {mouse_name} (s for square, o for circle): ").lower().strip()
                if choice in ['s', 'o']:
                    markers[mouse_name] = choice
                    break
                else:
                    print("Invalid choice. Please enter 's' for square or 'o' for circle.")
        
        # Run the analysis with marker choices
        fig, results = analyze_mouse_data(file_paths, markers)

        # Add title and labels to the plot
        plt.title(f'Average Speed Over Time - {len(file_paths)} Mice')
        plt.xlabel('Day')
        plt.ylabel('Average Speed')
        plt.grid(True)

        # Adjust legend based on number of mice
        if len(file_paths) > 10:
            # For many mice, place legend outside the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.85)  # Make room for legend
        else:
            # For fewer mice, keep legend inside
            plt.legend()


        # Set tick marks to face inward
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set minimum for y-axis to -10
        ax.set_ylim(bottom=-10)

        # Set x-axis to start at 0
        ax.set_xlim(left=0)

        # Set x-axis major ticks to every 5 days
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        # Adjust layout to prevent label cutoff and ensure even spacing
        plt.tight_layout()

        # Display the plot first
        plt.show()

        # Ask if user wants to save the plot
        save = input("Would you like to save the plot? (yes/no): ").lower().strip()
        if save.startswith('y'):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
                title="Save plot as",
                initialfile=f"mouse_speed_comparison_{len(file_paths)}mice.svg"
            )
            if save_path:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['Arial']
                plt.rcParams['svg.fonttype'] = 'none'
                fig.savefig(save_path, bbox_inches='tight', format='svg')
                print(f"Plot saved to: {save_path}")
    else:
        print("No file selected. Exiting...")