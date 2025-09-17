#Saving data
import pandas as pd
import json
import os
import numpy as np
from typing import Dict, Any, Union
import datetime
import re
import csv

def save_numpy_array(fidelity_data, filepath):
    '''
    Saves curve of fidelity into numpy format in the desided path
    :fidelity_data:(nparray(float)) Data to save
    :filepath:(str) Path to save
    '''
    np.save(f"{filepath}.npy", fidelity_data)
    return


def fetch_numpy_array(filepath):
    '''
    Retrieves previously stored data

    Args:
        filepath: String value
    
    Returns:
        loaded_array: data from the path as numpy array
    '''
    loaded_array = np.load(f"{filepath}.npy")
    return loaded_array


def read_plot_data_from_csv(filepath):
    """Reads x and y data from a CSV file into two separate lists.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        tuple: (x_data, y_data) as two lists
    """
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 2:  # Ensure there are at least two columns
                x_data.append(float(row[0]))
                y_data.append(float(row[1]))

    filename = os.path.basename(filepath)  # Get just the filename with extension
    filename_without_ext = os.path.splitext(filename)[0]  # Remove the extension
    
    return x_data, y_data, filename_without_ext


def save_plot_data_to_csv(x_data, y_data, filepath, headers=None):
    """
    Saves two lists/arrays to a CSV file for plotting.
    
    Args:
        x_data (list/array): X-axis values (e.g., time steps)
        y_data (list/array): Y-axis values (e.g., fidelity)
        filename (str): Output CSV filename
        headers (list): Column headers (e.g., ["Time", "Fidelity"])
    """
    # Ensure inputs are numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Verify equal lengths
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    # Default headers
    if headers is None:
        headers = ["Time_step", "Fidelity"]
    
    # Write to CSV
    with open(f'{filepath}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write header
        writer.writerows(zip(x_data, y_data))  # Write data rows
    
    print(f"Data saved to {filepath}")
    print(f"First 5 rows:\n{headers[0]}\t{headers[1]}")
    for x, y in zip(x_data[:5], y_data[:5]):
        print(f"{x:.4f}\t{y:.4f}")
    
    return



def create_data_filename(N: int, 
                        J: str, 
                        L: float, 
                        data_dict: Dict[str, Any],
                        base_name: str = "data") -> str:
    """
    Create a filename with variables N, J, L, dictionary content, and timestamp.
    
    Args:
        N: Integer or float value
        J: String value
        L: Float value
        data_dict: Dictionary containing parameters/data
        base_name: Base name for the file
        extension: File extension
    
    Returns:
        str: Generated filename with timestamp to avoid overwriting
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a safe string from dictionary (remove special characters)
    def dict_to_safe_string(d: Dict[str, Any], max_items: int = 3) -> str:
        items = []
        for k, v in list(d.items())[:max_items]:
            # Convert value to string and make filesystem-safe
            safe_v = str(v).replace(' ', '_').replace('/', '_')
            safe_v = re.sub(r'[^\w\-_.]', '', safe_v)
            # Truncate very long values
            if len(safe_v) > 20:
                safe_v = safe_v[:20]
            items.append(f"{k}_{safe_v}")
        return "_".join(items)
    
    # Convert dictionary to safe string
    dict_str = dict_to_safe_string(data_dict)
    
    # Format numerical values appropriately
    N_str = f"{N:.0f}" if isinstance(N, float) and N.is_integer() else f"{N:.0f}"
    L_str = f"{L:.3f}"  # Format float to 3 decimal places
    
    # Create filename
    filename = f"{base_name}_N{N_str}_J{J}_L{L_str}_{dict_str}_{timestamp}"
    
    return filename


