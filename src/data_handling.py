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
    Saves curve of fidelity into numpy format in the desired path
    :fidelity_data: (nparray(float)) Data to save
    :filepath: (str) Path to save
    '''
    # Ensure filepath has .npy extension
    if not filepath.endswith('.npy'):
        filepath = filepath + '.npy'
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if file already exists
    if os.path.exists(filepath):
        raise FileExistsError(f"File '{filepath}' already exists. Operation aborted to prevent overwriting.")
    
    # Save the data
    np.save(filepath, fidelity_data)
    print(f"Data successfully saved to: {filepath}")
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



def save_three_arrays(array1, array2, array3, filepath, description=None):
    '''
    Saves three numpy arrays together into a single .npz file
    
    Parameters:
    -----------
    array1 : np.array
        First array to save
    array2 : np.array
        Second array to save  
    array3 : np.array
        Third array to save
    filepath : str
        Path to save the file
    description : str, optional
        Optional description of the data being saved
    '''
    # Ensure filepath has .npz extension
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if file already exists
    if os.path.exists(filepath):
        raise FileExistsError(f"File '{filepath}' already exists. Operation aborted to prevent overwriting.")
    
    # Save all three arrays together
    np.savez(filepath, array1=array1, array2=array2, array3=array3)
    
    # Optional: save metadata if description provided
    if description:
        metadata_file = filepath.replace('.npz', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Description: {description}\n")
            f.write(f"Array1 shape: {array1.shape}, dtype: {array1.dtype}\n")
            f.write(f"Array2 shape: {array2.shape}, dtype: {array2.dtype}\n") 
            f.write(f"Array3 shape: {array3.shape}, dtype: {array3.dtype}\n")
            f.write(f"Saved on: {np.datetime64('now')}\n")
    
    print(f"Three arrays successfully saved to: {filepath}")
    if description:
        print(f"Metadata saved to: {metadata_file}")
    return


def fetch_three_arrays(filepath):
    '''
    Retrieves three previously stored arrays from a .npz file
    
    Parameters:
    -----------
    filepath : str
        Path to the .npz file (with or without extension)
    
    Returns:
    --------
    tuple : (array1, array2, array3) - The three loaded arrays
    '''
    # Add .npz extension if not present
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")
    
    # Load the data
    data = np.load(filepath)
    
    # Extract the three arrays
    array1 = data['array1']
    array2 = data['array2'] 
    array3 = data['array3']
    
    print(f"Successfully loaded three arrays from: {filepath}")
    print(f"Array shapes: {array1.shape}, {array2.shape}, {array3.shape}")
    
    return array1, array2, array3



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
                        base_name: str = "data",
                        extension = None) -> str:
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
    if extension is None:
        filename = f"{base_name}_N{N_str}_J{J}_L{L_str}_{dict_str}_{timestamp}"
    else:
        filename = f"{base_name}_N{N_str}_J{J}_L{L_str}_{dict_str}_{timestamp}_{extension}"
    
    return filename


