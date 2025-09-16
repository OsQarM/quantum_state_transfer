#Saving data
import pandas as pd
import json
import os
import numpy as np
from typing import Dict, Any, Union
import datetime
import re

def save_numpy_array(fidelity_data, filepath):
    '''
    Saves curve of fidelity into numpy format in the desided path
    :fidelity_data:(nparray(float)) Data to save
    :filepath:(str) Path to save
    '''
    np.save(filepath, fidelity_data)
    return


def fetch_numpy_array(filepath):
    '''
    Retrieves previously stored data

    Args:
        filepath: String value
    
    Returns:
        loaded_array: data from the path as numpy array
    '''
    loaded_array = np.load(filepath)
    return loaded_array



def create_data_filename(N: int, 
                        J: str, 
                        L: float, 
                        data_dict: Dict[str, Any],
                        base_name: str = "data",
                        extension: str = "json") -> str:
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
    filename = f"{base_name}_N{N_str}_J{J}_L{L_str}_{dict_str}_{timestamp}.{extension}"
    
    return filename


