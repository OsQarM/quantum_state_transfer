import qutip as qt
import numpy as np

import Hamiltonian as Ham

def apply_gaussian_rel_error(base_list, standard_deviation):
    """
    Applies gaussian relative error to list of floats centered around the value of each element
    and with a standard deviation that is calculated as a percentage of the mean (relative error)

    Args:
        base_list: List of floats
        standard_deviation: usually a fraction, < 1

    Returns:
        error_list: List of input values with added gaussian errors

    """
    error_list = []
    for value in base_list:
        error_list.append(value + np.random.normal(0, abs(standard_deviation*value) , 1)) #Abs to avoid negative std

    return error_list


def apply_gaussian_abs_error(base_list, standard_deviation):
    """
    Applies gaussian absolute error to list of floats centered around the value of each element
    and with a standard deviation that is given as input

    Args:
        base_list: List of floats
        standard_deviation: float

    Returns:
        error_list: List of input values with added gaussian errors

    """
    error_list = []
    for value in base_list:
        error_list.append(value + np.random.normal(0, abs(standard_deviation) , 1))

    return error_list

