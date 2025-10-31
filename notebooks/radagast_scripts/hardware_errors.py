import qutip as qt
import numpy as np

# Tracking info of simulations 
import tqdm
import time
import warnings

#Saving data
import pandas as pd
import json
import os

import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib.gridspec import GridSpec

import sys
sys.path.append('/Users/omichel/Desktop/qilimanjaro/projects/quantum_transport/quantum_state_transfer/src')

import model_building as md
import Hamiltonian as Ham
import dynamics as dyn
import data_handling as dh
import plots as plots

############################## HELPER FUNCTIONS

def initialize_system(state_dictionary, N):
    initial_state = md.create_domain_wall_state(state_dictionary, register='Alice', one_step=True )
    final_state   = md.create_domain_wall_state(state_dictionary, register='Bob'  , one_step=True)

    initial_chain = md.initialize_general_system(N, initial_state, register='Alice')
    final_chain   = md.initialize_general_system(N, final_state, register='Bob')

    register_size = len(initial_state.dims[0])
    
    return initial_chain, final_chain, register_size

def build_hamiltonians(N, lmd, J, reg_size):

    H_transport = Ham.Hamiltonian(system_size = N,
                        mode = "transport",
                        lambda_factor = lmd,
                        global_J = J
                        )
    H_reset     = Ham.Hamiltonian(system_size = N,
                        mode = "reset",
                        lambda_factor = lmd,
                        register_size = reg_size,
                        global_J = J
                        )
    
    return H_transport, H_reset


def calculate_logs(x_data, y_data):
    log_axis = [np.log(i) for i in x_data]
    log_error = [np.log(1-i) for i in y_data]
    return log_axis, log_error

def calculate_logs_2(x_data, y_data):
    log_axis = [np.log(i) for i in x_data]
    log_error = [np.log(i) for i in y_data]
    return log_axis, log_error

##############################

############################## SPECIFIC FUNCTIONS


def select_error(err, error_type):
    if error_type == "j":
        return err, 0, 0
    elif error_type == "l":
        return 0, err, 0
    elif error_type == "z":
         return 0, 0, err
    else:
        raise ValueError(f"{error_type} is not a valid type of error, only \'j\', \'l\', and \'z\' allowed")

def error_loop(N, lmd, J, ti, tf, Nsteps, Nshots, error_list, error_type, register_size = None):
    fidelity_means = np.array([])
    fidelity_errors = np.array([])
    for err in error_list:
        fidelities = []
        for _ in range(Nshots):

            j_err, l_err, z_err = select_error(err, error_type)

            #Run simulations, add fidelities, and save average of n_shots
            H_t = Ham.Hamiltonian(system_size = N,
                        mode = "transport",
                        lambda_factor = lmd,
                        global_J = J,
                        j_error = j_err,
                        l_error = l_err,
                        z_error = z_err
                        )
            
            fidelities.append(dyn.LightweightAlgorithm(initial_system, final_system, ti, tf, Nsteps, H_t))
            

        fmean, ferror = calculate_result_statistics(fidelities)
        fidelity_means = np.append(fidelity_means, fmean)
        fidelity_errors = np.append(fidelity_errors, ferror)

    return fidelity_means, fidelity_errors


def calculate_result_statistics(fidelity):
    fidelity_mean = np.mean(fidelity)
    fidelity_error = np.std(fidelity, ddof=1) / np.sqrt(len(fidelity))

    return fidelity_mean, fidelity_error

##############################


############################## MAIN PROGRAM


N=8
lmd = 0.02272
J = 0.5
state_dictionary = {"1":1}

ti = 0
tf = np.pi/lmd
Nsteps = 500

l_errors = np.linspace(0.01, 0.1, 10)
j_errors = np.linspace(0.001, 0.01, 10)
Nshots = 10

initial_system, final_system, register_size = initialize_system(state_dictionary, N)

#This can be commented if only interested in plotting

fidelities, fidelity_errors = error_loop(N, lmd, J, ti, tf, Nsteps, Nshots, j_errors, "j", register_size)


# Save three arrays together
dh.save_three_arrays(j_errors, fidelities, fidelity_errors,
                 '../../files/errors/j_err_N8_10_points_10_shots',
                 description='Z errors vs fidelity data for experiment 1')

#Comment up to here

loaded_x, loaded_means, loaded_errors = dh.fetch_three_arrays('../../files/errors/j_err_N8_10_points_10_shots')

plots.plot_fidelity_vs_error(loaded_x, loaded_means, loaded_errors,
                       title='Fidelity vs. $\lambda$ error',
                       xlabel='Relative error (%)',
                       save_figure=True,
                       save_path='../../files/figures/errors',
                       file_tag='j_err_N8_10_points_10_shots')