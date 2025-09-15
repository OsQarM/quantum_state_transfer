import qutip as qt
import numpy as np

def time_evolution(Ham, initial_state, initial_time, final_time, timesteps):
    '''
    Simulation of transport protocol

    :Ham:(Hamiltonian class object) Hamiltonian under which we evolve the system
    :initial_state:(qt.state) state of the chain at time 0
    :initial_time, final_time:(float) self-explanatory
    :timesteps:(float) number of simulation steps between the two times 
    '''
    hamiltonian_object = Ham.ham
    times = np.linspace(initial_time, final_time, timesteps)
    #apply hamiltonian to initial state and don't track any observables
    options = {
    'method': 'adams', 
    'progress_bar': 'tqdm'
    }
    simulation_results = qt.sesolve(hamiltonian_object, initial_state, times, options = options)

    return simulation_results


def calculate_full_fidelity(state_evolution, target_state): 
    '''
    Calculates fidelity between every simulated time step and a target state that we have
    previously defined

    :state_evolution:(array(np.states)) simulated state dynamics
    :target_state:(np.state) reference state for validation
    '''

    fidelity = np.zeros(len(state_evolution.times))
    for index, state in enumerate(state_evolution.states):
         fidelity[index] = (qt.fidelity(target_state, state))

    return fidelity


def calculate_z_expectation_values(state_evolution, sigma_z_list):    
    #Find minimum difference between expected Z val of last spin and initial Z of first spin
    #calculate expectation value of sz for each spin
    magn_t = np.array([[qt.expect(op, state) 
                        for op in sigma_z_list] 
                       for state in state_evolution.states])
    return magn_t

def get_z_expectation_maximums(n_spins, z_expectation_values):
    '''
    Find maximum in every spin magnetization curve and their indices
    '''
    max_magn_i = [np.max(z_expectation_values[:,i]) for i in range(n_spins)]
    max_magn_i_index = [np.argmax(z_expectation_values[:,i]) for i in range(n_spins)]

    return max_magn_i, max_magn_i_index


def calculate_concurrence(n_spins, state_evolution):
    '''
    Calculate concurrence of state evolution (used for Marta's suggestion of 2-way transport
    for ladder cluster states)
    '''
    concurrence = []
    for index, state in enumerate(state_evolution):
        rho_1 = state.ptrace([0, n_spins-1])
        concurrence.append(qt.concurrence(rho_1))
    return concurrence


