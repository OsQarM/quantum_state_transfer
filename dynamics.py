import qutip as qt
import numpy as np
import Hamiltonian as Ham
import model_building as md
import data_handling as dh


def TwoStepAlgorithm(initial_chain, final_chain, H_transport, H_reset, ti, tf, Nstep, AutoSwitch = False):
    """
    
    Runs the 2-step protocol to achieve quantum transport with domain walls

    Args:
        initial_chain: qutip product state of the initial configuration
        final chain: qutip product state of the target state
        H_transport: Hamiltonian class object with the dynamics of the first step
        H_reset: Hamiltonian class object with the dynamics of the second step
        ti: Float initial simulation time
        tf: Float final simulation time
        Nsteps: Integer number of timesteps

    Returns:
        total_full_fidelity: Array of fidelities between simulated states and target state
        magnetizations: Array of expectation values of sigma_z for each spin at each timestep

    """

    # Create DW registers and whole systems for initial and target state
    result_calibration         = time_evolution                (H_transport, initial_chain, ti, tf, Nstep)
    full_fidelity_calibration  = calculate_full_fidelity       (result_calibration, final_chain)
    magnetizations_calibration = calculate_z_expectation_values(result_calibration, H_transport.sz_list)

    # lazy method. Needs refining
    if AutoSwitch == False:
        step_of_min_z = 460
    elif AutoSwitch == True:
        step_of_min_z = max(int(np.argmin(magnetizations_calibration[:,-1])),10)

    period = (tf - ti)*step_of_min_z/Nstep

    result_transport         = time_evolution                (H_transport, initial_chain, ti, period, int(step_of_min_z))
    full_fidelity_transport  = calculate_full_fidelity       (result_transport, final_chain)
    magnetizations_transport = calculate_z_expectation_values(result_transport, H_transport.sz_list)

    middle_chain = result_transport.states[-1]

    result_reset         = time_evolution                (H_reset, middle_chain, ti, period*1.12, int(step_of_min_z*1.12)) #Additional time for validation and aesthetics
    full_fidelity_reset  = calculate_full_fidelity       (result_reset, final_chain)
    magnetizations_reset = calculate_z_expectation_values(result_reset, H_reset.sz_list)

    total_full_fidelity = np.concatenate((full_fidelity_transport, full_fidelity_reset), axis=0)
    magnetizations      = np.concatenate((magnetizations_transport, magnetizations_reset), axis=0)

    return total_full_fidelity, magnetizations

def time_evolution(Ham, initial_state, initial_time, final_time, timesteps, progessbar = None):
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

    options = {'method': 'adams'}

    if progessbar:
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


