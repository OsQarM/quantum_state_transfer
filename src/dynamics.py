import qutip as qt
import numpy as np
import Hamiltonian as Ham
import model_building as md
import data_handling as dh
import matplotlib.pyplot as plt

#˙

def chain_calibration(initial_chain, H_transport, ti, tf, Nstep, AutoSwitch = True):
    """
    Runs transport with forward hamiltonian, finds point of swithc to be used in the full algorithm

    Args:
        initial_chain: qutip product state of the initial configuration
        H_transport: Hamiltonian class object with the dynamics of the calibration
        ti: Float initial simulation time
        tf: Float final simulation time
        Nsteps: Integer number of timesteps
        AutoSwitch: True: Find minimum ; False, return a hardcoded input (used for testing and debugging)

    Returns:
        step_of_min_z: Step in which the hamiltonian must swtich (i.e transport completed)
        period: time of H swithc, can be used if step size is changed in future simulations
        Both are unique to the ratio J/lambda
    
    """

    result_calibration         = time_evolution                (H_transport, initial_chain, ti, tf, Nstep)
    magnetizations_calibration = calculate_expectation_values(result_calibration, H_transport)

    # lazy method. Needs refining
    if AutoSwitch == False:
        step_of_min_z = 460
    elif AutoSwitch == True:
        # Different options depending on the type of state sent (To do: Find a unifying method for an arbitraty state)

        step_of_min_z = max(int(np.argmin(magnetizations_calibration["Sz"][:,-1])),10)
        #step_of_min_z = max(int(np.argmin(magnetizations_calibration[:,-3])),10)
        #step_of_min_z = max(int(Nstep//2 + np.argmax(magnetizations_calibration[Nstep//2:,-2])),10)
        #step_of_min_z = np.argsort(np.abs(magnetizations_calibration[:,-1]))[1]

    period = (tf - ti)*step_of_min_z/Nstep

    return step_of_min_z, period



def LightweightAlgorithm(initial_chain, final_chain, ti, period, Nstep, H_transport, H_reset = None):
    '''
    Version of the transporr protocol where only the last point of fidelity is calculated. Used for loops with 
    a lot of simulations
    '''
    result_transport =  time_evolution(H_transport, initial_chain, ti, period, int(Nstep))

    if H_reset:
        result_reset = time_evolution (H_reset, result_transport.states[-1], ti, period, int(Nstep))
        #fidelity = calculate_full_fidelity(result_reset, final_chain)
        fidelity = qt.fidelity(result_reset.states[-1],final_chain)
    else:
        #fidelity = calculate_full_fidelity(result_transport, final_chain)
        fidelity = qt.fidelity(result_transport.states[-1],final_chain)
    
    return fidelity



def TwoStepAlgorithm(initial_chain, final_chain, H_transport, H_reset, ti, period, Nstep, factor = 1.0):
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
        factor: Has to be >= 1, extends simulation of reset to better see the evolution after transport time

    Returns:
        total_full_fidelity: Array of fidelities between simulated states and target state
        magnetizations: Array of expectation values of sigma_z for each spin at each timestep

    """

    result_transport         = time_evolution                (H_transport, initial_chain, ti, period, int(Nstep))
    full_fidelity_transport  = calculate_full_fidelity       (result_transport, final_chain)
    magnetizations_transport = calculate_expectation_values  (result_transport, H_transport)

    middle_chain = result_transport.states[-1]

    result_reset         = time_evolution                (H_reset, middle_chain, ti, period*factor, int(Nstep*factor)) #Additional time for validation and aesthetics
    full_fidelity_reset  = calculate_full_fidelity       (result_reset, final_chain)
    magnetizations_reset = calculate_expectation_values  (result_reset, H_reset)

    #Merge results
    total_full_fidelity = np.concatenate((full_fidelity_transport, full_fidelity_reset), axis=0)
    magnetizations = {key: np.concatenate([magnetizations_transport[key], magnetizations_reset[key]], axis=0) for key in magnetizations_transport}

    return total_full_fidelity, magnetizations


def OneStepAlgorithm(initial_chain, final_chain, H_transport, ti, period, Nstep, factor = 1.0, N = None, correction = None, H_correction= None):
    """
    
    Runs the 2-step protocol to achieve quantum transport with domain walls

    Args:
        initial_chain: qutip product state of the initial configuration
        final chain: qutip product state of the target state
        H_transport: Hamiltonian class object with the dynamics of the first step
        ti: Float initial simulation time
        tf: Float final simulation time
        Nsteps: Integer number of timesteps

    Returns:
        total_full_fidelity: Array of fidelities between simulated states and target state
        magnetizations: Array of expectation values of sigma_z for each spin at each timestep

    """

    # Create DW registers and whole systems for initial and target state
    simulation_result = time_evolution                  (H_transport, initial_chain, ti, period*factor, int(Nstep*factor))
    full_fidelity     = calculate_full_fidelity_standard(simulation_result, final_chain)
    magnetizations    = calculate_expectation_values  (simulation_result, H_transport)

    if correction == "Full":
        corrected_fidelities = calculate_corrections(simulation_result.states, H_correction, ti, final_chain, Nstep//10)

    # IN PROGRESS
    #elif correction == "End":
    #    #calculate relative phase of last state
    #    relative_phase, correction_time = find_correction_parameters(H_transport.J, H_transport.lambda_factor)
    #    #duplicate list of fidelities, but correct only the last one
    #    corrected_fidelities = full_fidelity
    #    corrected_fidelities[-1] = calculate_corrections([simulation_result.states[-1]], H_correction, ti, final_chain, Nstep, tf=correction_time)[-1]

    elif correction == None:
        corrected_fidelities = full_fidelity
  

    return full_fidelity, magnetizations, corrected_fidelities


def find_correction_parameters(J, lmd):
    phase = divmod(2*np.pi*J/lmd, 2*np.pi)[1]
    time = phase/J
    return phase, time


def calculate_corrections(simulation, Hamiltonian, ti, target, Nstep):
    '''
    Add phase correction to each timestep of the chain evolution 
    '''
    corrected_fidelities = []

    #Get period of oscillations (technically, each step could be evolved under a different time, the one it takes it to reach the maximum, which is deterministic)
    tf = np.pi/Hamiltonian.J
    #iterate for each state, and evolve under correction hamiltonian (ZZ)
    for state in simulation:
       corrected_result = time_evolution (Hamiltonian, state, ti, tf, Nstep)
       #calculate fidelities and maximums
       corrected_fidelity = calculate_full_fidelity_standard(corrected_result, target)
       corrected_fidelities.append(max(corrected_fidelity))
    

    return corrected_fidelities

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

def calculate_full_fidelity_standard(state_evolution, target_state, N = None): 
    '''
    Calculates fidelity between every simulated time step and a target state that we have
    previously defined

    :state_evolution:(array(np.states)) simulated state dynamics
    :target_state:(np.state) reference state for validation
    '''
    fidelity = np.zeros(len(state_evolution.times))
    for index, state in enumerate(state_evolution.states):
    # Apply phase correction and then calculate fidelity
    # DOESN'T WORK YET (or hamiltonian dynamics are wrong)
        #phase_correction = (-1j)**(N-1)
        #overlap = target_state.dag()*state
        #amplitude = phase_correction*overlap
        #fidelity[index] = amplitude*np.conj(amplitude)
        fidelity[index] = (qt.fidelity(target_state, state))


    return fidelity


def calculate_expectation_values(state_evolution, Hamiltonian):
    """
    Calculate expectation value of spin X,Y,Z on time evolution of chain

    Args:
        state_evolution: result of qutip.sesolve of length Nsteps
        Hamiltonian: qutip object containing properties of transport Hamiltonian

    Returns:
        magn_t: Dictionary containing result of each observable for each qubit and each timestep
    """   

    sigma_x_list = Hamiltonian.sx_list
    sigma_y_list = Hamiltonian.sy_list
    sigma_z_list = Hamiltonian.sz_list

    magn_t = {}

    #calculate expectation value of sz for each spin
    magn_t["Sx"] = calculate_observable_along_chain(state_evolution, sigma_x_list)
    magn_t["Sy"] = calculate_observable_along_chain(state_evolution, sigma_y_list)
    magn_t["Sz"] = calculate_observable_along_chain(state_evolution, sigma_z_list)

    return magn_t

def calculate_observable_along_chain(state_evolution, observable):
    """
    Calculates a given observable over every spin of the chain and every timestep of simulation

    Args:
    state_evolution: result of qutip.sesolve of length Nsteps
    observable: list of operators where each index applies an observable on a chain qubit

    Returns: 
    2D array of the results for each timestep and qubit
    """
    return np.array([[qt.expect(op, state) 
                        for op in observable] 
                       for state in state_evolution.states])



def get_expectation_maximums(n_spins, expectation_values):
    '''
    Find maximum in every spin magnetization curve and their indices
    '''
    z_expectation_values = expectation_values["Sz"]

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


