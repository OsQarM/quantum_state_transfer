# Numerics
import qutip as qt
import numpy as np
import Hamiltonian as Ham

ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)


### STATE GENERATION FOR DOMAIN WALLS

def create_domain_wall_state(state_dictionary, register, one_step = False):
    '''
    Translates superposition of states into domain wall state of a register (Alice or Bob)

    Args:
        state_dictionary:(dict) Components of state and weights 
                             example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
        register:(str) Either "Alice" or "Bob", puts state either at begining or end of chain 
    
    Returns:
        Normalized DW state as product state of qutip objects

    '''
    state = 0
    #loop over states in the dictionary (components)
    for i, term in enumerate(state_dictionary.keys()):
        dw_spins = []
        #first bit of state is always a down wire qubit
        previous_spin = "0"

        #Add auxiliary qubit if necessary
        aux_term = append_auxiliary_qubit(term, one_step)

        #Invert order if building state in Alice's register (build starting from last qubit, the one touching the chain)
        if register == "Alice":
            loop_list = aux_term[::-1]
        elif register == "Bob":
            loop_list = aux_term

        for bit in loop_list:
            #put 0 or 1 to create (or not) domain wall and update previous_spin variable
            dw_spins, previous_spin = append_bit(bit, dw_spins, previous_spin)

        #reverse order of list and make tensor product (for Alice restores order, for Bob, reverts it (mirror image))
        dw_term = qt.tensor(dw_spins[::-1])
        #add to full state with corresponding weight
        state += state_dictionary[term]*dw_term

    # normalize
    norm = np.sqrt(state.dag()*state)
    return state/norm

def append_auxiliary_qubit(state, one_step):
        '''
        Add an extra qubit to the spin picture state if we do the one-step protocol
        '''
        #for one-step protocol only
        if one_step == True:
            #count number of domain walls
            excitation_number = state.count("1")
            #if odd, make it even using auxiliary qubit 
            if excitation_number % 2 != 0:
                new_term = state + "1"
            else:
                new_term = state + "0"
        else:
            # don't do anything to the variable
            new_term = state

        return new_term


def append_bit(bit, spin_list, previous_spin):
    """
    Adds bit to chain to create (or not) domain wall.
    If we encounter a 1 (DW) we need to pu a spin in the opposite orientation of the previous one
    If we encounter a 0, we put it in the same orientation as the previous one
    Update previous spin after adding it
    """
    if bit == "1":
        if previous_spin == "0":
            spin_list.append(ket1)
            previous_spin = "1"
        elif previous_spin == "1":
            spin_list.append(ket0)
            previous_spin = "0"
    elif bit == "0":
        if previous_spin == "0":
            spin_list.append(ket0)
            previous_spin = "0"
        elif previous_spin == "1":
            spin_list.append(ket1)
            previous_spin = "1"

    return spin_list, previous_spin


### STATE GENERATION FOR DIRECT STATES

def create_standard_state(state_dictionary, register):
    '''
    Translates superposition of states into domain wall state of a register (Alice or Bob)

    :state_dictionary:(dict) Components of state and weights 
                             example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
    :register:(str) Either "Alice" or "Bob", puts state either at begining or end of chain 

    '''
    state = 0
    #loop over states in the dictionary (components)
    for i, term in enumerate(state_dictionary.keys()):
        spins = []
        #Invert order if building state in Alice's register (build starting from last qubit)
        if register == "Alice":
            loop_list = term
        elif register == "Bob":
            loop_list = term[::-1]
        #loop over bits
        for bit in loop_list:
            #put 0 or 1 to create (or not) domain wall and update previous_spin variable
            if bit == "1":
                spins.append(ket1)

            elif bit == "0":
                spins.append(ket0)
        #make tensor product
        st_term = qt.tensor(spins)
        #add to full state with corresponding weight
        state += state_dictionary[term]*st_term
    # normalize
    norm = np.sqrt(state.dag()*state)
    return state/norm


def initialize_general_system(size, dw_state, register):
    '''
    Builds full chain with a domain wall state at an end

    :size:(int) Total length of chain
    :dw_state:(qutip.state) state in domain wall encoding to be appended
    :register:(str) Either "Alice" or "Bob", puts state either at begining or end of chain

    '''
    #check that initial state is a valid one
    if abs(1 - abs(dw_state.dag()*dw_state)) > 1e-7:
        print("Initial state is not valid")
        return
    
    #check that chain is long enough to do transport
    reg_len = len(dw_state.dims[0])
    if reg_len*2 >= size:
        print("Chain too short to transport initial state")
        return

    chain = [qt.basis(2, 0)]*(size - reg_len)
    chain_state = qt.tensor(chain)

    #Total state is product state of spins
    if register == "Alice":
        initial_state = qt.tensor(dw_state, chain_state)   
    elif register == "Bob":
        initial_state = qt.tensor(chain_state, dw_state) 
     
    return initial_state




def initialize_system(state_dictionary, N, encoding = 'dw', one_step = False):
    if encoding == 'dw':
        initial_state = create_domain_wall_state(state_dictionary, register='Alice', one_step=one_step)
        final_state   = create_domain_wall_state(state_dictionary, register='Bob', one_step=one_step)
    elif encoding == 'st':
        initial_state = create_standard_state(state_dictionary, register='Alice')
        final_state   = create_standard_state(state_dictionary, register='Bob')
    else:
        raise(ValueError('Encoding not properly defined. Pass either dw (domain wall) or st (standard)'))


    initial_chain = initialize_general_system(N, initial_state, register='Alice')
    final_chain   = initialize_general_system(N, final_state, register='Bob')

    register_size = len(initial_state.dims[0])
    
    return initial_chain, final_chain, register_size


