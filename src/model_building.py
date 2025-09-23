# Numerics
import qutip as qt
import numpy as np

ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)

### STATE GENERATION FOR DIRECT STATES

def initialize_1_qubit_system(size, alpha, beta, register): 
    '''
    Generate chain in state |0> with either first or last spin 
    in a superposition of |0> and |1> 
    
    :size:(int) Length of chain
    :alpha, beta:(complex) weights of 0 and 1 in end spin
    :register:(str) Either "Alice" or "Bob", puts state either at begining or end of chain

    '''
    end_spin = alpha*qt.basis(2,0) + beta*qt.basis(2,1)
    norm = np.sqrt(end_spin.dag()*end_spin)
    end_spin = end_spin/norm

    #put spin at chosen end depending on register
    if register == "Alice":
        state_list = [end_spin] + [qt.basis(2, 0)] * (size - 1)
    elif register == "Bob":
        state_list =  [qt.basis(2, 0)] * (size - 1) + [end_spin]
    #Total state is product state of spins
    chain_state = qt.tensor(state_list) 

    return chain_state


### STATE GENERATION FOR DOMAIN WALLS

def crate_domain_wall_state(state_dictionary, register):
    '''
    Translates superposition of states into domain wall state of a register (Alice or Bob)

    :state_dictionary:(dict) Components of state and weights 
                             example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
    :register:(str) Either "Alice" or "Bob", puts state either at begining or end of chain 

    '''
    state = 0
    #loop over states in the dictionary (components)
    for i, term in enumerate(state_dictionary.keys()):
        dw_spins = []
        #first bit of state is always a down wire qubit
        previous_spin = "0"

        #Invert order if building state in Alice's register (build starting from last qubit)
        if register == "Alice":
            loop_list = term[::-1]
        elif register == "Bob":
            loop_list = term

        #loop over bits
        for bit in loop_list:
            #put 0 or 1 to create (or not) domain wall and update previous_spin variable
            if bit == "1":
                if previous_spin == "0":
                    dw_spins.append(ket1)
                    previous_spin = "1"
                elif previous_spin == "1":
                    dw_spins.append(ket0)
                    previous_spin = "0"
            elif bit == "0":
                if previous_spin == "0":
                    dw_spins.append(ket0)
                    previous_spin = "0"
                elif previous_spin == "1":
                    dw_spins.append(ket1)
                    previous_spin = "1"

        #reverse order of list and make tensor product
        dw_term = qt.tensor(dw_spins[::-1])
        #add to full state with corresponding weight
        state += state_dictionary[term]*dw_term
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
    if abs(1 - abs(dw_state.dag()*dw_state)) > 0.0001:
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


### SOON TO BE DEPRECATED

# def generate_target(size, alpha, beta): 
#     #Initial state
#     #Initialize first spin in a superposition of |0> and |1>
#     #Initialize the rest of the spins in |0>
#     t_i = time.time()
#     psi0_1 = alpha*qt.basis(2,0) + beta*qt.basis(2,1)
#     state_list =  [qt.basis(2, 0)] * (size - 1) + [psi0_1]

#     #Total state is product state of spins
#     final_state = qt.tensor(state_list) 

#     t_f = time.time()

#     print("State initialization:", t_f - t_i)

#     print("Initializing system of size", size, "\n"
#           "First spin in state {:.4f} |0> + {:.4f} |1>\n".format(alpha, beta))
    
#     return final_state

###

### SOON TO BE DEPRECATED

# def generate_target(size, bob_register): 
#     #check that final state is a valid one
#     if abs(1 - abs(bob_register.dag()*bob_register)) > 0.0001:
#         print("Initial state is not valid")
#         return
    
#     bob_reg_len = len(bob_register.dims[0])
#     #check that chain is long enough to do transport
#     if bob_reg_len*2 >= size:
#         print("Chain too short to transport initial state")
#         return
    
     
#     chain = [qt.basis(2, 0)]*(size - bob_reg_len)
#     chain_state = qt.tensor(chain)

#     #Total state is product state of spins
#     final_state = qt.tensor(chain_state, bob_register) 
    
#     return final_state

###

### SOON TO BE DEPRECATED

# def create_domain_wall_target(state_dictionary):
#     # example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
#     state = 0
#     for i, term in enumerate(state_dictionary.keys()):
#         dw_spins = []
#         #first bit is always a down wire qubit
#         previous_spin = "0"
#         #loop over bits (w/o reversing order this time)
#         for bit in term:
#             #put 0 or 1 to create (or not) domain wall
#             if bit == "1":
#                 if previous_spin == "0":
#                     dw_spins.append(ket1)
#                     previous_spin = "1"
#                 elif previous_spin == "1":
#                     dw_spins.append(ket0)
#                     previous_spin = "0"
#             elif bit == "0":
#                 if previous_spin == "0":
#                     dw_spins.append(ket0)
#                     previous_spin = "0"
#                 elif previous_spin == "1":
#                     dw_spins.append(ket1)
#                     previous_spin = "1"
#         #reverse order of list and make tensor product
#         dw_term = qt.tensor(dw_spins[::-1])
#         #add to full state with corresponding weight
#         state += state_dictionary[term]*dw_term
#     # normalize
#     norm = np.sqrt(state.dag()*state)
#     return state/norm

###


