# Numerics
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


ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)

### STATE GENERATION FOR DIRECT STATES

def initialize_system(size, alpha, beta): 
    #Initial state
    #Initialize first spin in a superposition of |0> and |1>
    #Initialize the rest of the spins in |0>
    t_i = time.time()
    psi0_1 = alpha*qt.basis(2,0) + beta*qt.basis(2,1)
    state_list = [psi0_1] + [qt.basis(2, 0)] * (size - 1)

    #Total state is product state of spins
    initial_state = qt.tensor(state_list) 

    t_f = time.time()

    print("State initialization:", t_f - t_i)

    print("Initializing system of size", size, "\n"
          "First spin in state {:.4f} |0> + {:.4f} |1>\n".format(alpha, beta))
    
    return initial_state


def generate_target(size, alpha, beta): 
    #Initial state
    #Initialize first spin in a superposition of |0> and |1>
    #Initialize the rest of the spins in |0>
    t_i = time.time()
    psi0_1 = alpha*qt.basis(2,0) + beta*qt.basis(2,1)
    state_list =  [qt.basis(2, 0)] * (size - 1) + [psi0_1]

    #Total state is product state of spins
    final_state = qt.tensor(state_list) 

    t_f = time.time()

    print("State initialization:", t_f - t_i)

    print("Initializing system of size", size, "\n"
          "First spin in state {:.4f} |0> + {:.4f} |1>\n".format(alpha, beta))
    
    return final_state


### STATE GENERATION FOR DOMAIN WALLS

def crate_domain_wall_state(state_dictionary, register):
    # example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
    state = 0
    for i, term in enumerate(state_dictionary.keys()):
        dw_spins = []
        #first bit is always a down wire qubit
        last_spin = "0"

        #Invert order if building state i Alice's register (build starting from last qubit)
        if register == "Alice":
            loop_list = term[::-1]
        elif register == "Bob":
            loop_list = term

        #loop over bits
        for bit in loop_list:
            #put 0 or 1 to create (or not) domain wall
            if bit == "1":
                if last_spin == "0":
                    dw_spins.append(ket1)
                    last_spin = "1"
                elif last_spin == "1":
                    dw_spins.append(ket0)
                    last_spin = "0"
            elif bit == "0":
                if last_spin == "0":
                    dw_spins.append(ket0)
                    last_spin = "0"
                elif last_spin == "1":
                    dw_spins.append(ket1)
                    last_spin = "1"
        #reverse order of list again and make tensor product
        dw_term = qt.tensor(dw_spins[::-1])
        #add to full state with corresponding weight
        state += state_dictionary[term]*dw_term
    # normalize
    norm = np.sqrt(state.dag()*state)
    return state/norm


def create_domain_wall_target(state_dictionary):
    # example: |psi> = ["001":0.4,"101":0.2,"111":0.6,"110":0.35]
    state = 0
    for i, term in enumerate(state_dictionary.keys()):
        dw_spins = []
        #first bit is always a down wire qubit
        last_spin = "0"
        #loop over bits (w/o reversing order this time)
        for bit in term:
            #put 0 or 1 to create (or not) domain wall
            if bit == "1":
                if last_spin == "0":
                    dw_spins.append(ket1)
                    last_spin = "1"
                elif last_spin == "1":
                    dw_spins.append(ket0)
                    last_spin = "0"
            elif bit == "0":
                if last_spin == "0":
                    dw_spins.append(ket0)
                    last_spin = "0"
                elif last_spin == "1":
                    dw_spins.append(ket1)
                    last_spin = "1"
        #reverse order of list and make tensor product
        dw_term = qt.tensor(dw_spins[::-1])
        #add to full state with corresponding weight
        state += state_dictionary[term]*dw_term
    # normalize
    norm = np.sqrt(state.dag()*state)
    return state/norm


def initialize_general_system(size, alice_register): # alice_register is already in domain wall encoding
    #initialize chain
    #check that initial state is a valid one
    if abs(1 - abs(alice_register.dag()*alice_register)) > 0.0001:
        print("Initial state is not valid")
        return
    
    alice_reg_len = len(alice_register.dims[0])
    #check that chain is long enough to do transport
    if alice_reg_len*2 >= size:
        print("Chain too short to transport initial state")
        return
     
    chain = [qt.basis(2, 0)]*(size - alice_reg_len)
    chain_state = qt.tensor(chain)

    #Total state is product state of spins
    initial_state = qt.tensor(alice_register, chain_state) 
    
    return initial_state


def generate_target(size, bob_register): 
    #check that final state is a valid one
    if abs(1 - abs(bob_register.dag()*bob_register)) > 0.0001:
        print("Initial state is not valid")
        return
    
    bob_reg_len = len(bob_register.dims[0])
    #check that chain is long enough to do transport
    if bob_reg_len*2 >= size:
        print("Chain too short to transport initial state")
        return
     
    chain = [qt.basis(2, 0)]*(size - bob_reg_len)
    chain_state = qt.tensor(chain)

    #Total state is product state of spins
    final_state = qt.tensor(chain_state, bob_register) 
    
    return final_state



