# Numerics
import qutip as qt
import numpy as np

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def standard_J(size, factor):
    strengths = np.zeros(size-1)
    for i in range(size-1):
        strengths[i] = 0.5*factor*np.sqrt((i+1)*(size-i-1))
    return strengths

#couplings in statndard encoding
def domain_wall_J(size, factor):
    strengths = np.zeros(size)
    for i in range(0,size):
        strengths[i] = 0.5*factor*np.sqrt((i+1)*(size-i))
    return strengths

#couplings in domain wall
def tn_definition(data_j):
    size = len(data_j)
    out = np.zeros(size)
    for i in range(size):
        out[i] = -data_j[i]
    return out


class Hamiltonian:
    '''Hamiltonian object. It is first initialized, then chosen the type 
        (standard,forward, backward)    
    '''
    def __init__(self, system_size, mode, lambda_factor, register_size, global_J=None):
        '''
        Parameter description
        
        :system_size:(int) Length of chain
        :mode:(str) Type of Hamiltonian (standard, forward, backward)
        :lambda_factor:(float) Global prefactor that controls error and simulation speed
        :global_J:(float) Domain wall coupling
        :register_size:(int) Size of Bob's register (needed only for backward case)
        
        '''
        self.n_spins = system_size
        self.lambda_factor = lambda_factor
        self.J = global_J
        self.register_size = register_size
        self.mode = mode
        self.sx_list, self.sy_list, self.sz_list = self._initialize_operators()
        self.ham = self._build_hamiltonian()

    
    def _initialize_operators(self, n_spins):
        '''Setup operators for individual qubits
           for each value of i it puts the paulis in different positions of the list, 
           then does IxIxI...sigma_ixIxI...xI
        '''
        sx_list, sy_list, sz_list = [], [], []
        for i in range(n_spins):
            #list of 2x2 identity matrices
            op_list = [qt.qeye(2)] * n_spins
            #replace i-th element with sigma_x
            op_list[i] = sx
            #create matrices of 2^Nx2^N
            sx_list.append(qt.tensor(op_list))
            #do the same for sigma_y and sigma_z
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list
    
    def _build_hamiltonian(self):
        '''Create a different type of hamiltonian depending on the string passed'''

        if self.mode == "standard":
            couplings = standard_J(self.n_spins, self.lambda_factor)
            H = self.hamiltonian_standard(couplings)
        else:
            couplings = domain_wall_J(self.n_spins, self.lambda_factor)
            tn = tn_definition(couplings)
            
            if self.mode == "forward":
                H = self.hamiltonian_forward(tn)
            elif self.mode == "backward":
                H = self.hamiltonian_backward()

        return H
    

    def hamiltonian_standard(self, couplings):
        '''
        Build Hamiltonian in standard encoding
        :couplings:(list(float)) coupling strength between every qubit pair of the chain
        '''
        Ham = 0
        # Interaction terms
        for n in range(self.n_spins - 1):
            Ham += 0.5 * couplings[n] * self.sx_list[n] * self.sx_list[n + 1]
            Ham += 0.5 * couplings[n] * self.sy_list[n] * self.sy_list[n + 1]
        return Ham
    
    def hamiltonian_forward(self, tn):
        '''
        Build Hamiltonian in domain wall encoding in forward configuration
        :tn:(list(float)) transverse field strength in every qubit
        '''
        Ham = 0
        #Transverse field but not in first spin
        for i in range(1, self.n_spins):
            Ham += -tn[i-1] * self.sx_list[i]
        #Virtual qubit down at end of chain
        Ham+= +self.J*self.sz_list[self.n_spins-1]
        #Interaction terms with the rest of the spins
        for i in range(0, self.n_spins-1):
            Ham += self.J* self.sz_list[i]*self.sz_list[i+1]
    
        return Ham

    def hamiltonian_backward(self, tn):
        '''
        Build Hamiltonian in domain wall encoding in forward configuration
        :tn:(list(float)) transverse field strength in every qubit
        '''
        Ham = 0
        #Transverse field but not in Bob's register
        for i in range(0, self.n_spins - self.register_size):
            Ham += -tn[i] * self.sx_list[i]
        #Virtual qubit down at start of chain
        Ham += +self.J*self.sz_list[0]
        #Interaction terms with the rest of the spins
        for i in range(0, self.n_spins-1):
            Ham += self.J* self.sz_list[i]*self.sz_list[i+1]

        return Ham
    


