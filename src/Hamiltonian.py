# Numerics
import qutip as qt
import numpy as np
import errors

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


#couplings in statndard encoding
def mirror_symmetric_terms(size, factor):
    strengths = np.zeros(size)
    for i in range(0,size):
        strengths[i] = -0.5*factor*np.sqrt((i+1)*(size-i))
    return strengths

class Hamiltonian:
    '''Hamiltonian object. It is first initialized, then chosen the type 
        (standard,forward, backward)    
    '''
    def __init__(self, system_size, mode, lambda_factor, register_size=None, global_J=None,
                 j_error = None, z_error = None, l_error = None, correction = None):
        '''
        Args:
        
        system_size:(int) Length of chain
        mode:(str) Type of Hamiltonian (standard, forward, backward)
        lambda_factor:(float) Global prefactor that controls error and simulation speed
        global_J:(float) Domain wall coupling (not needed for standard)
        register_size:(int) Size of Bob's register (needed only for backward case)
        correction: debug parameter to test phase correction
        
        '''
        self.n_spins = system_size
        self.lambda_factor = lambda_factor
        self.J = global_J
        self.register_size = register_size
        self.mode = mode
        self.j_err = j_error
        self.l_err = l_error
        self.z_err = z_error
        self.correction = correction

        self.sx_list, self.sy_list, self.sz_list = self._initialize_operators()
        self.couplings = self._calculate_couplings()
        self.ham = self._build_hamiltonian()
    
    def _initialize_operators(self):
        '''Setup operators for individual qubits
           for each value of i it puts the paulis in different positions of the list, 
           then does IxIxI...sigma_ixIxI...xI
        '''
        sx_list, sy_list, sz_list = [], [], []
        for i in range(self.n_spins):
            #list of 2x2 identity matrices
            op_list = [qt.qeye(2)] * self.n_spins
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
            H = self.hamiltonian_standard()
        elif self.mode == "transport":
            H = self.hamiltonian_transport()
        elif self.mode == "reset":
            H = self.hamiltonian_reset()
        elif self.mode == "correction":
            H = self.hamiltonian_correction()
        return H
    

    def _calculate_couplings(self): 
        """
        Calculate couplings for lambda, J and z fields for different situations (standard or domain walls)
        If errors are zeros we avoid calling the errors.py script

        Returns: 
            couplings: Dictionary containing all necesary values for the hamiltonian, indexed by their name
                       with errors added if requested

        """
        #Define all Hamiltonian couplings as dictionary
        couplings = {}

        #Calculate ideal mirror symmetry
        length = self.n_spins - (self.register_size if self.mode == "reset" else 1)
        error_free_l = mirror_symmetric_terms(length, self.lambda_factor)

        #errors in transverse fields
        if self.l_err and self.l_err != 0.0:
            couplings["lambda"] = errors.apply_gaussian_rel_error(error_free_l, self.l_err)
        else:
            couplings["lambda"] = error_free_l

        #errors in domain wall couplings
        if self.J:  #not used in standard encoding
            error_free_j = [self.J]*(self.n_spins -1)
            #J at ends without error, correspond to local fields at chain ends
            if self.j_err and self.j_err != 0.0:
                couplings["J"] = [self.J] + errors.apply_gaussian_rel_error(error_free_j, self.j_err) + [self.J]
            else:
                couplings["J"] = 2*[self.J] + error_free_j

        if self.z_err and self.z_err != 0.0:
            couplings["z"] = errors.apply_gaussian_abs_error([0]*self.n_spins, self.z_err)
        else:
            couplings["z"] = [0]*self.n_spins

        return couplings


    def hamiltonian_standard(self):
        '''
        Build Hamiltonian in standard encoding
        :couplings:(list(float)) coupling strength between every qubit pair of the chain
        '''
        Ham = 0

        l_terms = self.couplings["lambda"]
        z_terms = self.couplings["z"]

        # Interaction terms (not in last spin)
        for i in range(self.n_spins - 1):
            #dynamics
            Ham += 0.5 * l_terms[i] * self.sx_list[i] * self.sx_list[i + 1]
            Ham += 0.5 * l_terms[i] * self.sy_list[i] * self.sy_list[i + 1]
        
        #phase correction (for 1 and 0 excitation subspaces)
        if self.correction == True:
            for i in range(self.n_spins):
                # - sign important!!
                Ham += 0-self.lambda_factor*(1*self.n_spins)/(4) * self.sz_list[i]

        #All-to-all phase correction
        # h2 = - self.lambda_factor / 4.0
        # for i in range(self.n_spins):
        #     for j in range(i + 1, self.n_spins):
        #         Ham += h2 * self.sz_list[i] * self.sz_list[j]


        #residual z fields
        for i in range(self.n_spins):
            Ham += z_terms[i] * self.sz_list[i]

        return Ham
    
    def hamiltonian_transport(self):
        '''
        Build Hamiltonian in domain wall encoding in forward configuration
        :tn:(list(float)) transverse field strength in every qubit
        '''
        Ham = 0

        l_terms = self.couplings["lambda"]
        z_terms = self.couplings["z"]
        j_terms = self.couplings["J"]

        if self.correction == True:
            corr_term = self.lambda_factor*self.n_spins/4.0
        else:
            corr_term = 0

        #Transverse field but not in first spin
        for i in range(1, self.n_spins):
            Ham += -l_terms[i-1] * self.sx_list[i]

        #Virtual qubit down at end of chain
        Ham+= (j_terms[-1] - corr_term)*self.sz_list[self.n_spins-1]

        #Interaction terms with the rest of the spins except for the last one
        for i in range(0, self.n_spins-1):
            Ham += (j_terms[i] - corr_term)*self.sz_list[i]*self.sz_list[i+1]
        
        #Residual z fields
        for i in range(0,self.n_spins):
            Ham += z_terms[i] * self.sz_list[i]

        return Ham

    def hamiltonian_reset(self):
        '''
        Build Hamiltonian in domain wall encoding in forward configuration
        :tn:(list(float)) transverse field strength in every qubit
        '''
        Ham = 0

        l_terms = self.couplings["lambda"]
        z_terms = self.couplings["z"]
        j_terms = self.couplings["J"]

        #Transverse field but not in Bob's register
        for i in range(0, self.n_spins - self.register_size):
            Ham += -l_terms[i] * self.sx_list[i]

        #Virtual qubit down at start of chain
        Ham += +j_terms[1]*self.sz_list[0]

        #Interaction terms with the rest of the spins
        for i in range(0, self.n_spins-1):
            Ham += j_terms[i]* self.sz_list[i]*self.sz_list[i+1]
        
        #Residual z fields
        for i in range(0,self.n_spins):
            Ham += z_terms[i] * self.sz_list[i]

        return Ham
    
    def hamiltonian_correction(self):
        '''
        Build Hamiltonian for correcting relative phase after domain wall evolution
        '''
        Ham = 0
        j_terms = self.couplings["J"]

        #Relative phase correction
        for i in range(0, self.n_spins-1):
            Ham += j_terms[1]*self.sz_list[i]*self.sz_list[i+1]
        
        return Ham
    


def build_hamiltonians(N, lmd, J, reg_size, j_err = 0, l_err = 0, z_err = 0):

    H_transport = Hamiltonian(system_size = N,
                        mode = "transport",
                        lambda_factor = lmd,
                        global_J = J,
                        j_error = j_err,
                        l_error = l_err,
                        z_error = z_err
                        )
    H_reset     = Hamiltonian(system_size = N,
                        mode = "reset",
                        lambda_factor = lmd,
                        register_size = reg_size,
                        global_J = J,
                        j_error = j_err,
                        l_error = l_err,
                        z_error = z_err 
                        )

    return H_transport, H_reset
